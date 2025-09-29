"""
Online ASR with Punctuation, Capitalization, and End-of-Utterance Detection
Combines streaming FastConformer ASR with punctuation/capitalization post-processing
and TurnSense EOU detection for better conversation handling.

This script does real-time speech recognition and applies punctuation and capitalization
to make the output more readable, with automatic context reset on end-of-utterance.
"""

import copy
import time
import pyaudio as pa
import numpy as np
import torch
import argparse
import queue
import logging
import os
import sys
from collections import deque

# Add EOU detection imports
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from omegaconf import OmegaConf, open_dict

import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

# Constants
SAMPLE_RATE = 16000  # Hz
ENCODER_STEP_LENGTH = 80  # ms for FastConformer models

# Available ASR models
AVAILABLE_ASR_MODELS = [
    "stt_en_fastconformer_hybrid_large_streaming_multi",
    "stt_en_fastconformer_hybrid_large_streaming_80ms", 
    "stt_en_fastconformer_hybrid_large_streaming_480ms",
    "stt_en_fastconformer_hybrid_large_streaming_1040ms"
]

# Available punctuation models
AVAILABLE_PUNCT_MODELS = [
    "punctuation_en_bert",
    "punctuation_en_distilbert",
]

class EndOfUtteranceDetector:
    def __init__(self, quiet_mode=False):
        """Initialize the TurnSense end-of-utterance detection model"""
        self.quiet_mode = quiet_mode
        self.model_id = "latishab/turnsense"
        self.threshold = 0.8  # Increased threshold to reduce false positives
        
        # Add additional criteria to reduce false positives
        self.min_words_for_eou = 8  # Require at least 8 words before considering EOU
        self.confirmation_needed = 2  # Require 2 consecutive positive detections
        self.recent_detections = []  # Track recent EOU detections
        
        try:
            if not quiet_mode:
                print("Loading TurnSense EOU detection model...")
            
            # Suppress output during model loading in quiet mode
            if quiet_mode:
                with open(os.devnull, 'w') as devnull:
                    old_stderr = sys.stderr
                    old_stdout = sys.stdout
                    sys.stderr = devnull
                    sys.stdout = devnull
                    try:
                        self._load_model()
                    finally:
                        sys.stderr = old_stderr
                        sys.stdout = old_stdout
            else:
                self._load_model()
                
            if not quiet_mode:
                print("TurnSense EOU model loaded successfully")
                
        except Exception as e:
            if not quiet_mode:
                print(f"Warning: Could not load EOU detection model: {e}")
                print("Continuing without EOU detection...")
            self.tokenizer = None
            self.session = None
    
    def _load_model(self):
        """Load the tokenizer and ONNX model"""
        # Download and load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Download and load ONNX model
        model_path = hf_hub_download(repo_id=self.model_id, filename="model_quantized.onnx")
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    def detect_eou(self, text):
        """
        Detect if the given text represents an end of utterance
        
        Args:
            text: Input text to analyze
            
        Returns:
            bool: True if end of utterance is detected, False otherwise
        """
        if not self.tokenizer or not self.session or not text.strip():
            return False
        
        # Check minimum length requirement
        word_count = len(text.strip().split())
        if word_count < self.min_words_for_eou:
            return False
        
        try:
            # Prepare input in the format expected by TurnSense
            formatted_text = f"<|user|> {text.strip()} <|im_end|>"
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_text,
                padding="max_length",
                max_length=256,
                return_tensors="pt",
                truncation=True
            )
            
            # Run inference
            ort_inputs = {
                'input_ids': inputs['input_ids'].numpy(),
                'attention_mask': inputs['attention_mask'].numpy()
            }
            
            # Get probabilities
            probabilities = self.session.run(None, ort_inputs)[0]
            
            # Extract EOU probability (assuming binary classification where index 1 is EOU)
            eou_probability = float(probabilities[0][1]) if len(probabilities[0]) > 1 else float(probabilities[0][0])
            
            # Track recent detections for confirmation
            is_above_threshold = eou_probability > self.threshold
            self.recent_detections.append(is_above_threshold)
            
            # Keep only recent detections (last 3)
            if len(self.recent_detections) > 3:
                self.recent_detections = self.recent_detections[-3:]
            
            # Require multiple consecutive confirmations
            confirmation_count = sum(self.recent_detections[-self.confirmation_needed:])
            is_eou = (confirmation_count >= self.confirmation_needed and 
                     len(self.recent_detections) >= self.confirmation_needed)
            
            # Additional checks for natural sentence endings
            text_lower = text.strip().lower()
            has_natural_ending = any(text_lower.endswith(ending) for ending in [
                '.', '?', '!', '. thank you', '. thanks', 'that\'s it', 'that is it'
            ])
            
            # Only trigger EOU if we have both model confidence AND natural ending indicators
            final_eou = is_eou and (has_natural_ending or eou_probability > 0.9)
            
            if not self.quiet_mode and final_eou:
                print(f"[EOU] Detected end of utterance (confidence: {eou_probability:.3f}, words: {word_count}, confirmations: {confirmation_count})")
            elif not self.quiet_mode and is_above_threshold:
                print(f"[EOU] Potential EOU detected but not confirmed (confidence: {eou_probability:.3f}, words: {word_count})")
            
            print(f"Current EOU probability: {eou_probability}, recent detections: {self.recent_detections}")
            
            # Reset detection history if we actually trigger EOU
            if final_eou:
                self.recent_detections = []
            
            return final_eou
            
        except Exception as e:
            if not self.quiet_mode:
                print(f"EOU detection error: {e}")
            return False

class FrameLevelSpeechDetector:
    def __init__(self, quiet_mode=False):
        """Analyze STT model frame-level outputs to detect speech activity"""
        self.quiet_mode = quiet_mode
        self.frame_history = []  # Store recent frame analysis results
        self.history_length = 50  # Keep last 50 frames (adjust based on frame rate)
        self.silence_threshold_frames = 20  # Number of consecutive low-activity frames for EOU
        self.min_confidence_threshold = 0.1  # Minimum confidence for "speech detected"
        self.speech_activity_threshold = 0.3  # Fraction of frames that need speech for "active"
        
    def analyze_model_outputs(self, pred_outputs, processed_signal_length=None):
        """
        Analyze the model's frame-level outputs to detect speech activity
        
        Args:
            pred_outputs: Raw model outputs (logits/probabilities)
            processed_signal_length: Length of processed signal for normalization
            
        Returns:
            dict: Analysis results including speech_detected, confidence, etc.
        """
        try:
            if pred_outputs is None or len(pred_outputs) == 0:
                return {"speech_detected": False, "confidence": 0.0, "frame_activity": 0.0}
            
            # Handle different output formats
            if isinstance(pred_outputs, list):
                if len(pred_outputs) > 0 and torch.is_tensor(pred_outputs[0]):
                    outputs = pred_outputs[0]  # Take first output if list
                else:
                    return {"speech_detected": False, "confidence": 0.0, "frame_activity": 0.0}
            elif torch.is_tensor(pred_outputs):
                outputs = pred_outputs
            else:
                return {"speech_detected": False, "confidence": 0.0, "frame_activity": 0.0}
            
            # Move to CPU for analysis
            if outputs.is_cuda:
                outputs = outputs.cpu()
            
            # Apply softmax to get probabilities if needed
            if outputs.dtype == torch.float32 or outputs.dtype == torch.float16:
                # Assume these are logits, apply softmax
                probs = torch.softmax(outputs, dim=-1)
            else:
                probs = outputs
            
            # Analyze frame-level activity
            # Look for non-blank predictions (assuming blank token is at index 0 for CTC-like models)
            if probs.dim() >= 2:
                # Shape is likely [time, vocab] or [batch, time, vocab]
                if probs.dim() == 3:
                    probs = probs.squeeze(0)  # Remove batch dimension
                
                # Get maximum probability for each frame (excluding blank if it's at index 0)
                if probs.size(-1) > 1:
                    # Assume blank token is at index 0, get max of non-blank tokens
                    non_blank_probs = probs[:, 1:] if probs.size(-1) > 1 else probs
                    max_non_blank_probs, _ = torch.max(non_blank_probs, dim=-1)
                    
                    # Calculate frame-level speech activity
                    speech_frames = (max_non_blank_probs > self.min_confidence_threshold).float()
                    frame_activity = torch.mean(speech_frames).item()
                    max_confidence = torch.max(max_non_blank_probs).item()
                    avg_confidence = torch.mean(max_non_blank_probs).item()
                    
                else:
                    frame_activity = 0.0
                    max_confidence = 0.0
                    avg_confidence = 0.0
            else:
                # Single frame or unexpected shape
                frame_activity = 0.0
                max_confidence = 0.0
                avg_confidence = 0.0
            
            # Determine if speech is detected
            speech_detected = frame_activity > self.speech_activity_threshold
            
            return {
                "speech_detected": speech_detected,
                "confidence": max_confidence,
                "avg_confidence": avg_confidence,
                "frame_activity": frame_activity,
                "num_frames": probs.size(0) if probs.dim() >= 1 else 1
            }
            
        except Exception as e:
            if not self.quiet_mode:
                print(f"Frame analysis error: {e}")
            return {"speech_detected": False, "confidence": 0.0, "frame_activity": 0.0}
    
    def update_history(self, analysis_result):
        """Update the frame history with new analysis"""
        self.frame_history.append(analysis_result)
        
        # Keep only recent history
        if len(self.frame_history) > self.history_length:
            self.frame_history = self.frame_history[-self.history_length:]
    
    def detect_silence_period(self):
        """
        Detect if we're in a silence period based on recent frame analysis
        
        Returns:
            bool: True if silence period detected (potential EOU)
        """
        if len(self.frame_history) < self.silence_threshold_frames:
            return False
        
        # Look at recent frames
        recent_frames = self.frame_history[-self.silence_threshold_frames:]
        
        # Count frames with low speech activity
        low_activity_count = sum(1 for frame in recent_frames 
                                if not frame.get("speech_detected", True))
        
        # Calculate recent average activity
        recent_activity = sum(frame.get("frame_activity", 0.0) for frame in recent_frames) / len(recent_frames)
        
        # Detect silence if most recent frames show low activity
        silence_ratio = low_activity_count / len(recent_frames)
        is_silence_period = (silence_ratio > 0.7 and recent_activity < 0.2)
        
        if not self.quiet_mode and is_silence_period:
            print(f"[FRAME-EOU] Silence detected: {silence_ratio:.2f} ratio, {recent_activity:.3f} activity")
        
        return is_silence_period
    
    def get_recent_activity_summary(self, frames=10):
        """Get summary of recent speech activity"""
        if len(self.frame_history) < frames:
            return {"avg_activity": 0.0, "speech_frames": 0, "total_frames": 0}
        
        recent = self.frame_history[-frames:]
        avg_activity = sum(f.get("frame_activity", 0.0) for f in recent) / len(recent)
        speech_frames = sum(1 for f in recent if f.get("speech_detected", False))
        
        return {
            "avg_activity": avg_activity,
            "speech_frames": speech_frames,
            "total_frames": len(recent)
        }

class OnlineASRWithPunctuation:
    def __init__(self, asr_model_name, punct_model_name=None, lookahead_size=480, decoder_type="rnnt", 
                 quiet_mode=False, enable_eou=True):
        """
        Initialize the Online ASR system with punctuation and frame-level EOU detection
        
        Args:
            asr_model_name: Name of the pretrained ASR model to use
            punct_model_name: Name of the punctuation model (None to disable)
            lookahead_size: Lookahead size in milliseconds
            decoder_type: "rnnt" or "ctc"
            quiet_mode: If True, suppress all logging
            enable_eou: If True, enable end-of-utterance detection
        """
        self.asr_model_name = asr_model_name
        self.punct_model_name = punct_model_name
        self.lookahead_size = lookahead_size
        self.decoder_type = decoder_type
        self.quiet_mode = quiet_mode
        self.enable_eou = enable_eou
        
        # Comprehensive logging suppression for quiet mode
        if quiet_mode:
            self._setup_quiet_mode()
            
        # Initialize models
        self._setup_asr_model()
        if punct_model_name:
            self._setup_punctuation_model()
        else:
            self.punct_model = None
            
        # Initialize EOU detection
        if enable_eou:
            self.eou_detector = EndOfUtteranceDetector(quiet_mode)
        else:
            self.eou_detector = None
            
        # Initialize frame-level speech detector
        self.frame_detector = FrameLevelSpeechDetector(quiet_mode)
        
        # Text buffer for punctuation processing and conversation tracking
        self.text_buffer = deque(maxlen=50)
        self.conversation_buffer = []  # Store full conversation for EOU analysis
        self._setup_preprocessing()
        self._reset_streaming_state()
    
    def _setup_quiet_mode(self):
        """Setup comprehensive logging suppression for quiet mode"""
        # Suppress all NeMo logging
        nemo_logger = logging.getLogger('nemo_logger')
        nemo_logger.setLevel(logging.CRITICAL)
        
        # Suppress PyTorch Lightning logging
        pl_logger = logging.getLogger('pytorch_lightning')
        pl_logger.setLevel(logging.CRITICAL)
        
        # Suppress root logger
        logging.getLogger().setLevel(logging.CRITICAL)
        
        # Suppress all NeMo sub-loggers
        for name in logging.Logger.manager.loggerDict:
            if 'nemo' in name.lower():
                logging.getLogger(name).setLevel(logging.CRITICAL)
        
        # Suppress progress bars
        os.environ['NEMO_DISABLE_PROGRESS_BAR'] = '1'
        os.environ['TQDM_DISABLE'] = '1'
        
    def _setup_asr_model(self):
        """Load and configure the ASR model"""
        if not self.quiet_mode:
            print(f"Loading ASR model: {self.asr_model_name}")
        
        # Suppress output during model loading in quiet mode
        if self.quiet_mode:
            with open(os.devnull, 'w') as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.asr_model_name)
                finally:
                    sys.stderr = old_stderr
        else:
            self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.asr_model_name)
        
        # Update attention context size for multi-lookahead model
        if self.asr_model_name == "stt_en_fastconformer_hybrid_large_streaming_multi":
            if self.lookahead_size not in [0, 80, 480, 1040]:
                raise ValueError(
                    f"Lookahead size {self.lookahead_size} not valid for multi model. "
                    "Must be one of: 0, 80, 480, 1040 ms"
                )
            
            # Update attention context size
            left_context_size = self.asr_model.encoder.att_context_size[0]
            right_context_size = int(self.lookahead_size / ENCODER_STEP_LENGTH)
            self.asr_model.encoder.set_default_att_context_size([left_context_size, right_context_size])
            if not self.quiet_mode:
                print(f"Set attention context: [{left_context_size}, {right_context_size}]")
        
        # Configure decoder type
        self.asr_model.change_decoding_strategy(decoder_type=self.decoder_type)
        
        # Optimize decoding configuration
        decoding_cfg = self.asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.preserve_alignments = False
            if hasattr(self.asr_model, 'joint'):  # RNNT model
                decoding_cfg.greedy.max_symbols = 10
                decoding_cfg.fused_batch_size = -1
        
        self.asr_model.change_decoding_strategy(decoding_cfg)
        self.asr_model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.asr_model = self.asr_model.cuda()
            if not self.quiet_mode:
                print(f"ASR model moved to GPU: {torch.cuda.get_device_name()}")
            
        if not self.quiet_mode:
            print(f"Using ASR decoder: {self.decoder_type}")
            print(f"Lookahead size: {self.lookahead_size}ms")
    
    def _setup_punctuation_model(self):
        """Load and configure the punctuation model"""
        if not self.quiet_mode:
            print(f"Loading punctuation model: {self.punct_model_name}")
        
        try:
            # Suppress output during model loading in quiet mode
            if self.quiet_mode:
                with open(os.devnull, 'w') as devnull:
                    old_stderr = sys.stderr
                    sys.stderr = devnull
                    try:
                        self.punct_model = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained(
                            model_name=self.punct_model_name
                        )
                    finally:
                        sys.stderr = old_stderr
            else:
                self.punct_model = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained(
                    model_name=self.punct_model_name
                )
                
            self.punct_model.eval()
            
            if torch.cuda.is_available():
                self.punct_model = self.punct_model.cuda()
                if not self.quiet_mode:
                    print("Punctuation model moved to GPU")
                
        except Exception as e:
            if not self.quiet_mode:
                print(f"Warning: Could not load punctuation model {self.punct_model_name}: {e}")
                print("Continuing without punctuation...")
            self.punct_model = None
    
    def _setup_preprocessing(self):
        """Initialize audio preprocessor"""
        cfg = copy.deepcopy(self.asr_model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)
        
        # Streaming-specific preprocessing config
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        
        self.preprocessor = EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
        self.preprocessor.to(self.asr_model.device)
        
    def _reset_streaming_state(self, reset_conversation=True):
        """Reset streaming state for new session"""
        # Initialize cache states
        self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = \
            self.asr_model.encoder.get_initial_cache_state(batch_size=1)
        
        # Initialize streaming variables
        self.previous_hypotheses = None
        self.pred_out_stream = None
        self.step_num = 0
        
        # Pre-encode cache initialization
        self.pre_encode_cache_size = self.asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
        num_channels = self.asr_model.cfg.preprocessor.features
        self.cache_pre_encode = torch.zeros(
            (1, num_channels, self.pre_encode_cache_size), 
            device=self.asr_model.device
        )
        
        # Reset text buffer
        self.text_buffer.clear()
        
        # Reset conversation buffer if requested
        if reset_conversation:
            self.conversation_buffer = []
        
        if not self.quiet_mode:
            print("Streaming state reset")
    
    def _extract_transcriptions(self, hyps):
        """Extract text from hypothesis objects"""
        if isinstance(hyps[0], Hypothesis):
            return [hyp.text for hyp in hyps]
        else:
            return hyps
            
    def _preprocess_audio(self, audio):
        """Convert audio to mel-spectrogram"""
        device = self.asr_model.device
        
        # Convert to tensor and move to device
        audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(device)
        audio_signal_len = torch.Tensor([audio.shape[0]]).to(device)
        
        # Apply preprocessing
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=audio_signal, length=audio_signal_len
        )
        return processed_signal, processed_signal_length
    
    def _apply_punctuation(self, text):
        """Apply punctuation and capitalization to text"""
        if not self.punct_model or not text.strip():
            return text
            
        try:
            # Suppress output during punctuation in quiet mode
            if self.quiet_mode:
                with open(os.devnull, 'w') as devnull:
                    old_stderr = sys.stderr
                    old_stdout = sys.stdout
                    sys.stderr = devnull
                    sys.stdout = devnull
                    try:
                        punctuated_text = self.punct_model.add_punctuation_capitalization([text])[0]
                    finally:
                        sys.stderr = old_stderr
                        sys.stdout = old_stdout
            else:
                punctuated_text = self.punct_model.add_punctuation_capitalization([text])[0]
                
            return punctuated_text
                
        except Exception as e:
            if not self.quiet_mode:
                print(f"Punctuation error: {e}")
            return text
    
    def _check_end_of_utterance(self, text):
        """Check if the current text represents an end of utterance"""
        if not self.eou_detector or not text.strip():
            return False
        
        # Add current text to conversation buffer
        self.conversation_buffer.append(text)
        
        # Only check EOU periodically, not on every update
        if len(self.conversation_buffer) % 3 != 0:  # Check every 3rd update
            return False
        
        # Analyze the full conversation context for EOU
        full_conversation = " ".join(self.conversation_buffer)
        
        # Additional safeguards: don't check EOU too frequently
        if len(full_conversation.split()) < 5:  # Need at least 5 words
            return False
        
        # Check if this represents an end of utterance
        is_eou = self.eou_detector.detect_eou(full_conversation)
        
        if is_eou:
            if not self.quiet_mode:
                print(f"[EOU] End of utterance confirmed, resetting context")
            # Reset conversation buffer and streaming state
            self.conversation_buffer = []
            self._reset_streaming_state(reset_conversation=False)
            return True
        
        # Keep conversation buffer manageable but allow longer context for EOU
        if len(self.conversation_buffer) > 30:  # Increased from 20 to allow more context
            self.conversation_buffer = self.conversation_buffer[-25:]  # Keep more recent context
        
        return False
    
    def _check_end_of_utterance_frame_based(self, raw_model_outputs):
        """
        Check for EOU using frame-level analysis of model outputs
        
        Args:
            raw_model_outputs: Raw outputs from the ASR model
            
        Returns:
            bool: True if EOU detected based on frame analysis
        """
        if not self.enable_eou:
            return False
        
        # Analyze current frame outputs
        analysis = self.frame_detector.analyze_model_outputs(raw_model_outputs)
        
        # Update history
        self.frame_detector.update_history(analysis)
        
        # Check for silence period
        is_silence = self.frame_detector.detect_silence_period()
        
        if is_silence:
            # Get recent activity summary for additional validation
            summary = self.frame_detector.get_recent_activity_summary()
            
            # Additional criteria: ensure we've had some speech before declaring EOU
            has_had_speech = any(f.get("speech_detected", False) for f in self.frame_detector.frame_history[-30:])
            
            if has_had_speech and summary["avg_activity"] < 0.15:
                if not self.quiet_mode:
                    print(f"[FRAME-EOU] End of utterance detected based on model frame analysis")
                    print(f"  Recent activity: {summary['avg_activity']:.3f}")
                    print(f"  Speech frames: {summary['speech_frames']}/{summary['total_frames']}")
                
                # Reset frame history for fresh start
                self.frame_detector.frame_history = []
                return True
        
        return False
    
    def transcribe_chunk(self, audio_chunk):
        """
        Transcribe a single audio chunk with punctuation and frame-based EOU detection
        
        Args:
            audio_chunk: numpy array of audio samples (int16)
            
        Returns:
            tuple: (raw_text, punctuated_text, is_eou)
        """
        # Convert int16 to float32 and normalize
        audio_data = audio_chunk.astype(np.float32) / 32768.0
        
        # Get mel-spectrogram
        processed_signal, processed_signal_length = self._preprocess_audio(audio_data)
        
        # Prepend with pre-encode cache
        processed_signal = torch.cat([self.cache_pre_encode, processed_signal], dim=-1)
        processed_signal_length += self.cache_pre_encode.shape[1]
        
        # Update cache for next iteration
        self.cache_pre_encode = processed_signal[:, :, -self.pre_encode_cache_size:]
        
        # Run streaming inference
        with torch.no_grad():
            (
                self.pred_out_stream,
                transcribed_texts,
                self.cache_last_channel,
                self.cache_last_time,
                self.cache_last_channel_len,
                self.previous_hypotheses,
            ) = self.asr_model.conformer_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=self.cache_last_channel,
                cache_last_time=self.cache_last_time,
                cache_last_channel_len=self.cache_last_channel_len,
                keep_all_outputs=False,
                previous_hypotheses=self.previous_hypotheses,
                previous_pred_out=self.pred_out_stream,
                drop_extra_pre_encoded=None,
                return_transcription=True,
            )
        
        # Extract transcription
        final_transcriptions = self._extract_transcriptions(transcribed_texts)
        raw_text = final_transcriptions[0] if final_transcriptions else ""
        
        # Frame-based EOU detection using raw model outputs
        is_frame_eou = self._check_end_of_utterance_frame_based(self.pred_out_stream)
        
        # Text-based EOU detection (existing logic)
        is_text_eou = False
        if self.eou_detector and raw_text.strip():
            # Add current text to conversation buffer
            self.conversation_buffer.append(raw_text)
            
            # Only check text-based EOU periodically
            if len(self.conversation_buffer) % 5 != 0:  # Check every 5th update
                pass
            else:
                # Analyze the full conversation context for EOU
                full_conversation = " ".join(self.conversation_buffer)
                
                # Additional safeguards
                if len(full_conversation.split()) >= 5:  # Need at least 5 words
                    is_text_eou = self.eou_detector.detect_eou(full_conversation)
        
        # Combine both EOU detection methods
        is_eou = is_frame_eou or is_text_eou
        
        if is_eou:
            if not self.quiet_mode:
                eou_type = "Frame" if is_frame_eou else "Text"
                print(f"[{eou_type}-EOU] End of utterance confirmed, resetting context")
            # Reset conversation buffer and streaming state
            self.conversation_buffer = []
            self._reset_streaming_state(reset_conversation=False)
        
        # Keep conversation buffer manageable
        if len(self.conversation_buffer) > 30:
            self.conversation_buffer = self.conversation_buffer[-25:]
        
        # Apply punctuation if enabled
        if self.punct_model and raw_text.strip():
            punctuated_text = self._apply_punctuation(raw_text)
        else:
            punctuated_text = raw_text
        
        self.step_num += 1
        
        return raw_text, punctuated_text, is_eou

def list_audio_devices():
    """List available audio input devices"""
    p = pa.PyAudio()
    print('Available audio input devices:')
    input_devices = []
    
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels'):
            input_devices.append(i)
            print(f"  {i}: {dev.get('name')} (channels: {dev.get('maxInputChannels')})")
    
    p.terminate()
    return input_devices

def run_streaming_asr_with_punct(asr_system, device_id=None, chunk_size_ms=None, show_raw=False, quiet_mode=False):
    """
    Run streaming ASR with punctuation and EOU detection using microphone input
    
    Args:
        asr_system: OnlineASRWithPunctuation instance
        device_id: Audio device ID (None for interactive selection)
        chunk_size_ms: Chunk size in milliseconds (None for automatic)
        show_raw: Whether to show raw (unpunctuated) text
        quiet_mode: If True, suppress all logs and only show punctuated output
    """
    # Calculate chunk size
    if chunk_size_ms is None:
        chunk_size_ms = asr_system.lookahead_size + ENCODER_STEP_LENGTH
    
    if not quiet_mode:
        print(f"Using chunk size: {chunk_size_ms}ms")
    
    # Initialize PyAudio
    p = pa.PyAudio()
    
    try:
        # Select audio device
        if not quiet_mode:
            input_devices = list_audio_devices()
        else:
            # Silent device listing for quiet mode
            input_devices = []
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                if dev.get('maxInputChannels'):
                    input_devices.append(i)
        
        if not input_devices:
            if not quiet_mode:
                print('ERROR: No audio input device found.')
            return
            
        if device_id is None:
            if not quiet_mode:
                device_id = -1
                while device_id not in input_devices:
                    try:
                        device_id = int(input('Please enter input device ID: '))
                    except ValueError:
                        print("Please enter a valid device ID number")
                        continue
            else:
                # In quiet mode, use the first available device
                device_id = input_devices[0]
        
        if device_id not in input_devices:
            if not quiet_mode:
                print(f"Error: Device {device_id} not found in available input devices")
            return
            
        if not quiet_mode:
            print(f"Using device {device_id}: {p.get_device_info_by_index(device_id)['name']}")
        
        # Calculate frames per buffer
        frames_per_buffer = int(SAMPLE_RATE * chunk_size_ms / 1000) - 1
        
        # Store last transcription to avoid repetition
        last_raw = ""
        last_punct = ""
        
        # Define callback function
        def stream_callback(in_data, frame_count, time_info, status):
            nonlocal last_raw, last_punct
            
            if status and not quiet_mode:
                print(f"Stream status: {status}")
                
            # Convert audio data and transcribe
            signal = np.frombuffer(in_data, dtype=np.int16)
            raw_text, punct_text, is_eou = asr_system.transcribe_chunk(signal)
            
            # Only print if text has changed
            if raw_text.strip() and raw_text != last_raw:
                if quiet_mode:
                    # In quiet mode, only print the final punctuated text
                    if punct_text.strip():
                        print(punct_text, flush=True)
                        # Add newline for EOU in quiet mode
                        if is_eou:
                            print("", flush=True)  # Extra newline for utterance separation
                else:
                    if show_raw:
                        print(f"\rRaw: {raw_text}", end='')
                        if punct_text != raw_text:
                            print(f" | Punct: {punct_text}", end='', flush=True)
                        else:
                            print('', end='', flush=True)
                    else:
                        print(f"\r{punct_text}", end='', flush=True)
                    
                    # Print newline for end of utterance
                    if is_eou:
                        print("\n" + "="*50)  # Visual separator for new utterance
                
                last_raw = raw_text
                last_punct = punct_text
            
            return (in_data, pa.paContinue)
        
        # Open audio stream
        stream = p.open(
            format=pa.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_id,
            stream_callback=stream_callback,
            frames_per_buffer=frames_per_buffer
        )
        
        if not quiet_mode:
            print('\nListening... (Press Ctrl+C to stop)')
            if asr_system.punct_model:
                print('Punctuation and capitalization enabled')
            else:
                print('No punctuation model loaded')
            if asr_system.eou_detector:
                print('End-of-utterance detection enabled')
            else:
                print('EOU detection disabled')
            print('=' * 50)
        
        # Start streaming
        stream.start_stream()
        
        try:
            while stream.is_active():
                time.sleep(0.1)
        except KeyboardInterrupt:
            if not quiet_mode:
                print('\n\nStopping...')
        finally:
            stream.stop_stream()
            stream.close()
            if not quiet_mode:
                print("Audio stream stopped")
            
    finally:
        p.terminate()

def main():
    parser = argparse.ArgumentParser(
        description="Online ASR with Punctuation, Capitalization, and Frame-based EOU Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available ASR models:
{chr(10).join(f"  - {model}" for model in AVAILABLE_ASR_MODELS)}

Available punctuation models:
{chr(10).join(f"  - {model}" for model in AVAILABLE_PUNCT_MODELS)}

Examples:
  # Basic usage with frame-based EOU detection
  python {__file__} --asr-model stt_en_fastconformer_hybrid_large_streaming_multi --punct-model punctuation_en_bert

  # Adjust frame-based EOU sensitivity
  python {__file__} --frame-silence-threshold 15 --frame-activity-threshold 0.2

  # Disable text-based EOU, use only frame-based
  python {__file__} --no-text-eou
        """
    )
    
    parser.add_argument(
        "--asr-model", 
        default="stt_en_fastconformer_hybrid_large_streaming_multi",
        choices=AVAILABLE_ASR_MODELS,
        help="ASR model name to use"
    )
    
    parser.add_argument(
        "--punct-model",
        default="punctuation_en_bert",
        choices=AVAILABLE_PUNCT_MODELS + [None],
        help="Punctuation model name (None to disable punctuation)"
    )
    
    parser.add_argument(
        "--lookahead", 
        type=int, 
        default=480,
        help="Lookahead size in milliseconds (0, 80, 480, 1040 for multi model)"
    )
    
    parser.add_argument(
        "--decoder", 
        default="rnnt",
        choices=["rnnt", "ctc"],
        help="ASR decoder type to use"
    )
    
    parser.add_argument(
        "--device", 
        type=int,
        help="Audio device ID (will prompt if not provided)"
    )
    
    parser.add_argument(
        "--chunk-size", 
        type=int,
        help="Chunk size in milliseconds (auto-calculated if not provided)"
    )
    
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Show both raw and punctuated text"
    )
    
    parser.add_argument(
        "--list-devices", 
        action="store_true",
        help="List available audio devices and exit"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode: suppress all logs and only show punctuated text output"
    )
    
    parser.add_argument(
        "--no-eou",
        action="store_true",
        help="Disable end-of-utterance detection"
    )
    
    parser.add_argument(
        "--eou-threshold",
        type=float,
        default=0.8,
        help="Threshold for end-of-utterance detection (0.0-1.0, higher = less sensitive)"
    )
    
    parser.add_argument(
        "--frame-silence-threshold",
        type=int,
        default=20,
        help="Number of consecutive low-activity frames needed for frame-based EOU detection"
    )
    
    parser.add_argument(
        "--frame-activity-threshold",
        type=float,
        default=0.3,
        help="Minimum frame activity ratio for speech detection (0.0-1.0)"
    )
    
    parser.add_argument(
        "--no-text-eou",
        action="store_true",
        help="Disable text-based EOU detection, use only frame-based detection"
    )
    
    args = parser.parse_args()
    
    # List devices and exit if requested
    if args.list_devices:
        list_audio_devices()
        return
    
    # Set up comprehensive quiet mode before any model loading
    if args.quiet:
        # Suppress all possible logging early
        import warnings
        warnings.filterwarnings("ignore")
        
        # Set environment variables before model loading
        os.environ['NEMO_DISABLE_PROGRESS_BAR'] = '1'
        os.environ['TQDM_DISABLE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Suppress all logging
        logging.basicConfig(level=logging.CRITICAL)
        logging.getLogger().setLevel(logging.CRITICAL)
    
    try:
        # Initialize ASR system
        if not args.quiet:
            print("Initializing Online ASR with Frame-based EOU Detection system...")
            
        asr_system = OnlineASRWithPunctuation(
            asr_model_name=args.asr_model,
            punct_model_name=args.punct_model,
            lookahead_size=args.lookahead,
            decoder_type=args.decoder,
            quiet_mode=args.quiet,
            enable_eou=not args.no_eou
        )
        
        # Configure frame-based EOU detector
        if asr_system.frame_detector:
            asr_system.frame_detector.silence_threshold_frames = args.frame_silence_threshold
            asr_system.frame_detector.speech_activity_threshold = args.frame_activity_threshold
            
            if not args.quiet:
                print(f"Frame-based EOU configured:")
                print(f"  Silence threshold: {args.frame_silence_threshold} frames")
                print(f"  Activity threshold: {args.frame_activity_threshold}")
        
        # Disable text-based EOU if requested
        if args.no_text_eou:
            asr_system.eou_detector = None
            if not args.quiet:
                print("Text-based EOU detection disabled")
        
        # Update EOU threshold if specified
        if asr_system.eou_detector and hasattr(args, 'eou_threshold'):
            asr_system.eou_detector.threshold = args.eou_threshold
            if not args.quiet:
                print(f"EOU threshold set to: {args.eou_threshold}")
        
        if not args.quiet:
            print("\nASR system ready!")
            print(f"ASR Model: {args.asr_model}")
            print(f"Punctuation Model: {args.punct_model or 'None'}")
            print(f"EOU Detection: {'Enabled' if not args.no_eou else 'Disabled'}")
            print(f"Lookahead: {args.lookahead}ms")
            print(f"Decoder: {args.decoder}")
        
        # Run streaming ASR
        run_streaming_asr_with_punct(
            asr_system=asr_system,
            device_id=args.device,
            chunk_size_ms=args.chunk_size,
            show_raw=args.show_raw,
            quiet_mode=args.quiet
        )
        
    except KeyboardInterrupt:
        if not args.quiet:
            print("\nExiting...")
    except Exception as e:
        if not args.quiet:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
