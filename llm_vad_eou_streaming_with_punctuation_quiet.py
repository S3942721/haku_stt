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
        """Analyze frame-level speech activity using NeMo VAD model"""
        self.quiet_mode = quiet_mode
        
        # Frame-level analysis parameters
        self.vad_frame_duration = 0.02  # 20ms per VAD frame
        self.analysis_window_seconds = 2.0  # Analyze last 2 seconds (reduced from 3s)
        self.frames_per_analysis_window = int(self.analysis_window_seconds / self.vad_frame_duration)  # 100 frames for 2 seconds
        
        # Rolling window to store frame-level speech detection results
        self.speech_frame_history = deque(maxlen=self.frames_per_analysis_window)
        
        # EOU detection parameters - made more sensitive
        self.speech_proportion_threshold = 0.2  # If less than 20% of frames have speech, consider it silence (increased from 10%)
        self.min_frames_for_eou = int(0.5 / self.vad_frame_duration)  # Need at least 0.5 second of data (reduced from 1s)
        self.speech_probability_threshold = 0.5  # VAD threshold for individual frame speech detection
        
        # Consecutive silence detection for more responsive EOU
        self.consecutive_silence_frames = 0
        self.consecutive_silence_threshold = int(0.8 / self.vad_frame_duration)  # 0.8 seconds of consecutive silence
        
        # Load NeMo VAD model
        try:
            if not quiet_mode:
                print("Loading NeMo VAD model...")
            
            if quiet_mode:
                with open(os.devnull, 'w') as devnull:
                    old_stderr = sys.stderr
                    sys.stderr = devnull
                    try:
                        self.vad_model = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(
                            model_name="nvidia/frame_vad_multilingual_marblenet_v2.0"
                        )
                    finally:
                        sys.stderr = old_stderr
            else:
                self.vad_model = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(
                    model_name="nvidia/frame_vad_multilingual_marblenet_v2.0"
                )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.vad_model = self.vad_model.cuda()
                if not quiet_mode:
                    print("VAD model moved to GPU")
            
            self.vad_model.eval()
            
            if not quiet_mode:
                print("NeMo VAD model loaded successfully")
                print(f"VAD analysis window: {self.analysis_window_seconds}s ({self.frames_per_analysis_window} frames)")
                
        except Exception as e:
            if not quiet_mode:
                print(f"Warning: Could not load VAD model: {e}")
                print("Falling back to basic frame analysis...")
            self.vad_model = None
        
    def analyze_audio_vad(self, audio_signal):
        """
        Analyze audio using NeMo VAD model to detect speech activity per frame
        
        Args:
            audio_signal: Raw audio signal (numpy array, float32, 16kHz)
            
        Returns:
            list: List of boolean values indicating speech detection for each 20ms frame
        """
        try:
            if self.vad_model is None or len(audio_signal) == 0:
                return []
            
            # Ensure audio is the right format
            if not isinstance(audio_signal, np.ndarray):
                audio_signal = np.array(audio_signal)
            
            if audio_signal.dtype != np.float32:
                audio_signal = audio_signal.astype(np.float32)
            
            # Minimum length check (VAD needs some audio to analyze)
            min_samples = 320  # 20ms at 16kHz
            if len(audio_signal) < min_samples:
                # Pad with zeros if too short
                padding = np.zeros(min_samples - len(audio_signal), dtype=np.float32)
                audio_signal = np.concatenate([audio_signal, padding])
            
            # Convert to tensor and add batch dimension
            input_signal = torch.from_numpy(audio_signal).unsqueeze(0).float()
            input_signal_length = torch.tensor([input_signal.shape[1]]).long()
            
            # Move to same device as model
            device = next(self.vad_model.parameters()).device
            input_signal = input_signal.to(device)
            input_signal_length = input_signal_length.to(device)
            
            # Run VAD inference
            with torch.no_grad():
                vad_outputs = self.vad_model(
                    input_signal=input_signal,
                    input_signal_length=input_signal_length
                )
            
            # Move outputs back to CPU for processing
            if hasattr(vad_outputs, 'cpu'):
                vad_probs = vad_outputs.cpu().numpy()
            else:
                vad_probs = vad_outputs
            
            # Handle VAD output format
            # Shape is typically [batch, time_frames, num_classes] where num_classes=2 for [non-speech, speech]
            if vad_probs.ndim == 3:
                vad_probs = vad_probs.squeeze(0)  # Remove batch dimension -> [time_frames, num_classes]
            
            # Convert probabilities to binary speech detection for each frame
            if vad_probs.ndim == 2:
                # Shape is [time_frames, 2] where columns are [non-speech_prob, speech_prob]
                if vad_probs.shape[1] == 2:
                    # Extract speech probabilities (second column)
                    speech_probs = vad_probs[:, 1]  # Get speech probabilities
                    speech_detected_frames = (speech_probs > self.speech_probability_threshold).tolist()
                    
                    if len(speech_detected_frames) > 0:
                    # if not self.quiet_mode and len(speech_detected_frames) > 0:
                        # Debug: show some frame stats occasionally
                        avg_speech_prob = np.mean(speech_probs)
                        speech_frame_count = sum(speech_detected_frames)
                        if len(self.speech_frame_history) % 50 == 0:  # Log every 50 updates
                            print(f"[VAD-DEBUG] {len(speech_detected_frames)} frames, "
                                  f"avg_speech_prob: {avg_speech_prob:.3f}, "
                                  f"speech_frames: {speech_frame_count}")
                    return speech_detected_frames
                else:
                    if not self.quiet_mode:
                        print(f"Unexpected VAD output shape: {vad_probs.shape} - expected [frames, 2]")
                    return []
            elif vad_probs.ndim == 1:
                # Single dimension - treat as speech probabilities directly
                speech_detected_frames = (vad_probs > self.speech_probability_threshold).tolist()
                return speech_detected_frames
            else:
                if not self.quiet_mode:
                    print(f"Unexpected VAD output shape: {vad_probs.shape}")
                return []
            
        except Exception as e:
            if not self.quiet_mode:
                print(f"VAD analysis error: {e}")
            return []
    
    def update_speech_frame_history(self, speech_frames):
        """
        Update the rolling window of speech frame detections
        
        Args:
            speech_frames: List of boolean values for speech detection per frame
        """
        # Add each frame's speech detection result to the rolling window
        for is_speech in speech_frames:
            self.speech_frame_history.append(is_speech)
            
            # Track consecutive silence frames for immediate EOU detection
            if not is_speech:
                self.consecutive_silence_frames += 1
            else:
                self.consecutive_silence_frames = 0
    
    def get_recent_speech_proportion(self, window_seconds=None):
        """
        Calculate the proportion of frames with speech in the recent window
        
        Args:
            window_seconds: Analysis window in seconds (default: use class setting)
            
        Returns:
            dict: Analysis results including speech proportion and frame counts
        """
        if window_seconds is None:
            window_seconds = self.analysis_window_seconds
        
        window_frames = int(window_seconds / self.vad_frame_duration)
        
        if len(self.speech_frame_history) < self.min_frames_for_eou:
            return {
                "speech_proportion": 0.0,
                "speech_frames": 0,
                "total_frames": len(self.speech_frame_history),
                "window_seconds": window_seconds,
                "sufficient_data": False
            }
        
        # Get recent frames within the window
        recent_frames = list(self.speech_frame_history)[-window_frames:] if len(self.speech_frame_history) >= window_frames else list(self.speech_frame_history)
        
        speech_frames_count = sum(recent_frames)
        total_frames = len(recent_frames)
        speech_proportion = speech_frames_count / total_frames if total_frames > 0 else 0.0
        
        return {
            "speech_proportion": speech_proportion,
            "speech_frames": speech_frames_count,
            "total_frames": total_frames,
            "window_seconds": len(recent_frames) * self.vad_frame_duration,
            "sufficient_data": len(self.speech_frame_history) >= self.min_frames_for_eou
        }
    
    def detect_silence_period(self):
        """
        Detect if we're in a silence period based on speech proportion in recent window
        Uses both proportion-based analysis and consecutive silence detection
        
        Returns:
            bool: True if silence period detected (potential EOU)
        """
        if len(self.speech_frame_history) < self.min_frames_for_eou:
            return False
        
        # Method 1: Consecutive silence detection (more immediate)
        if self.consecutive_silence_frames >= self.consecutive_silence_threshold:
            if not self.quiet_mode:
                silence_duration = self.consecutive_silence_frames * self.vad_frame_duration
                print(f"[VAD-EOU] Consecutive silence detected: {silence_duration:.1f}s ({self.consecutive_silence_frames} frames)")
            return True
        
        # Method 2: Proportion-based analysis (smoother)
        analysis = self.get_recent_speech_proportion()
        
        if not analysis["sufficient_data"]:
            return False
        
        # Detect silence if speech proportion is below threshold
        is_silence_period = analysis["speech_proportion"] < self.speech_proportion_threshold
        
        if not self.quiet_mode and is_silence_period:
            print(f"[VAD-EOU] Low speech proportion detected: {analysis['speech_proportion']:.3f} "
                  f"({analysis['speech_frames']}/{analysis['total_frames']} frames over {analysis['window_seconds']:.1f}s)")
        
        return is_silence_period
    
    def get_recent_activity_summary(self, frames=None):
        """Get summary of recent speech activity"""
        if frames is None:
            # Use last 0.5 seconds for summary
            frames = int(0.5 / self.vad_frame_duration)
        
        analysis = self.get_recent_speech_proportion(window_seconds=frames * self.vad_frame_duration)
        
        return {
            "avg_activity": analysis["speech_proportion"],
            "speech_frames": analysis["speech_frames"],
            "total_frames": analysis["total_frames"],
            "avg_confidence": analysis["speech_proportion"]  # For compatibility
        }
    
    def has_recent_speech_activity(self, lookback_seconds=1.0):
        """
        Check if there has been significant speech activity in recent history
        
        Args:
            lookback_seconds: How far back to look for speech activity
            
        Returns:
            bool: True if there has been recent speech activity
        """
        lookback_analysis = self.get_recent_speech_proportion(window_seconds=lookback_seconds)
        
        if not lookback_analysis["sufficient_data"]:
            return False
        
        # Consider it "recent speech activity" if more than 15% of frames had speech (reduced from 20%)
        return lookback_analysis["speech_proportion"] > 0.15
    
    def reset_silence_tracking(self):
        """Reset consecutive silence tracking"""
        self.consecutive_silence_frames = 0
    
    # Keep legacy method for compatibility
    def analyze_model_outputs(self, pred_outputs, processed_signal_length=None):
        """Fallback method - kept for compatibility but VAD is preferred"""
        return {"speech_detected": True, "confidence": 0.5, "frame_activity": 0.5}

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
        self.current_utterance = ""  # Accumulate text for complete utterance output
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
        """Reset streaming state for new session - complete ASR reset"""
        # Initialize cache states - complete reset
        self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = \
            self.asr_model.encoder.get_initial_cache_state(batch_size=1)
        
        # Initialize streaming variables - complete reset
        self.previous_hypotheses = None
        self.pred_out_stream = None
        self.step_num = 0
        
        # Pre-encode cache initialization - complete reset
        self.pre_encode_cache_size = self.asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
        num_channels = self.asr_model.cfg.preprocessor.features
        self.cache_pre_encode = torch.zeros(
            (1, num_channels, self.pre_encode_cache_size), 
            device=self.asr_model.device
        )
        
        # Reset text buffer - complete reset
        self.text_buffer.clear()
        
        # Reset conversation buffer if requested
        if reset_conversation:
            self.conversation_buffer = []
        
        # Reset current utterance accumulation
        self.current_utterance = ""
        
        if not self.quiet_mode:
            print("Complete ASR streaming state reset")
    
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
    
    def _check_end_of_utterance_frame_based(self, audio_chunk_raw):
        """
        Check for EOU using VAD analysis of raw audio with rolling window approach
        
        Args:
            audio_chunk_raw: Raw audio chunk (numpy array, float32)
            
        Returns:
            bool: True if EOU detected based on VAD analysis
        """
        if not self.enable_eou or self.frame_detector.vad_model is None:
            return False
        
        # Analyze current audio chunk with VAD to get frame-level speech detection
        speech_frames = self.frame_detector.analyze_audio_vad(audio_chunk_raw)
        
        if not speech_frames:  # No frames analyzed
            return False
        
        # Update the rolling window with new frame results
        self.frame_detector.update_speech_frame_history(speech_frames)
        
        # Check if current period represents silence (low speech proportion or consecutive silence)
        is_silence = self.frame_detector.detect_silence_period()
        
        if is_silence:
            # Additional validation: ensure we've had some speech before declaring EOU
            has_had_recent_speech = self.frame_detector.has_recent_speech_activity(lookback_seconds=1.5)  # Reduced from 2.0s
            
            if has_had_recent_speech:
                # Get summary for logging
                summary = self.frame_detector.get_recent_activity_summary()
                
                if not self.quiet_mode:
                    print(f"[VAD-EOU] End of utterance detected based on speech analysis")
                    print(f"  Recent speech proportion: {summary['avg_activity']:.3f}")
                    print(f"  Speech frames: {summary['speech_frames']}/{summary['total_frames']}")
                    print(f"  Consecutive silence frames: {self.frame_detector.consecutive_silence_frames}")
                
                # Reset frame history and silence tracking for fresh start
                self.frame_detector.speech_frame_history.clear()
                self.frame_detector.reset_silence_tracking()
                return True
            else:
                if not self.quiet_mode:
                    print(f"[VAD-DEBUG] Silence detected but no recent speech activity - not triggering EOU")
        
        return False
    
    def transcribe_chunk(self, audio_chunk):
        """
        Transcribe a single audio chunk with punctuation and VAD-based EOU detection
        
        Args:
            audio_chunk: numpy array of audio samples (int16)
            
        Returns:
            tuple: (raw_text, punctuated_text, is_eou, complete_utterance)
        """
        # Convert int16 to float32 and normalize
        audio_data = audio_chunk.astype(np.float32) / 32768.0
        
        # VAD-based EOU detection using raw audio (before preprocessing)
        is_frame_eou = self._check_end_of_utterance_frame_based(audio_data)
        
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
        
        # Apply punctuation if enabled
        if self.punct_model and raw_text.strip():
            punctuated_text = self._apply_punctuation(raw_text)
        else:
            punctuated_text = raw_text
        
        # Accumulate current utterance for complete output
        if punctuated_text.strip():
            self.current_utterance = punctuated_text
        
        # Handle EOU detection and complete utterance output
        complete_utterance = None
        if is_eou:
            # Store the final complete utterance for output
            complete_utterance = self.current_utterance.strip() if self.current_utterance.strip() else punctuated_text.strip()
            
            if not self.quiet_mode:
                eou_type = "VAD" if is_frame_eou else "Text"
                print(f"[{eou_type}-EOU] End of utterance confirmed, performing complete ASR reset")
            
            # COMPLETE ASR RESET - Reset conversation buffer and streaming state entirely
            self.conversation_buffer = []
            self._reset_streaming_state(reset_conversation=False)
            
            # Return the complete utterance for final output
            return raw_text, punctuated_text, True, complete_utterance
        else:
            # Keep conversation buffer manageable
            if len(self.conversation_buffer) > 30:
                self.conversation_buffer = self.conversation_buffer[-25:]
        
        self.step_num += 1
        
        return raw_text, punctuated_text, is_eou, complete_utterance

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
            raw_text, punct_text, is_eou, complete_utterance = asr_system.transcribe_chunk(signal)
            
            # Handle EOU - output complete utterance with clear separators
            if is_eou and complete_utterance:
                if quiet_mode:
                    # In quiet mode, output complete utterance with separators
                    print(f"\n{'='*70}")
                    print(f"COMPLETE UTTERANCE:")
                    print(f"{'='*70}")
                    print(f"{complete_utterance}")
                    print(f"{'='*70}\n")
                else:
                    # In non-quiet mode, show completion with separators
                    print(f"\n{'='*70}")
                    print(f"[COMPLETE UTTERANCE - ASR FULLY RESET]")
                    print(f"{complete_utterance}")
                    print(f"{'='*70}")
                
                # Reset tracking variables for fresh start
                last_raw = ""
                last_punct = ""
                return (in_data, pa.paContinue)
            
            # Only print if text has changed and it's not an EOU
            if raw_text.strip() and raw_text != last_raw and not is_eou:
                if quiet_mode:
                    # In quiet mode, show incremental updates but suppress frequent updates
                    if punct_text.strip() and len(punct_text.split()) > len(last_punct.split() if last_punct else []):
                        print(f"\r{punct_text}", end='', flush=True)
                else:
                    if show_raw:
                        print(f"\rRaw: {raw_text}", end='')
                        if punct_text != raw_text:
                            print(f" | Punct: {punct_text}", end='', flush=True)
                        else:
                            print('', end='', flush=True)
                    else:
                        print(f"\r{punct_text}", end='', flush=True)
                
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
        description="Online ASR with Punctuation, Capitalization, and VAD-based EOU Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available ASR models:
{chr(10).join(f"  - {model}" for model in AVAILABLE_ASR_MODELS)}

Available punctuation models:
{chr(10).join(f"  - {model}" for model in AVAILABLE_PUNCT_MODELS)}

Examples:
  # Basic usage with VAD-based EOU detection
  python {__file__} --asr-model stt_en_fastconformer_hybrid_large_streaming_multi --punct-model punctuation_en_bert

  # Adjust VAD-based EOU sensitivity
  python {__file__} --vad-silence-threshold 10 --vad-speech-threshold 0.4

  # Disable text-based EOU, use only VAD-based
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
        "--vad-silence-threshold",
        type=int,
        default=15,
        help="Number of consecutive low-activity frames needed for VAD-based EOU detection"
    )
    
    parser.add_argument(
        "--vad-speech-threshold",
        type=float,
        default=0.5,
        help="VAD probability threshold for speech detection (0.0-1.0)"
    )
    
    parser.add_argument(
        "--vad-activity-threshold",
        type=float,
        default=0.3,
        help="Minimum frame activity ratio for speech detection (0.0-1.0)"
    )
    
    parser.add_argument(
        "--no-text-eou",
        action="store_true",
        help="Disable text-based EOU detection, use only frame-based detection"
    )
    
    parser.add_argument(
        "--vad-speech-proportion-threshold",
        type=float,
        default=0.2,  # Changed default from 0.1 to 0.2
        help="Speech proportion threshold for EOU detection (0.0-1.0, lower = more sensitive)"
    )
    
    parser.add_argument(
        "--vad-analysis-window",
        type=float,
        default=2.0,  # Changed default from 3.0 to 2.0
        help="Analysis window in seconds for speech proportion calculation"
    )
    
    parser.add_argument(
        "--vad-consecutive-silence-threshold",
        type=float,
        default=0.8,
        help="Consecutive silence duration in seconds needed for immediate EOU detection"
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
            print("Initializing Online ASR with VAD-based EOU Detection system...")
            
        asr_system = OnlineASRWithPunctuation(
            asr_model_name=args.asr_model,
            punct_model_name=args.punct_model,
            lookahead_size=args.lookahead,
            decoder_type=args.decoder,
            quiet_mode=args.quiet,
            enable_eou=not args.no_eou
        )
        
        # Configure VAD-based EOU detector
        if asr_system.frame_detector and asr_system.frame_detector.vad_model:
            asr_system.frame_detector.speech_probability_threshold = args.vad_speech_threshold
            asr_system.frame_detector.speech_proportion_threshold = args.vad_speech_proportion_threshold
            asr_system.frame_detector.analysis_window_seconds = args.vad_analysis_window
            
            # Configure consecutive silence threshold
            if hasattr(args, 'vad_consecutive_silence_threshold'):
                asr_system.frame_detector.consecutive_silence_threshold = int(
                    args.vad_consecutive_silence_threshold / asr_system.frame_detector.vad_frame_duration
                )
            
            # Recalculate frames per analysis window
            asr_system.frame_detector.frames_per_analysis_window = int(
                args.vad_analysis_window / asr_system.frame_detector.vad_frame_duration
            )
            asr_system.frame_detector.speech_frame_history = deque(
                maxlen=asr_system.frame_detector.frames_per_analysis_window
            )
            
            if not args.quiet:
                print(f"VAD-based EOU configured:")
                print(f"  Analysis window: {args.vad_analysis_window}s")
                print(f"  Speech probability threshold: {args.vad_speech_threshold}")
                print(f"  Speech proportion threshold: {args.vad_speech_proportion_threshold}")
                print(f"  Consecutive silence threshold: {args.vad_consecutive_silence_threshold if hasattr(args, 'vad_consecutive_silence_threshold') else 0.8}s")
                print(f"  Frames per window: {asr_system.frame_detector.frames_per_analysis_window}")
        
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
