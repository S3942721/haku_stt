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
import logging.handlers
import os
import sys
import json
import asyncio
import threading
from pathlib import Path
from collections import deque

# Add WebSocket imports
try:
    import websockets
    WEBSOCKET_AVAILABLE = True
except ImportError:
    # Can't use logger here as it's not set up yet
    import sys
    print("WARNING: websockets not available - WebSocket functionality disabled", file=sys.stderr)
    WEBSOCKET_AVAILABLE = False

# Add remote audio stream import
try:
    from remote_audio_stream import RemoteAudioStream
    REMOTE_AUDIO_AVAILABLE = True
except ImportError:
    # Can't use logger here as it's not set up yet
    import sys
    print("WARNING: Remote audio stream not available - falling back to microphone only", file=sys.stderr)
    REMOTE_AUDIO_AVAILABLE = False

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

# Progress bar suppression context manager
class NoStdStreams(object):
    """Context manager to suppress stdout/stderr output (like progress bars)"""
    def __init__(self, stdout=None, stderr=None):
        self.devnull = open(os.devnull, 'w')
        self._stdout = stdout or self.devnull
        self._stderr = stderr or self.devnull

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()

# Add custom log level for VERBOSE (between DEBUG and INFO)
VERBOSE = 15
logging.addLevelName(VERBOSE, 'VERBOSE')

def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)

logging.Logger.verbose = verbose

class LoggerMixin:
    """Mixin to provide consistent logging to classes"""
    
    def _resolve_log_level(self, log_level):
        # Accepts string or int log level
        if isinstance(log_level, int):
            return log_level
        if isinstance(log_level, str):
            level_map = {
                "critical": logging.CRITICAL,
                "error": logging.ERROR,
                "warning": logging.WARNING,
                "info": logging.INFO,
                "verbose": VERBOSE,
                "debug": logging.DEBUG
            }
            # Try to parse as int string
            try:
                return int(log_level)
            except Exception:
                pass
            return level_map.get(log_level.lower(), logging.DEBUG)
        return logging.DEBUG

    def __init_logger__(self, name=None, log_level="debug", log_format=None, log_file=None, 
                       log_max_size=10, log_backup_count=3):
        """Initialize logger for the class"""
        if name is None:
            name = self.__class__.__name__
        self.logger = logging.getLogger(f"haku_stt.{name}")
        # Always set logger level to match root logger - ignore passed log_level parameter
        root_level = logging.getLogger().level
        self.logger.setLevel(root_level)

        
        # Only add handlers if they don't exist (avoid duplicate handlers)
        if not self.logger.handlers:
            # Default format if not provided
            if log_format is None:
                log_format = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
            
            # Console formatter without timestamp
            console_format = "[%(name)s] [%(levelname)s] %(message)s"
            console_formatter = logging.Formatter(console_format)
            
            # File formatter with timestamp (use provided format)
            file_formatter = logging.Formatter(log_format)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # File handler if specified
            if log_file:
                try:
                    log_path = Path(log_file)
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    file_handler = logging.handlers.RotatingFileHandler(
                        log_file,
                        maxBytes=log_max_size * 1024 * 1024,  # Convert MB to bytes
                        backupCount=log_backup_count
                    )
                    file_handler.setFormatter(file_formatter)
                    self.logger.addHandler(file_handler)
                except Exception as e:
                    # Can't use self.logger here as it may not be fully set up
                    import sys
                    print(f"Warning: Could not setup file logging to {log_file}: {e}", file=sys.stderr)
        
        # Prevent propagation to root logger to avoid duplicate messages
        self.logger.propagate = False

class UnifiedEOUDetector(LoggerMixin):
    def __init__(self, log_config=None, config=None):
        """
        Initialize unified EOU detection that combines multiple detection methods
        
        Args:
            log_config: Logging configuration
            config: Configuration dictionary with EOU settings and weights
        """
        # Initialize logging first
        if log_config is None:
            log_config = {"log_level": "debug"}
        self.__init_logger__("UnifiedEOU", **log_config)
        
        # Default configuration
        default_config = {
            "eou-threshold": 0.7,  # Overall EOU confidence threshold
            "eou-weights": {
                "vad": 2,      # VAD-based detection weight
                "text": 1,     # Text-based detection weight  
                "silence": 0,  # Pure silence detection weight
                "buffer_vad": 1 # Text buffer + low VAD detection weight
            },
            "eou-min-words": 4,
            "eou-confirmation-needed": 2,
            "enable-text-eou": True,
            "enable-vad-eou": True,
            "enable-silence-eou": True
        }
        
        # Merge with provided config
        self.config = {**default_config, **(config or {})}
        
        # Validate and normalize weights
        self._validate_weights()
        
        # Initialize individual detectors
        self.text_detector = None
        self.vad_detector = None  # Will be set from outside
        
        # Initialize text-based detector if enabled
        if self.config["enable-text-eou"]:
            try:
                self.text_detector = TextEOUDetector(
                    log_config=log_config,
                    eou_threshold=self.config.get("text-eou-threshold", 0.8),
                    min_words=self.config["eou-min-words"],
                    confirmation_needed=self.config["eou-confirmation-needed"]
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize text EOU detector: {e}")
                self.text_detector = None
        
        # EOU timing control
        self.last_eou_time = 0
        self.min_eou_interval = 2.0  # Minimum seconds between EOU detections
        
        self.logger.info("Unified EOU detector initialized")
        self.logger.debug(f"Weights: {self.config['eou-weights']}")
        self.logger.debug(f"Overall threshold: {self.config['eou-threshold']}")
    
    def _validate_weights(self):
        """Validate and normalize EOU detection weights"""
        weights = self.config["eou-weights"]
        
        # Ensure all weights are non-negative
        for method, weight in weights.items():
            if weight < 0:
                self.logger.warning(f"Negative weight for {method}: {weight}, setting to 0")
                weights[method] = 0
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight == 0:
            # If all weights are 0, set default equal weights
            self.logger.warning("All EOU weights are 0, using equal weights")
            num_methods = len(weights)
            for method in weights:
                weights[method] = 1.0 / num_methods
        else:
            # Normalize to sum to 1.0
            for method in weights:
                weights[method] = weights[method] / total_weight
        
        self.logger.debug(f"Normalized EOU weights: {weights}")
    
    def set_vad_detector(self, vad_detector):
        """Set the VAD detector reference"""
        self.vad_detector = vad_detector
    
    def detect_eou(self, text=None, audio_data=None):
        """
        Unified EOU detection combining multiple methods
        
        Args:
            text: Text for text-based EOU detection
            audio_data: Audio data for VAD-based detection
            
        Returns:
            dict: {
                "is_eou": bool,
                "confidence": float,
                "method_scores": dict,
                "triggered_by": list
            }
        """
        current_time = time.time()
        
        # Prevent too frequent EOU detections
        if current_time - self.last_eou_time < self.min_eou_interval:
            return {
                "is_eou": False,
                "confidence": 0.0,
                "method_scores": {},
                "triggered_by": [],
                "reason": "too_soon"
            }
        
        method_scores = {}
        triggered_methods = []
        
        # 1. VAD-based EOU detection
        vad_score = 0.0
        vad_details = {}
        if self.config["enable-vad-eou"] and self.vad_detector and audio_data is not None:
            try:
                vad_eou, vad_details = self._detect_vad_eou_with_details()
                
                # Calculate VAD confidence score based on silence metrics
                speech_proportion = vad_details.get('speech_proportion', 1.0)
                proportion_threshold = vad_details.get('proportion_threshold', 0.15)
                consecutive_silence = vad_details.get('consecutive_silence_frames', 0)
                silence_threshold = vad_details.get('silence_threshold', 25)
                
                # Convert metrics to confidence scores (0.0 to 1.0)
                # Lower speech proportion = higher EOU confidence
                proportion_confidence = max(0.0, (proportion_threshold - speech_proportion) / proportion_threshold)
                
                # Higher consecutive silence = higher EOU confidence
                silence_confidence = min(1.0, consecutive_silence / silence_threshold)
                
                # CONSECUTIVE SILENCE OVERRIDE: If we have strong consecutive silence, boost confidence significantly
                if consecutive_silence >= silence_threshold:
                    # Strong consecutive silence detected - give very high confidence
                    consecutive_silence_boost = min(1.0, consecutive_silence / silence_threshold)
                    # Apply exponential boost for consecutive silence - this should dominate
                    boosted_silence_confidence = min(1.0, consecutive_silence_boost ** 0.5)  # Square root for gentler curve
                    vad_score = max(boosted_silence_confidence, silence_confidence, proportion_confidence)
                    
                    # Additional boost if consecutive silence is much higher than threshold
                    if consecutive_silence >= silence_threshold * 1.5:
                        vad_score = min(1.0, vad_score * 2)  # 200% boost for very long silence
                else:
                    # Normal calculation when consecutive silence hasn't reached threshold
                    vad_score = max(proportion_confidence, silence_confidence)
                
                # Clamp to valid range
                vad_score = max(0.0, min(1.0, vad_score))
                
                if vad_eou:
                    triggered_methods.append("vad")
                
                # Log VAD detection details
                self.logger.verbose(f"VAD EOU: triggered={vad_eou}, score={vad_score:.3f}, "
                                  f"speech_proportion={vad_details.get('speech_proportion', 'N/A'):.3f}, "
                                  f"proportion_threshold={vad_details.get('proportion_threshold', 'N/A'):.3f}, "
                                  f"consecutive_silence={vad_details.get('consecutive_silence_frames', 'N/A')}, "
                                  f"silence_threshold={vad_details.get('silence_threshold', 'N/A')}")
            except Exception as e:
                self.logger.debug(f"VAD EOU detection error: {e}")
        
        method_scores["vad"] = vad_score
        
        # 2. Text-based EOU detection
        text_score = 0.0
        text_details = {}
        if self.config["enable-text-eou"] and self.text_detector and text:
            try:
                text_eou, text_details = self.text_detector.detect_eou_with_details(text)
                
                # Use the raw probability as the confidence score
                text_score = text_details.get('eou_probability', 0.0)
                
                # Clamp to valid range
                text_score = max(0.0, min(1.0, text_score))
                
                if text_eou:
                    triggered_methods.append("text")
                
                # Log text detection details
                self.logger.verbose(f"Text EOU: triggered={text_eou}, score={text_score:.3f}, "
                                  f"probability={text_details.get('eou_probability', 'N/A'):.3f}, "
                                  f"threshold={text_details.get('threshold', 'N/A'):.3f}, "
                                  f"word_count={text_details.get('word_count', 'N/A')}, "
                                  f"min_words={text_details.get('min_words', 'N/A')}, "
                                  f"confirmations={text_details.get('confirmation_count', 'N/A')}/{text_details.get('confirmation_needed', 'N/A')}")
            except Exception as e:
                self.logger.debug(f"Text EOU detection error: {e}")
        
        method_scores["text"] = text_score
        
        # 3. Silence-based EOU detection (simple timeout)
        silence_score = 0.0
        if self.config["enable-silence-eou"]:
            silence_eou = self._detect_silence_eou()
            silence_score = 1.0 if silence_eou else 0.0
            if silence_eou:
                triggered_methods.append("silence")
        
        method_scores["silence"] = silence_score
        
        # 4. Text buffer + Low VAD EOU detection
        buffer_vad_score = 0.0
        if self.config.get("enable-buffer-vad-eou", True) and self.vad_detector and text:
            try:
                buffer_vad_eou, buffer_vad_details = self._detect_buffer_vad_eou_with_details(text)
                
                if buffer_vad_eou:
                    # Calculate confidence based on text length and VAD silence level
                    text_length_factor = min(1.0, len(text.split()) / 10.0)  # Normalize by 10 words
                    vad_silence_factor = 1.0 - buffer_vad_details.get('recent_speech_proportion', 1.0)
                    
                    # Combined confidence: more text + more silence = higher confidence
                    buffer_vad_score = (text_length_factor * 0.6) + (vad_silence_factor * 0.4)
                    buffer_vad_score = max(0.0, min(1.0, buffer_vad_score))
                    
                    triggered_methods.append("buffer_vad")
                    
                    # Log buffer+VAD detection details
                    self.logger.verbose(f"Buffer+VAD EOU: triggered={buffer_vad_eou}, score={buffer_vad_score:.3f}, "
                                      f"text_words={buffer_vad_details.get('text_word_count', 'N/A')}, "
                                      f"recent_vad_proportion={buffer_vad_details.get('recent_speech_proportion', 'N/A'):.3f}, "
                                      f"vad_threshold={buffer_vad_details.get('vad_threshold', 'N/A'):.3f}")
            except Exception as e:
                self.logger.debug(f"Buffer+VAD EOU detection error: {e}")
        
        method_scores["buffer_vad"] = buffer_vad_score
        
        # Calculate weighted confidence
        weights = self.config["eou-weights"]
        weighted_confidence = 0.0
        
        # Log detailed weighted confidence calculation
        confidence_breakdown = []
        for method, score in method_scores.items():
            weight = weights.get(method, 0.0)
            contribution = score * weight
            weighted_confidence += contribution
            confidence_breakdown.append(f"{method}:{score:.1f}*{weight:.2f}={contribution:.3f}")
        
        # Log the detailed breakdown at verbose level
        self.logger.verbose(f"Unified EOU confidence calculation: {' + '.join(confidence_breakdown)} = {weighted_confidence:.3f}")
        
        # Use raw confidence directly - no smoothing needed since individual detectors already use historical data
        final_confidence = weighted_confidence
        
        # CONSECUTIVE SILENCE OVERRIDE: Check if VAD has detected very strong consecutive silence
        consecutive_silence_override = False
        if self.vad_detector and method_scores.get("vad", 0) >= 1.0:
            # Check if this is due to strong consecutive silence
            consecutive_silence = getattr(self.vad_detector, 'consecutive_silence_frames', 0)
            silence_threshold = getattr(self.vad_detector, 'consecutive_silence_threshold', 25)
            
            if consecutive_silence >= silence_threshold * 1.5:  # 1.5x the threshold
                consecutive_silence_override = True
                # Boost final confidence significantly for very long silence
                silence_boost = min(0.3, (consecutive_silence - silence_threshold) * 0.01)
                final_confidence = min(1.0, final_confidence + silence_boost)
                
                self.logger.debug(f"Consecutive silence override: {consecutive_silence} frames >= {silence_threshold * 1.5:.0f}, "
                                f"boosting confidence by {silence_boost:.3f} to {final_confidence:.3f}")
        
        # Determine if EOU should trigger
        # Lower threshold slightly if we have consecutive silence override
        effective_threshold = self.config["eou-threshold"]
        if consecutive_silence_override:
            effective_threshold = max(0.5, self.config["eou-threshold"] * 0.9)  # 10% lower threshold
        
        is_eou = final_confidence >= effective_threshold
        
        # Always log the final decision at debug level for analysis
        self.logger.debug(f"Unified EOU decision: confidence={final_confidence:.3f}, "
                         f"threshold={self.config['eou-threshold']:.3f}, "
                         f"triggered={is_eou}, methods={list(triggered_methods)}")
        
        if is_eou:
            self.last_eou_time = current_time
            self.logger.info(f"Unified EOU triggered - confidence: {final_confidence:.3f} "
                           f"(threshold: {self.config['eou-threshold']:.3f})")
            self.logger.debug(f"Method scores: {method_scores}")
            self.logger.debug(f"Triggered by: {triggered_methods}")
        
        result = {
            "is_eou": is_eou,
            "confidence": final_confidence,
            "raw_confidence": weighted_confidence,
            "method_scores": method_scores,
            "triggered_by": triggered_methods,
            "weights_used": weights,
            "confidence_breakdown": confidence_breakdown
        }
        
        # Store last result for tuning tools
        self.last_eou_result = result
        
        return result
    
    def _detect_vad_eou(self):
        """Detect EOU using VAD detector"""
        if not self.vad_detector:
            return False
        
        # Check if VAD detector indicates silence period
        return self.vad_detector.detect_silence_period()
    
    def _detect_vad_eou_with_details(self):
        """Detect EOU using VAD detector with detailed metrics"""
        if not self.vad_detector:
            return False, {}
        
        # Get detailed analysis from VAD detector
        analysis = self.vad_detector.get_recent_speech_proportion()
        
        # Check consecutive silence
        consecutive_silence_frames = self.vad_detector.consecutive_silence_frames
        silence_threshold = self.vad_detector.consecutive_silence_threshold
        
        # Check if silence period is detected
        is_silence_period = self.vad_detector.detect_silence_period()
        
        # Collect detailed metrics
        details = {
            "speech_proportion": analysis.get("speech_proportion", 0.0),
            "proportion_threshold": self.vad_detector.speech_proportion_threshold,
            "consecutive_silence_frames": consecutive_silence_frames,
            "silence_threshold": silence_threshold,
            "speech_frames": analysis.get("speech_frames", 0),
            "total_frames": analysis.get("total_frames", 0),
            "sufficient_data": analysis.get("sufficient_data", False),
            "silence_duration_seconds": consecutive_silence_frames * self.vad_detector.vad_frame_duration,
            "silence_threshold_seconds": silence_threshold * self.vad_detector.vad_frame_duration
        }
        
        return is_silence_period, details
    
    def _detect_silence_eou(self):
        """Simple silence-based EOU detection"""
        # This could be enhanced with additional silence detection logic
        # For now, just return False as VAD handles silence detection
        return False
    
    def _detect_buffer_vad_eou_with_details(self, text):
        """
        Detect EOU based on text buffer existence combined with low VAD activity
        
        Args:
            text: Current accumulated text in buffer
            
        Returns:
            tuple: (is_eou: bool, details: dict) - EOU result and detailed metrics
        """
        if not self.vad_detector or not text.strip():
            return False, {
                "error": "No VAD detector or empty text",
                "text_word_count": 0,
                "recent_speech_proportion": 1.0,
                "vad_threshold": 0.0
            }
        
        # Check text buffer criteria
        text_words = text.strip().split()
        min_text_words = self.config.get("buffer-vad-min-words", 3)  # Minimum words in buffer
        
        if len(text_words) < min_text_words:
            return False, {
                "error": "Insufficient text in buffer",
                "text_word_count": len(text_words),
                "min_words_required": min_text_words,
                "recent_speech_proportion": 0.0,
                "vad_threshold": 0.0
            }
        
        # Get recent VAD activity
        vad_lookback_seconds = self.config.get("buffer-vad-lookback-seconds", 1.0)  # How far back to check VAD
        vad_analysis = self.vad_detector.get_recent_speech_proportion(window_seconds=vad_lookback_seconds)
        
        if not vad_analysis.get("sufficient_data", False):
            return False, {
                "error": "Insufficient VAD data",
                "text_word_count": len(text_words),
                "recent_speech_proportion": 0.0,
                "vad_threshold": 0.0,
                "sufficient_vad_data": False
            }
        
        # Check if recent speech proportion is low (indicating silence after text accumulation)
        recent_speech_proportion = vad_analysis.get("speech_proportion", 1.0)
        vad_silence_threshold = self.config.get("buffer-vad-silence-threshold", 0.05)  # Very low activity threshold
        
        # EOU detected if we have text AND low recent speech activity
        is_buffer_vad_eou = recent_speech_proportion <= vad_silence_threshold
        
        details = {
            "text_word_count": len(text_words),
            "min_words_required": min_text_words,
            "recent_speech_proportion": recent_speech_proportion,
            "vad_threshold": vad_silence_threshold,
            "vad_lookback_seconds": vad_lookback_seconds,
            "vad_speech_frames": vad_analysis.get("speech_frames", 0),
            "vad_total_frames": vad_analysis.get("total_frames", 0),
            "sufficient_vad_data": vad_analysis.get("sufficient_data", False),
            "text_criteria_met": len(text_words) >= min_text_words,
            "vad_silence_criteria_met": recent_speech_proportion <= vad_silence_threshold
        }
        
        return is_buffer_vad_eou, details
    
    def reset(self):
        """Reset detector state"""
        if self.text_detector:
            self.text_detector.reset()

class TextEOUDetector(LoggerMixin):
    def __init__(self, log_config=None, eou_threshold=0.8, min_words=4, confirmation_needed=2):
        """Initialize the TurnSense text-based end-of-utterance detection model"""
        # Initialize logging first
        if log_config is None:
            log_config = {"log_level": "debug"}
        self.__init_logger__("TextEOU", **log_config)
        
        self.model_id = "latishab/turnsense"
        self.threshold = eou_threshold
        self.min_words_for_eou = min_words
        self.confirmation_needed = confirmation_needed
        self.recent_detections = []
        
        try:
            self.logger.info("Loading TurnSense EOU detection model...")
            self._load_model()
            self.logger.info("TurnSense EOU model loaded successfully")
        except Exception as e:
            self.logger.error(f"Could not load EOU detection model: {e}")
            self.logger.warning("Continuing without EOU detection...")
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
        
        # Remove final punctuation before EOU analysis
        cleaned_text = text.strip()
        if cleaned_text and cleaned_text[-1] in '.!?':
            cleaned_text = cleaned_text[:-1].strip()
        
        # Check minimum length requirement on cleaned text
        word_count = len(cleaned_text.split())
        if word_count < self.min_words_for_eou:
            self.logger.verbose(f"Text too short for EOU analysis: {word_count} < {self.min_words_for_eou} words")
            return False
        
        try:
            # Prepare input in the format expected by TurnSense using cleaned text
            formatted_text = f"<|user|> {cleaned_text} <|im_end|>"
            
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
            
            # Additional checks for natural sentence endings (using original text for context)
            text_lower = text.strip().lower()
            has_natural_ending = any(text_lower.endswith(ending) for ending in [
                '.', '?', '!', '. thank you', '. thanks', 'that\'s it', 'that is it'
            ])
            
            # Only trigger EOU if we have both model confidence AND natural ending indicators
            final_eou = is_eou and (has_natural_ending or eou_probability > 0.9)
            
            # Logging based on level
            self.logger.verbose(f"EOU analysis - probability: {eou_probability:.3f}, threshold: {self.threshold}, "
                              f"above_threshold: {is_above_threshold}, confirmations: {confirmation_count}/{self.confirmation_needed}")
            
            if final_eou:
                self.logger.debug(f"TEXT-EOU TRIGGERED - confidence: {eou_probability:.3f}, words: {word_count}, "
                                f"confirmations: {confirmation_count}, natural_ending: {has_natural_ending}")
                self.logger.info(f"End of utterance detected via text analysis")
                # Reset detection history
                self.recent_detections = []
            elif is_above_threshold:
                self.logger.verbose(f"Potential EOU detected but not confirmed - confidence: {eou_probability:.3f}")
            
            return final_eou
            
        except Exception as e:
            self.logger.error(f"EOU detection error: {e}")
            return False
    
    def detect_eou_with_details(self, text):
        """
        Detect if the given text represents an end of utterance with detailed metrics
        
        Args:
            text: Input text to analyze
            
        Returns:
            tuple: (is_eou: bool, details: dict) - EOU result and detailed metrics
        """
        if not self.tokenizer or not self.session or not text.strip():
            return False, {
                "error": "Model not available or empty text",
                "eou_probability": 0.0,
                "threshold": self.threshold,
                "word_count": 0,
                "min_words": self.min_words_for_eou,
                "confirmation_count": 0,
                "confirmation_needed": self.confirmation_needed
            }
        
        # Remove final punctuation before EOU analysis
        cleaned_text = text.strip()
        if cleaned_text and cleaned_text[-1] in '.!?':
            cleaned_text = cleaned_text[:-1].strip()
        
        # Check minimum length requirement on cleaned text
        word_count = len(cleaned_text.split())
        if word_count < self.min_words_for_eou:
            self.logger.verbose(f"Text too short for EOU analysis: {word_count} < {self.min_words_for_eou} words")
            return False, {
                "error": "Insufficient word count",
                "eou_probability": 0.0,
                "threshold": self.threshold,
                "word_count": word_count,
                "min_words": self.min_words_for_eou,
                "confirmation_count": 0,
                "confirmation_needed": self.confirmation_needed
            }
        
        try:
            # Prepare input in the format expected by TurnSense using cleaned text
            formatted_text = f"<|user|> {cleaned_text} <|im_end|>"
            
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
            
            # Additional checks for natural sentence endings (using original text for context)
            text_lower = text.strip().lower()
            has_natural_ending = any(text_lower.endswith(ending) for ending in [
                '.', '?', '!', '. thank you', '. thanks', 'that\'s it', 'that is it'
            ])
            
            # Only trigger EOU if we have both model confidence AND natural ending indicators
            final_eou = is_eou and (has_natural_ending or eou_probability > 0.9)
            
            # Prepare detailed results
            details = {
                "eou_probability": eou_probability,
                "threshold": self.threshold,
                "word_count": word_count,
                "min_words": self.min_words_for_eou,
                "confirmation_count": confirmation_count,
                "confirmation_needed": self.confirmation_needed,
                "is_above_threshold": is_above_threshold,
                "has_natural_ending": has_natural_ending,
                "recent_detections": list(self.recent_detections),
                "model_confidence_met": is_eou,
                "high_confidence_override": eou_probability > 0.9,
                "cleaned_text_length": len(cleaned_text)
            }
            
            # Logging based on level
            self.logger.verbose(f"EOU analysis - probability: {eou_probability:.3f}, threshold: {self.threshold}, "
                              f"above_threshold: {is_above_threshold}, confirmations: {confirmation_count}/{self.confirmation_needed}")
            
            if final_eou:
                self.logger.debug(f"TEXT-EOU TRIGGERED - confidence: {eou_probability:.3f}, words: {word_count}, "
                                f"confirmations: {confirmation_count}, natural_ending: {has_natural_ending}")
                self.logger.info(f"End of utterance detected via text analysis")
                # Reset detection history
                self.recent_detections = []
            elif is_above_threshold:
                self.logger.verbose(f"Potential EOU detected but not confirmed - confidence: {eou_probability:.3f}")
            
            return final_eou, details
            
        except Exception as e:
            self.logger.error(f"EOU detection error: {e}")
            return False, {
                "error": str(e),
                "eou_probability": 0.0,
                "threshold": self.threshold,
                "word_count": word_count,
                "min_words": self.min_words_for_eou,
                "confirmation_count": 0,
                "confirmation_needed": self.confirmation_needed
            }
    
    def reset(self):
        """Reset detection state"""
        self.recent_detections = []

class FrameLevelSpeechDetector(LoggerMixin):
    def __init__(self, vad_model_type="nvidia", config=None, log_config=None):
        """Analyze frame-level speech activity using configurable VAD model"""
        # Initialize logging first
        if log_config is None:
            log_config = {"log_level": "debug"}
        self.__init_logger__("VAD", **log_config)
        
        self.vad_model_type = vad_model_type.lower()
        
        # Frame-level analysis parameters
        if self.vad_model_type == "silero":
            # Silero VAD uses 32ms frames (512 samples at 16kHz)
            self.vad_frame_duration = 0.032  # 32ms per VAD frame for Silero
            self.window_size_samples = 512  # Fixed for 16kHz
        else:
            # NeMo VAD uses 20ms frames
            self.vad_frame_duration = 0.02  # 20ms per VAD frame for NeMo
        
        # Apply configuration parameters if provided
        if config:
            self.analysis_window_seconds = config.get("vad-analysis-window", 2.0)
            self.speech_proportion_threshold = config.get("vad-speech-proportion-threshold", 0.15)
            self.speech_probability_threshold = config.get("vad-speech-threshold", 0.25)
            self.consecutive_silence_threshold_seconds = config.get("vad-consecutive-silence-threshold", 0.8)
            self.min_frames_for_eou_seconds = config.get("vad-min-frames-for-eou", 0.5)
            self.lookback_seconds = config.get("vad-recent-speech-lookback-seconds", 1.5)
            self.recent_speech_threshold = config.get("vad-recent-speech-threshold", 0.10)
        else:
            # Default values
            self.analysis_window_seconds = 2.0
            self.speech_proportion_threshold = 0.15
            self.speech_probability_threshold = 0.25
            self.consecutive_silence_threshold_seconds = 0.8
            self.min_frames_for_eou_seconds = 0.5
            self.lookback_seconds = 1.5
            self.recent_speech_threshold = 0.10
            
        # Calculate frame-based values from time-based config
        self.frames_per_analysis_window = int(self.analysis_window_seconds / self.vad_frame_duration)
        self.min_frames_for_eou = int(self.min_frames_for_eou_seconds / self.vad_frame_duration)
        self.consecutive_silence_threshold = int(self.consecutive_silence_threshold_seconds / self.vad_frame_duration)
        
        # Rolling window to store frame-level speech detection results
        self.speech_frame_history = deque(maxlen=self.frames_per_analysis_window)
        
        # Consecutive silence detection for more responsive EOU
        self.consecutive_silence_frames = 0
        
        # Initialize VAD model based on type
        self.vad_model = None
        self._load_vad_model()
    
    def _load_vad_model(self):
        """Load the appropriate VAD model based on configuration"""
        try:
            if self.vad_model_type == "silero":
                self._load_silero_vad()
            else:  # Default to nvidia
                self._load_nvidia_vad()
        except Exception as e:
            self.logger.error(f"Could not load {self.vad_model_type} VAD model: {e}")
            self.logger.warning("Falling back to basic frame analysis...")
            self.vad_model = None
    
    def _load_nvidia_vad(self):
        """Load NeMo VAD model"""
        self.logger.info("Loading NeMo VAD model...")
        
        # Suppress NeMo logging during model load
        nemo_logger = logging.getLogger('nemo_logger')
        old_level = nemo_logger.level
        nemo_logger.setLevel(logging.CRITICAL)
        
        try:
            self.vad_model = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(
                model_name="nvidia/frame_vad_multilingual_marblenet_v2.0"
            )
            
            if torch.cuda.is_available():
                self.vad_model = self.vad_model.cuda()
                self.logger.info(f"NeMo VAD model moved to GPU: {torch.cuda.get_device_name()}")
            
            self.vad_model.eval()
            self.logger.info("NeMo VAD model loaded successfully")
            self.logger.debug(f"VAD analysis window: {self.analysis_window_seconds}s ({self.frames_per_analysis_window} frames)")
            
        finally:
            nemo_logger.setLevel(old_level)
    
    def _load_silero_vad(self):
        """Load Silero VAD model"""
        self.logger.info("Loading Silero VAD model...")
        torch.set_num_threads(1)
        
        try:
            self.logger.debug("Loading Silero VAD via torch.hub...")
            self.vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                verbose=False
            )
            
        except Exception as torch_hub_error:
            self.logger.warning(f"torch.hub loading failed: {torch_hub_error}")
            self.logger.debug("Trying silero-vad package as fallback...")
            
            try:
                from silero_vad import load_silero_vad
                self.vad_model = load_silero_vad(onnx=False)
            except ImportError as import_error:
                raise Exception(f"Both torch.hub and silero-vad package failed. torch.hub error: {torch_hub_error}, import error: {import_error}")
        
        # Move to appropriate device
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            self.vad_model = self.vad_model.to(device)
            self.logger.info(f"Silero VAD model moved to GPU: {torch.cuda.get_device_name(device)}")
        else:
            self.vad_model = self.vad_model.cpu()
            self.logger.info("Silero VAD model using CPU (no GPU available)")
        
        self.vad_model.eval()
        self.logger.info("Silero VAD model loaded successfully")
        self.logger.debug(f"Frame duration: {self.vad_frame_duration*1000:.1f}ms, Window size: {self.window_size_samples} samples")
    
    def analyze_audio_vad(self, audio_signal):
        """
        Analyze audio using configured VAD model to detect speech activity per frame
        
        Args:
            audio_signal: Raw audio signal (numpy array, float32, 16kHz)
            
        Returns:
            list: List of boolean values indicating speech detection for each frame
        """
        try:
            if self.vad_model is None or len(audio_signal) == 0:
                return []
            
            if self.vad_model_type == "silero":
                return self._analyze_silero_vad(audio_signal)
            else:
                return self._analyze_nvidia_vad(audio_signal)
            
        except Exception as e:
            self.logger.error(f"VAD analysis error: {e}")
            return []
    
    def _analyze_nvidia_vad(self, audio_signal):
        """Analyze audio using NeMo VAD model"""
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
                
                # Apply softmax to convert logits to probabilities if needed
                if np.any(speech_probs < 0) or np.any(speech_probs > 1):
                    # These look like logits, apply softmax
                    exp_probs = np.exp(vad_probs - np.max(vad_probs, axis=1, keepdims=True))
                    softmax_probs = exp_probs / np.sum(exp_probs, axis=1, keepdims=True)
                    speech_probs = softmax_probs[:, 1]  # Speech probabilities after softmax
                
                # Now apply threshold to get binary decisions
                speech_detected_frames = (speech_probs > self.speech_probability_threshold).tolist()
                
                if len(speech_detected_frames) > 0:
                    # Debug: show some frame stats occasionally
                    avg_speech_prob = np.mean(speech_probs)
                    speech_frame_count = sum(speech_detected_frames)
                    if len(self.speech_frame_history) % 50 == 0:  # Log every 50 updates
                        self.logger.verbose(f"NeMo VAD: {len(speech_detected_frames)} frames, "
                              f"avg_speech_prob: {avg_speech_prob:.3f}, "
                              f"speech_frames: {speech_frame_count}, "
                              f"threshold: {self.speech_probability_threshold}")
                return speech_detected_frames
            else:
                self.logger.warning(f"Unexpected NeMo VAD output shape: {vad_probs.shape} - expected [frames, 2]")
                return []
        elif vad_probs.ndim == 1:
            # Single dimension - treat as speech probabilities directly
            # Apply sigmoid if values look like logits
            if np.any(vad_probs < 0) or np.any(vad_probs > 1):
                speech_probs = 1.0 / (1.0 + np.exp(-vad_probs))  # Sigmoid
            else:
                speech_probs = vad_probs
            
            speech_detected_frames = (speech_probs > self.speech_probability_threshold).tolist()
            return speech_detected_frames
        else:
            self.logger.warning(f"Unexpected NeMo VAD output shape: {vad_probs.shape}")
            return []
    
    def _analyze_silero_vad(self, audio_signal):
        """Analyze audio using Silero VAD model"""
        # Ensure audio is the right format
        if not isinstance(audio_signal, np.ndarray):
            audio_signal = np.array(audio_signal)
        
        if audio_signal.dtype != np.float32:
            audio_signal = audio_signal.astype(np.float32)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_signal)
        
        # Move to same device as model
        device = next(self.vad_model.parameters()).device
        audio_tensor = audio_tensor.to(device)
        
        # Silero VAD processes audio in 32ms chunks (512 samples at 16kHz)
        speech_probs = []
        speech_detected_frames = []
        
        with torch.no_grad():
            # Process audio in chunks
            for i in range(0, len(audio_tensor), self.window_size_samples):
                chunk = audio_tensor[i:i + self.window_size_samples]
                
                # Pad if chunk is too short
                if len(chunk) < self.window_size_samples:
                    padding = torch.zeros(self.window_size_samples - len(chunk), device=device)
                    chunk = torch.cat([chunk, padding])
                
                # Get speech probability for this chunk
                speech_prob = self.vad_model(chunk, 16000).item()
                speech_probs.append(speech_prob)
                
                # Apply threshold to get binary decision
                is_speech = speech_prob > self.speech_probability_threshold
                speech_detected_frames.append(is_speech)
        
        # Debug output occasionally
        if len(speech_detected_frames) > 0 and len(self.speech_frame_history) % 50 == 0:
            avg_speech_prob = np.mean(speech_probs)
            speech_frame_count = sum(speech_detected_frames)
            self.logger.verbose(f"Silero VAD: {len(speech_detected_frames)} frames, "
                  f"avg_speech_prob: {avg_speech_prob:.3f}, "
                  f"speech_frames: {speech_frame_count}, "
                  f"threshold: {self.speech_probability_threshold}")
        
        # Reset model states after processing
        self.vad_model.reset_states()
        
        return speech_detected_frames
    
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
            silence_duration = self.consecutive_silence_frames * self.vad_frame_duration
            self.logger.debug(f"VAD-EOU TRIGGERED: Consecutive silence - {silence_duration:.1f}s ({self.consecutive_silence_frames} frames)")
            return True
        
        # Method 2: Proportion-based analysis (smoother)
        analysis = self.get_recent_speech_proportion()
        
        if not analysis["sufficient_data"]:
            return False
        
        # Detect silence if speech proportion is below threshold
        is_silence_period = analysis["speech_proportion"] < self.speech_proportion_threshold
        
        if is_silence_period:
            self.logger.debug(f"VAD-EOU TRIGGERED: Low speech proportion - {analysis['speech_proportion']:.3f} "
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
            "avg_confidence": analysis["speech_proportion"]
        }
    
    def has_recent_speech_activity(self, lookback_seconds=None):
        """
        Check if there has been significant speech activity in recent history
        
        Args:
            lookback_seconds: How far back to look for speech activity (uses config default if None)
            
        Returns:
            bool: True if there has been recent speech activity
        """
        if lookback_seconds is None:
            lookback_seconds = self.lookback_seconds
            
        lookback_analysis = self.get_recent_speech_proportion(window_seconds=lookback_seconds)
        
        if not lookback_analysis["sufficient_data"]:
            return False
        
        # Use configurable threshold for recent speech activity
        return lookback_analysis["speech_proportion"] > self.recent_speech_threshold
    
    def reset_silence_tracking(self):
        """Reset consecutive silence tracking"""
        self.consecutive_silence_frames = 0
    
    # Keep legacy method for compatibility
    def analyze_model_outputs(self, pred_outputs, processed_signal_length=None):
        """Fallback method - kept for compatibility but VAD is preferred"""
        return {"speech_detected": True, "confidence": 0.5, "frame_activity": 0.5}

class WebSocketServer(LoggerMixin):
    def __init__(self, host='localhost', port=8765, log_config=None):
        """Initialize WebSocket server for real-time ASR output"""
        if log_config is None:
            log_config = {"log_level": "debug"}
        self.__init_logger__("WebSocket", **log_config)
        
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None
        self.loop = None
        self.thread = None
        self.running = False
        # Add command queue for processing control commands
        self.command_queue = queue.Queue()
        # Add state management
        self.current_state = 'running'
        self.start_time = time.time()
        
    async def register_client(self, websocket):
        """Register a new WebSocket client"""
        self.clients.add(websocket)
        self.logger.info(f"Client connected from {websocket.remote_address}")
        try:
            # Send welcome message
            welcome_msg = {
                "type": "status",
                "status": "connected",
                "details": {"server": "ASR WebSocket Server", "version": "1.0"},
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(welcome_msg))
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    command = json.loads(message)
                    await self.handle_command(command, websocket)
                except json.JSONDecodeError:
                    error_msg = {
                        "type": "error",
                        "error": "Invalid JSON format",
                        "timestamp": time.time()
                    }
                    await websocket.send(json.dumps(error_msg))
                except Exception as e:
                    error_msg = {
                        "type": "error", 
                        "error": str(e),
                        "timestamp": time.time()
                    }
                    await websocket.send(json.dumps(error_msg))
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            self.logger.error(f"Error with WebSocket client: {e}")
        finally:
            if websocket in self.clients:
                self.clients.remove(websocket)
            self.logger.info(f"WebSocket client disconnected")
    
    async def handle_command(self, command, websocket):
        """Enhanced command handler with state management support"""
        if not isinstance(command, dict) or "type" not in command:
            await websocket.send(json.dumps({
                "type": "error",
                "error": "Command must be a JSON object with 'type' field",
                "timestamp": time.time()
            }))
            return
        
        command_type = command.get("type")
        
        if command_type == "control":
            action = command.get("action")
            
            # Enhanced control actions
            if action in ["pause", "resume", "purge", "reset", "stop", "ping", "get_state"]:
                # Add command to queue for main thread processing
                self.command_queue.put({
                    "action": action,
                    "timestamp": time.time(),
                    "client": websocket.remote_address if hasattr(websocket, 'remote_address') else "unknown",
                    "healthCheck": command.get("healthCheck", False)
                })
                
                # Send immediate acknowledgment
                response = {
                    "type": "command_ack",
                    "action": action,
                    "status": "queued",
                    "timestamp": time.time()
                }
                await websocket.send(json.dumps(response))
                
                # Handle immediate responses for certain commands
                if action == "ping":
                    # Send pong response for health checks
                    pong_response = {
                        "type": "pong" if command.get("healthCheck") else "status",
                        "status": getattr(self, 'current_state', 'running'),
                        "healthCheck": command.get("healthCheck", False),
                        "timestamp": time.time()
                    }
                    await websocket.send(json.dumps(pong_response))
                
                elif action == "get_state":
                    # Send current state immediately - always use current_state from ASR system
                    current_actual_state = getattr(self, 'current_state', 'running')
                    state_response = {
                        "type": "status",
                        "status": current_actual_state,
                        "details": {
                            "uptime": time.time() - getattr(self, 'start_time', time.time()),
                            "clients_connected": len(self.clients),
                            "is_processing": current_actual_state == 'running'
                        },
                        "timestamp": time.time()
                    }
                    await websocket.send(json.dumps(state_response))
                
                self.logger.verbose(f"WebSocket command received: {action}")
                    
            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": f"Unknown control action: {action}",
                    "timestamp": time.time()
                }))
        else:
            await websocket.send(json.dumps({
                "type": "error", 
                "error": f"Unknown command type: {command_type}",
                "timestamp": time.time()
            }))
    
    def get_pending_commands(self):
        """Get all pending commands from the queue"""
        commands = []
        while not self.command_queue.empty():
            try:
                commands.append(self.command_queue.get_nowait())
            except queue.Empty:
                break
        return commands
    
    async def broadcast_message(self, message):
        """Broadcast message to all connected clients"""
        if not self.clients:
            return
            
        # Create a copy of clients to avoid modification during iteration
        clients_copy = self.clients.copy()
        disconnected_clients = set()
        
        for client in clients_copy:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                self.logger.error(f"Error sending to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients
    
    def send_partial_transcription(self, text, confidence=None):
        """Send partial transcription update"""
        if not self.running or not self.loop or not self.clients:
            return
            
        message = {
            "type": "partial",
            "text": text,
            "timestamp": time.time(),
            "confidence": confidence
        }
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.broadcast_message(message), 
                self.loop
            )
            # Don't wait for completion to avoid blocking
        except Exception as e:
            self.logger.error(f"Error sending partial transcription: {e}")
    
    def send_complete_utterance(self, text, confidence=None):
        """Send complete utterance"""
        if not self.running or not self.loop or not self.clients:
            return
            
        message = {
            "type": "complete",
            "text": text,
            "timestamp": time.time(),
            "confidence": confidence
        }
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.broadcast_message(message), 
                self.loop
            )
            # Don't wait for completion to avoid blocking
        except Exception as e:
            self.logger.error(f"Error sending complete utterance: {e}")
    
    def send_status_update(self, status, details=None):
        """Send status update"""
        if not self.running or not self.loop:
            return
            
        message = {
            "type": "status",
            "status": status,
            "details": details,
            "timestamp": time.time()
        }
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.broadcast_message(message), 
                self.loop
            )
            # Don't wait for completion to avoid blocking
        except Exception as e:
            self.logger.error(f"Error sending status update: {e}")
    
    async def send_state_update(self, state, details=None):
        """Send state update to all connected clients"""
        self.current_state = state
        message = {
            "type": "status",
            "status": state,
            "details": details or {},
            "timestamp": time.time()
        }
        await self.broadcast_message(message)
        
        self.logger.debug(f"State updated to: {state}")
    
    def get_current_state(self):
        """Get current processing state"""
        return getattr(self, 'current_state', 'running')
    
    def start_server(self):
        """Start WebSocket server in a separate thread"""
        def run_server():
            try:
                # Create new event loop for this thread
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                
                # Start the WebSocket server
                async def start_server_async():
                    try:
                        self.server = await websockets.serve(
                            self.register_client, 
                            self.host, 
                            self.port,
                            ping_interval=20,
                            ping_timeout=10
                        )
                        self.running = True
                        self.logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
                        
                        # Keep the server running
                        await self.server.wait_closed()
                    except Exception as e:
                        self.logger.error(f"WebSocket server startup error: {e}")
                        self.running = False
                
                # Run the server
                self.loop.run_until_complete(start_server_async())
                
            except Exception as e:
                self.logger.error(f"WebSocket server thread error: {e}")
                self.running = False
            finally:
                if self.loop and not self.loop.is_closed():
                    self.loop.close()
        
        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        
        # Give server time to start
        time.sleep(1.0)
        
        # Check if server started successfully
        return self.running
    
    def stop_server(self):
        """Stop WebSocket server"""
        if self.running and self.loop and self.server:
            try:
                # Schedule server close in the event loop
                future = asyncio.run_coroutine_threadsafe(
                    self.server.close(),
                    self.loop
                )
                # Wait for close to complete
                future.result(timeout=2.0)
                
                # Stop the event loop
                self.loop.call_soon_threadsafe(self.loop.stop)
                self.running = False
                
                self.logger.info("WebSocket server stopped")
            except Exception as e:
                self.logger.error(f"Error stopping WebSocket server: {e}")

class OnlineASRWithPunctuation(LoggerMixin):
    def __init__(self, asr_model_name, punct_model_name=None, lookahead_size=480, decoder_type="rnnt", 
                 quiet_mode=False, enable_eou=True, websocket_server=None, log_config=None):
        """
        Initialize the Online ASR system with punctuation and frame-level EOU detection
        
        Args:
            asr_model_name: Name of the pretrained ASR model to use
            punct_model_name: Name of the punctuation model (None to disable)
            lookahead_size: Lookahead size in milliseconds
            decoder_type: "rnnt" or "ctc"
            quiet_mode: If True, suppress all logging (legacy, now use log_config)
            enable_eou: If True, enable end-of-utterance detection
            websocket_server: WebSocketServer instance for real-time output
            log_config: Logging configuration dictionary
        """
        # Initialize logging first
        if log_config is None:
            log_config = {"log_level": "debug"}
        self.__init_logger__("ASR", **log_config)
        self.asr_model_name = asr_model_name
        self.punct_model_name = punct_model_name
        self.lookahead_size = lookahead_size
        self.decoder_type = decoder_type
        self.quiet_mode = quiet_mode
        self.enable_eou = enable_eou
        self.websocket_server = websocket_server
        
        # Add enhanced state management
        self.current_state = 'running'
        self.is_paused = False
        self.is_stopped = False
        self.last_command_check = time.time()
        self.command_check_interval = 0.1  # Check for commands every 100ms
        
        # Store start time for uptime reporting
        self.start_time = time.time()
        if websocket_server:
            websocket_server.start_time = self.start_time
            websocket_server.current_state = self.current_state

        # Initialize models
        self._setup_asr_model()
        if punct_model_name:
            self._setup_punctuation_model()
        else:
            self.punct_model = None
            
        # Initialize unified EOU detection (will be configured later in main)
        if enable_eou:
            self.unified_eou_detector = None  # Will be initialized in main with full config
        else:
            self.unified_eou_detector = None
            
        # Initialize frame-level speech detector with default model (will be reconfigured in main)
        self.frame_detector = FrameLevelSpeechDetector("nvidia", None, log_config)
        
        # Text buffer for punctuation processing and conversation tracking
        self.text_buffer = deque(maxlen=50)
        self.conversation_buffer = []  # Store full conversation for EOU analysis
        self.current_utterance = ""  # Accumulate text for complete utterance output
        self._setup_preprocessing()
        self._reset_streaming_state()
    
    def _setup_asr_model(self):
        """Load and configure the ASR model"""
        self.logger.info(f"Loading ASR model: {self.asr_model_name}")
        
        # Suppress NeMo logging during model loading
        nemo_logger = logging.getLogger('nemo_logger')
        old_level = nemo_logger.level
        nemo_logger.setLevel(logging.CRITICAL)
        
        try:
            # Suppress progress bars during model loading if not in verbose mode
            current_log_level = logging.getLogger().level
            if current_log_level != VERBOSE:
                with NoStdStreams():
                    self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.asr_model_name)
            else:
                self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.asr_model_name)
        finally:
            nemo_logger.setLevel(old_level)
        
        # Update attention context size for multi-lookahead model
        if self.asr_model_name == "stt_en_fastconformer_hybrid_large_streaming_multi":
            if self.lookahead_size not in [0, 80, 480, 1040]:
                raise ValueError(
                    f"Lookahead size {self.lookahead_size} not valid for multi model. "
                    "Must be one of: 0, 80, 480, 1040 ms"
                )
            
            # Update attention context
            left_context_size = self.asr_model.encoder.att_context_size[0]
            right_context_size = int(self.lookahead_size / ENCODER_STEP_LENGTH)
            self.asr_model.encoder.set_default_att_context_size([left_context_size, right_context_size])
            self.logger.debug(f"Set attention context: [{left_context_size}, {right_context_size}]")
        
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
            self.logger.info(f"ASR model moved to GPU: {torch.cuda.get_device_name()}")
            
        self.logger.debug(f"Using ASR decoder: {self.decoder_type}")
        self.logger.debug(f"Lookahead size: {self.lookahead_size}ms")
    
    def _setup_punctuation_model(self):
        """Load and configure the punctuation model"""
        self.logger.info(f"Loading punctuation model: {self.punct_model_name}")
        
        try:
            # Suppress NeMo logging during model loading
            nemo_logger = logging.getLogger('nemo_logger')
            old_level = nemo_logger.level
            nemo_logger.setLevel(logging.CRITICAL)
            
            try:
                # Suppress progress bars during model loading if not in verbose mode
                current_log_level = logging.getLogger().level
                if current_log_level != VERBOSE:
                    with NoStdStreams():
                        self.punct_model = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained(
                            model_name=self.punct_model_name
                        )
                else:
                    self.punct_model = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained(
                        model_name=self.punct_model_name
                    )
            finally:
                nemo_logger.setLevel(old_level)
                
            self.punct_model.eval()
            
            if torch.cuda.is_available():
                self.punct_model = self.punct_model.cuda()
                self.logger.info("Punctuation model moved to GPU")
                
        except Exception as e:
            self.logger.warning(f"Could not load punctuation model {self.punct_model_name}: {e}")
            self.logger.warning("Continuing without punctuation...")
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
        
    def _reset_streaming_state(self, reset_conversation=True, send_websocket_updates=True):
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
        
        # Send reset notification via WebSocket with follow-up running status
        if send_websocket_updates and self.websocket_server and self.websocket_server.loop:
            try:
                # Send reset status ONCE
                asyncio.run_coroutine_threadsafe(
                    self.websocket_server.send_state_update('reset', {
                        "conversation_reset": reset_conversation, 
                        "timestamp": time.time()
                    }),
                    self.websocket_server.loop
                )
                
                # Schedule follow-up "running" state after brief delay
                async def send_running_after_reset():
                    await asyncio.sleep(0.1)  # 100ms delay
                    # Update WebSocket server state to running
                    self.websocket_server.current_state = 'running'
                    await self.websocket_server.send_state_update('running', {
                        'auto_reset_complete': True,
                        'timestamp': time.time()
                    })
                
                asyncio.run_coroutine_threadsafe(
                    send_running_after_reset(),
                    self.websocket_server.loop
                )
                
            except Exception as e:
                self.logger.error(f"Error sending reset state updates: {e}")
        
        self.logger.debug("Complete ASR streaming state reset")
    
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
            # Suppress output during punctuation in quiet mode or if not in verbose mode
            current_log_level = logging.getLogger().level
            suppress_output = self.quiet_mode or (current_log_level != VERBOSE)
            
            if suppress_output:
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
            self.logger.error(f"Punctuation error: {e}")
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
            self.logger.debug("TEXT-EOU: End of utterance confirmed, resetting context")
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
                
                self.logger.info(f"VAD-EOU: End of utterance detected based on speech analysis")
                self.logger.debug(f"  Recent speech proportion: {summary['avg_activity']:.3f}")
                self.logger.debug(f"  Speech frames: {summary['speech_frames']}/{summary['total_frames']}")
                self.logger.debug(f"  Consecutive silence frames: {self.frame_detector.consecutive_silence_frames}")
                
                # Reset frame history and silence tracking for fresh start
                self.frame_detector.speech_frame_history.clear()
                self.frame_detector.reset_silence_tracking()
                return True
            else:
                self.logger.verbose(f"VAD: Silence detected but no recent speech activity - not triggering EOU")
        
        return False
    
    def transcribe_chunk(self, audio_chunk):
        """
        Transcribe a single audio chunk with punctuation and VAD-based EOU detection
        
        Args:
            audio_chunk: numpy array of audio samples (int16)
            
        Returns:
            tuple: (raw_text, punctuated_text, is_eou, complete_utterance)
        """
        # Check if processing should continue (includes command checking)
        if not self.should_process_audio():
            return "", "", False, None
        
        try:
            # Debug audio chunk input (very detailed)
            if self.step_num % 50 == 0:
                self.logger.verbose(f"Processing chunk #{self.step_num} - size: {len(audio_chunk)}, "
                      f"dtype: {audio_chunk.dtype}, range: [{np.min(audio_chunk)}, {np.max(audio_chunk)}]")
            
            # Convert int16 to float32 and normalize
            audio_data = audio_chunk.astype(np.float32) / 32768.0
            
            # VAD-based EOU detection using raw audio (before preprocessing)
            is_frame_eou = self._check_end_of_utterance_frame_based(audio_data)
            
            # Get mel-spectrogram
            processed_signal, processed_signal_length = self._preprocess_audio(audio_data)
            
            # Debug preprocessing output (very detailed)
            if self.step_num % 50 == 0:
                self.logger.verbose(f"Preprocessed signal shape: {processed_signal.shape}, "
                      f"length: {processed_signal_length}")
            
            # Prepend with pre-encode cache
            processed_signal = torch.cat([self.cache_pre_encode, processed_signal], dim=-1)
            processed_signal_length += self.cache_pre_encode.shape[1]
            
            # Update cache for next iteration
            self.cache_pre_encode = processed_signal[:, :, -self.pre_encode_cache_size:]
            
            # Debug tensor shapes before ASR (very detailed)
            if self.step_num % 50 == 0:
                self.logger.verbose(f"Final signal shape: {processed_signal.shape}, "
                      f"length: {processed_signal_length}, "
                      f"cache shapes: [{self.cache_last_channel.shape if self.cache_last_channel is not None else None}, "
                      f"{self.cache_last_time.shape if self.cache_last_time is not None else None}]")
            
            # Run streaming inference
            with torch.no_grad():
                try:
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
                    
                    # Debug ASR model output (huge output, so limit frequency)
                    if self.step_num % 200 == 0:
                        self.logger.verbose(f"Model output type: {type(transcribed_texts)}")
                        if isinstance(transcribed_texts, list):
                            self.logger.verbose(f"Model output length: {len(transcribed_texts)}")

                except Exception as asr_error:
                    self.logger.error(f"ASR ERROR in conformer_stream_step: {asr_error}", exc_info=True)
                    # Return empty results on ASR error
                    return "", "", False, None
            
            # Extract transcription
            final_transcriptions = self._extract_transcriptions(transcribed_texts)
            raw_text = final_transcriptions[0] if final_transcriptions else ""
            
            # Debug transcription extraction
            if (raw_text.strip() or self.step_num % 100 == 0):
                self.logger.verbose(f"Extracted transcription: '{raw_text}'")
            
            # Unified EOU detection combining multiple methods
            is_eou = False
            eou_result = None
            
            if self.unified_eou_detector and raw_text.strip():
                # Add current text to conversation buffer for text-based EOU
                self.conversation_buffer.append(raw_text)
                
                # Only check EOU periodically to avoid too frequent calls
                if len(self.conversation_buffer) % 3 != 0:  # Check every 3rd update
                    pass
                else:
                    # Analyze using unified EOU detector
                    full_conversation = " ".join(self.conversation_buffer)
                    eou_result = self.unified_eou_detector.detect_eou(
                        text=full_conversation, 
                        audio_data=audio_data
                    )
                    is_eou = eou_result["is_eou"]
                    
                    # Log detailed EOU information
                    if eou_result["is_eou"]:
                        self.logger.info(f"Unified EOU triggered - confidence: {eou_result['confidence']:.3f}, "
                                       f"methods: {eou_result['triggered_by']}")
                        self.logger.debug(f"Method scores: {eou_result['method_scores']}")
            elif is_frame_eou:
                # Fallback to legacy frame-based EOU if unified detector not available
                is_eou = is_frame_eou
            
            # Apply punctuation if enabled
            if self.punct_model and raw_text.strip():
                punctuated_text = self._apply_punctuation(raw_text)
            else:
                punctuated_text = raw_text
            
            # Accumulate current utterance for complete output
            if punctuated_text.strip():
                self.current_utterance = punctuated_text
            
            # Send partial transcription via WebSocket
            if self.websocket_server and punctuated_text.strip():
                self.websocket_server.send_partial_transcription(punctuated_text)
            
            # Handle EOU detection and complete utterance output
            complete_utterance = None
            if is_eou:
                # Check if there was recent speech activity or accumulated utterance text
                has_recent_speech = self.frame_detector.has_recent_speech_activity(lookback_seconds=2.0)
                has_accumulated_text = bool(self.current_utterance.strip())
                
                if has_recent_speech or has_accumulated_text:
                    # Store the final complete utterance for output
                    complete_utterance = self.current_utterance.strip() if self.current_utterance.strip() else punctuated_text.strip()
                    
                    # Send complete utterance via WebSocket
                    if self.websocket_server and complete_utterance:
                        self.websocket_server.send_complete_utterance(complete_utterance)
                    
                    # Always log complete utterances at CRITICAL level to ensure they're always shown
                    self.logger.critical(f"COMPLETE UTTERANCE: {complete_utterance}")
                    
                    eou_type = "VAD" if is_frame_eou else "Text"
                    self.logger.debug(f"{eou_type}-EOU: End of utterance confirmed with recent speech or accumulated text, performing internal ASR reset")
                    
                    # INTERNAL ASR RESET - Reset conversation buffer and streaming state silently
                    # This is just internal cleanup, no status messages needed for clients
                    self.conversation_buffer = []
                    self._reset_streaming_state(reset_conversation=False, send_websocket_updates=False)
                else:
                    self.logger.verbose(f"EOU detected but no recent speech activity or accumulated text - skipping reset")
                    # Don't reset, just continue processing
                    complete_utterance = None

            self.step_num += 1
            
            return raw_text, punctuated_text, is_eou, complete_utterance
        except Exception as e:
            self.logger.error(f"ASR ERROR in transcribe_chunk: {e}", exc_info=True)
            return "", "", False, None
    
    def check_and_process_commands(self):
        """Check for pending WebSocket commands and process them"""
        if not self.websocket_server:
            return
        
        current_time = time.time()
        if current_time - self.last_command_check < self.command_check_interval:
            return
        
        self.last_command_check = current_time
        
        # Get all pending commands
        commands = self.websocket_server.get_pending_commands()
        
        for command in commands:
            action = command.get("action")
            self.logger.verbose(f"Processing command: {action}")

            if action == "pause":
                self.is_paused = True
                self.current_state = 'paused'
                self.logger.info("Received pause command - Pausing audio processing")
                
                # Manual purge and reset without WebSocket updates
                self.conversation_buffer = []
                self.current_utterance = ""
                self._reset_streaming_state(reset_conversation=True, send_websocket_updates=False)
                
                # Reset frame detector state
                if self.frame_detector:
                    self.frame_detector.speech_frame_history.clear()
                    self.frame_detector.reset_silence_tracking()
                
                # Reset unified EOU detector state
                if self.unified_eou_detector:
                    self.unified_eou_detector.reset()
                
                # Send paused status
                if self.websocket_server and self.websocket_server.loop:
                    # Update WebSocket server state
                    self.websocket_server.current_state = self.current_state
                    # Use asyncio to send state update
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self.websocket_server.send_state_update('paused', {
                                'action': 'pause',
                                'purged': True,
                                'timestamp': time.time()
                            }),
                            self.websocket_server.loop
                        )
                    except Exception as e:
                        self.logger.error(f"Error sending pause state update: {e}")
                
            elif action == "resume":
                self.is_paused = False
                self.current_state = 'running'
                self.logger.info("Received resume command - Resuming audio processing")
                if self.websocket_server and self.websocket_server.loop:
                    # Update WebSocket server state
                    self.websocket_server.current_state = self.current_state
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self.websocket_server.send_state_update('running', {
                                'action': 'resume',
                                'timestamp': time.time()
                            }),
                            self.websocket_server.loop
                        )
                    except Exception as e:
                        self.logger.error(f"Error sending resume state update: {e}")
                
            elif action == "stop":
                self.is_stopped = True
                self.current_state = 'stopped'
                self.logger.info("Received stop command - Stopping audio processing")
                if self.websocket_server and self.websocket_server.loop:
                    # Update WebSocket server state
                    self.websocket_server.current_state = self.current_state
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self.websocket_server.send_state_update('stopped', {
                                'action': 'stop',
                                'timestamp': time.time()
                            }),
                            self.websocket_server.loop
                        )
                    except Exception as e:
                        self.logger.error(f"Error sending stop state update: {e}")
                
            elif action == "purge" or action == "reset":
                # Clear internal buffers first - this will send reset + running messages
                self._purge_and_reset()
                
                # Update state flags
                self.is_paused = False
                self.is_stopped = False
                self.current_state = 'running'  # Set to running immediately
                self.logger.info(f"Received {action} command - Purging buffers and resetting ASR state")
                
                # No need to send additional reset messages since _purge_and_reset() handles it

    def should_process_audio(self):
        """Check if audio processing should continue"""
        # Always check for commands first
        self.check_and_process_commands()
        
        # Return processing state
        return not (self.is_paused or self.is_stopped)

    def _check_websocket_commands(self):
        """Legacy method - now calls enhanced check_and_process_commands"""
        self.check_and_process_commands()
    
    def _purge_and_reset(self):
        """Purge current utterance and reset ASR and EOU detection"""
        # Reset conversation buffer and streaming state
        self.conversation_buffer = []
        self.current_utterance = ""
        self._reset_streaming_state(reset_conversation=True, send_websocket_updates=True)
        
        # Reset frame detector state
        if self.frame_detector:
            self.frame_detector.speech_frame_history.clear()
            self.frame_detector.reset_silence_tracking()
        
        # Reset unified EOU detector state
        if self.unified_eou_detector:
            self.unified_eou_detector.reset()
    
def list_audio_devices():
    """List available audio input devices"""
    p = pa.PyAudio()
    
    # Get logger for this function
    logger = logging.getLogger("haku_stt.audio_devices")
    logger.info('Available audio input devices:')
    input_devices = []
    
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels'):
            input_devices.append(i)
            logger.info(f"  {i}: {dev.get('name')} (channels: {dev.get('maxInputChannels')})")
    
    # Add remote audio stream option if available
    if REMOTE_AUDIO_AVAILABLE:
        logger.info(f"  remote: Remote audio stream (GStreamer RTP)")
    
    p.terminate()
    return input_devices

def run_streaming_asr_with_punct(asr_system, device_id=None, chunk_size_ms=None, show_raw=False, quiet_mode=False, remote_audio_port=5004):
    """
    Run streaming ASR with punctuation and EOU detection using microphone or remote audio stream
    
    Args:
        asr_system: OnlineASRWithPunctuation instance
        device_id: Audio device ID (None for interactive selection, 'remote' for remote stream)
        chunk_size_ms: Chunk size in milliseconds (None for automatic)
        show_raw: Whether to show raw (unpunctuated) text
        quiet_mode: If True, suppress all logs and only show punctuated output
        remote_audio_port: Port for remote audio stream
    """
    # Calculate chunk size
    if chunk_size_ms is None:
        chunk_size_ms = asr_system.lookahead_size + ENCODER_STEP_LENGTH
    
    if not quiet_mode:
        logger = logging.getLogger("haku_stt.streaming")
        logger.info(f"Using chunk size: {chunk_size_ms}ms")
    
    # Send initial status via WebSocket
    if asr_system.websocket_server:
        asr_system.websocket_server.send_status_update(
            "started", 
            {"chunk_size_ms": chunk_size_ms, "device_id": device_id}
        )
    
    # Check if remote audio stream is requested
    use_remote_audio = (device_id == 'remote' or device_id == -1) and REMOTE_AUDIO_AVAILABLE
    
    if use_remote_audio:
        # Use remote audio stream
        return run_streaming_asr_with_remote_audio(
            asr_system, remote_audio_port, chunk_size_ms, show_raw, quiet_mode
        )
    else:
        # Use local microphone (existing code)
        return run_streaming_asr_with_microphone(
            asr_system, device_id, chunk_size_ms, show_raw, quiet_mode
        )

def run_streaming_asr_with_remote_audio(asr_system, remote_port, chunk_size_ms, show_raw, quiet_mode):
    """Run streaming ASR using remote audio stream - PyAudio compatible"""
    
    logger = logging.getLogger("haku_stt.remote_audio")
    
    if not quiet_mode:
        logger.info(f"Starting remote audio stream receiver on port {remote_port}")
        logger.info("Waiting for audio stream from remote device...")
    
    # Initialize remote audio stream
    remote_stream = RemoteAudioStream(
        listen_port=remote_port, 
        sample_rate=SAMPLE_RATE, 
        verbose=not quiet_mode
    )
    
    if not remote_stream.start():
        logger.error("Failed to start remote audio stream")
        return
    
    try:
        # Calculate frames per buffer - EXACTLY like PyAudio
        frames_per_buffer = int(SAMPLE_RATE * chunk_size_ms / 1000) - 1
        
        if not quiet_mode:
            print(f"Remote audio stream started successfully")
            print(f"Frames per buffer: {frames_per_buffer} (matching PyAudio)")
            print(f"Chunk size: {chunk_size_ms}ms")
            print("Listening for audio... (Press Ctrl+C to stop)")
            print('=' * 50)
        
        # Store last transcription to avoid repetition
        last_raw = ""
        last_punct = ""
        
        # Processing statistics
        chunks_processed = 0
        consecutive_errors = 0
        max_consecutive_errors = 3
        transcription_count = 0
        
        # Main processing loop - modeled after PyAudio callback
        while True:
            try:
                # Check if processing should continue (includes command checking)
                if not asr_system.should_process_audio():
                    # If paused/stopped, skip audio processing but continue checking commands
                    time.sleep(0.1)
                    continue
                
                # Read exactly frames_per_buffer samples - like PyAudio callback
                audio_chunk = remote_stream.read_audio_pyaudio_compatible(
                    chunk_size=frames_per_buffer,
                    timeout=1.0
                )
                
                if audio_chunk is not None:
                    # Verify chunk size consistency
                    if len(audio_chunk) != frames_per_buffer:
                        logger.error(f"Chunk size inconsistency - expected {frames_per_buffer}, got {len(audio_chunk)}")
                        continue
                    
                    chunks_processed += 1
                    
                    # Debug chunk format periodically (verbose level)
                    if chunks_processed % 50 == 0:
                        rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
                        logger.verbose(f"CHUNK #{chunks_processed}: {len(audio_chunk)} samples, "
                              f"dtype={audio_chunk.dtype}, range=[{np.min(audio_chunk)}, {np.max(audio_chunk)}], "
                              f"RMS={rms:.0f}")
                    
                    # Process audio chunk with ASR - exactly like PyAudio callback
                    raw_text, punct_text, is_eou, complete_utterance = asr_system.transcribe_chunk(audio_chunk)
                    
                    # Debug ASR output (debug level)
                    if raw_text.strip():
                        transcription_count += 1
                        logger.debug(f"ASR OUTPUT #{transcription_count}: '{raw_text}' -> '{punct_text}'")
                    elif chunks_processed % 100 == 0:
                        logger.verbose(f"ASR: No transcription for chunk #{chunks_processed} (total transcriptions: {transcription_count})")
                    
                    # Reset error counter on successful processing
                    consecutive_errors = 0
                    
                    # Handle EOU - output complete utterance with clear separators
                    if is_eou and complete_utterance:
                        # Always log complete utterances at CRITICAL level to ensure they're always shown
                        logger.critical(f"COMPLETE UTTERANCE: {complete_utterance}")
                        
                        if quiet_mode:
                            print(f"\n{'='*70}")
                            print(f"COMPLETE UTTERANCE:")
                            print(f"{'='*70}")
                            print(f"{complete_utterance}")
                            print(f"{'='*70}\n")
                        else:
                            print(f"\n{'='*70}")
                            print(f"[COMPLETE UTTERANCE - ASR FULLY RESET]")
                            print(f"{complete_utterance}")
                            print(f"{'='*70}")
                        
                        # Reset tracking variables for fresh start
                        last_raw = ""
                        last_punct = ""
                        continue
                    
                    # Only print if text has changed and it's not an EOU
                    if raw_text.strip() and raw_text != last_raw and not is_eou:
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
                else:
                    # No audio data received - check stream status
                    if not quiet_mode and chunks_processed % 100 == 0:
                        status = remote_stream.get_status()
                        pause_indicator = " [PAUSED]" if asr_system.is_paused else ""
                        print(f"\rWaiting for audio data... (connected: {status['connected']}, queue: {status['queue_size']}, chunks: {chunks_processed}){pause_indicator}", end='', flush=True)
                    
                    # Small delay to prevent busy waiting
                    time.sleep(0.01)
                        
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error processing audio chunk #{chunks_processed}: {e}")
                if not quiet_mode:
                    if consecutive_errors <= 2:  # Only show traceback for first few errors
                        logger.error(f"Full traceback:", exc_info=True)
                
                # If we have too many consecutive errors, do a complete system reset
                if consecutive_errors >= max_consecutive_errors:
                    print(f"ERROR: {consecutive_errors} consecutive errors - performing complete system reset")
                    
                    # Clear all state
                    last_raw = ""
                    last_punct = ""
                    
                    # Complete ASR reset
                    try:
                        asr_system._reset_streaming_state(reset_conversation=True)
                        print("ASR system reset completed")
                        consecutive_errors = 0  # Reset error counter after system reset
                    except Exception as reset_error:
                        print(f"ERROR: Failed to reset ASR system: {reset_error}")
                        break  # Exit on critical failure
                else:
                    # Small delay before continuing to prevent rapid error cycling
                    time.sleep(0.1)
            
    except KeyboardInterrupt:
        if not quiet_mode:
            print('\n\nStopping remote audio stream...')
    finally:
        remote_stream.stop()
        if not quiet_mode:
            print("Remote audio stream stopped")
            if chunks_processed > 0:
                print(f"Total chunks processed: {chunks_processed}")
                print(f"Total transcriptions: {transcription_count}")

def run_streaming_asr_with_microphone(asr_system, device_id, chunk_size_ms, show_raw, quiet_mode):
    """Run streaming ASR using local microphone (original implementation)"""
    
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
            logger = logging.getLogger("haku_stt.microphone")
            if not quiet_mode:
                logger.error('No audio input device found.')
            return
        
        # Convert device_id to int if it's a valid number (not None or 'remote')
        if device_id is not None and device_id != 'remote':
            try:
                device_id = int(device_id)
            except ValueError:
                logger = logging.getLogger("haku_stt.microphone")
                if not quiet_mode:
                    logger.error(f"Invalid device ID '{device_id}'. Must be a number or 'remote'.")
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
        
        if device_id not in input_devices and device_id != 'remote':
            if not quiet_mode:
                print(f"Error: Device {device_id} not found in available input devices")
            return
            
        if not quiet_mode:
            if device_id == 'remote':
                print("Using remote audio stream")
            else:
                print(f"Using device {device_id}: {p.get_device_info_by_index(device_id)['name']}")
        
        # Calculate frames per buffer
        frames_per_buffer = int(SAMPLE_RATE * chunk_size_ms / 1000) - 1
        
        # Store last transcription to avoid repetition
        last_raw = ""
        last_punct = ""
        
        # Define callback function
        def stream_callback(in_data, frame_count, time_info, status):
            nonlocal last_raw, last_punct
            
            # Check if processing should continue (includes command checking)
            if not asr_system.should_process_audio():
                return (in_data, pa.paContinue)
            
            if status and not quiet_mode:
                print(f"Stream status: {status}")
                
            # Convert audio data and transcribe
            signal = np.frombuffer(in_data, dtype=np.int16)
            raw_text, punct_text, is_eou, complete_utterance = asr_system.transcribe_chunk(signal)
            
            # Handle EOU - output complete utterance with clear separators
            if is_eou and complete_utterance:
                # Always log complete utterances at CRITICAL level to ensure they're always shown
                logger = logging.getLogger("haku_stt.microphone")
                logger.critical(f"COMPLETE UTTERANCE: {complete_utterance}")
                
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
            print('Send WebSocket commands to control processing:')
            print('  {"type": "control", "action": "pause"}   - Pause processing and purge utterance')
            print('  {"type": "control", "action": "resume"}  - Resume processing')
            print('  {"type": "control", "action": "purge"}   - Purge current utterance and reset')
            print('  {"type": "control", "action": "reset"}   - Complete system reset')
            if asr_system.punct_model:
                print('Punctuation and capitalization enabled')
            else:
                print('No punctuation model loaded')
            if asr_system.unified_eou_detector:
                print('Unified end-of-utterance detection enabled')
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

def load_config(config_path):
    """Load and validate configuration from JSON file with VAD-specific defaults"""
    if not os.path.exists(config_path):
        # Can't use logger here as it's not set up yet
        import sys
        print(f"Warning: Config file '{config_path}' not found. Using hardcoded defaults.", file=sys.stderr)
        return get_hardcoded_defaults()
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate structure
        if "mode-selection" not in config or "modes" not in config:
            raise ValueError("Config must contain 'mode-selection' and 'modes'")
        
        selected_mode = config["mode-selection"]
        if selected_mode not in config["modes"]:
            raise ValueError(f"Selected mode '{selected_mode}' not found in modes")
        
        # Start with default settings
        default_settings = config["modes"]["default"].copy()
        mode_overrides = config["modes"][selected_mode].copy()
        
        # Merge mode overrides into default settings
        merged_settings = {**default_settings, **mode_overrides}
        
        # Apply VAD-specific defaults based on selected VAD model
        vad_model = merged_settings.get("vad-model", "nvidia")
        if "vad-defaults" in config and vad_model in config["vad-defaults"]:
            vad_defaults = config["vad-defaults"][vad_model]
            
            # Apply VAD defaults first, then override with any explicit settings
            for key, default_value in vad_defaults.items():
                if key not in merged_settings:
                    merged_settings[key] = default_value
        
        return merged_settings
    except Exception as e:
        print(f"Error loading config: {e}. Using hardcoded defaults.")
        return get_hardcoded_defaults()

def setup_global_logging(log_level="debug", log_format=None, log_file=None, 
                        log_max_size=10, log_backup_count=3):
    """Setup global logging configuration"""
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    # Suppress third-party library logging
    logging.getLogger('nemo_logger').setLevel(logging.CRITICAL)
    logging.getLogger('pytorch_lightning').setLevel(logging.CRITICAL)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    # Set global log level
    level_map = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING, 
        "info": logging.INFO,
        "verbose": VERBOSE,
        "debug": logging.DEBUG
    }
    if isinstance(log_level, int):
        resolved_level = log_level
    elif isinstance(log_level, str):
        try:
            resolved_level = int(log_level)
        except Exception:
            resolved_level = level_map.get(log_level.lower(), logging.DEBUG)
    else:
        resolved_level = logging.DEBUG
    
    # Debug: Print the input and resolved level
    print(f"[LOGGING] Input log level: {log_level} (type: {type(log_level)})")
    print(f"[LOGGING] Resolved level: {resolved_level} ({logging.getLevelName(resolved_level)})")
    
    root_logger.setLevel(resolved_level)
    
    # Update all existing haku_stt loggers to match root level
    for name in logging.Logger.manager.loggerDict:
        if name.startswith('haku_stt.'):
            logger = logging.getLogger(name)
            logger.setLevel(resolved_level)

def get_hardcoded_defaults():
    """Fallback hardcoded defaults with logging settings"""
    return {
        "asr-model": "stt_en_fastconformer_hybrid_large_streaming_multi",
        "punct-model": "punctuation_en_bert",
        "lookahead": 480,
        "decoder": "rnnt",
        "device": None,
        "remote-port": 5004,
        "chunk-size": None,
        "show-raw": False,
        "no-eou": False,
        "eou-threshold": 0.8,
        "eou-min-words": 4,
        "eou-confirmation-needed": 2,
        "eou-vad-weight": 0.6,
        "eou-text-weight": 0.3,
        "eou-silence-weight": 0.1,
        "vad-model": "nvidia",
        "vad-silence-threshold": 15,
        "vad-speech-threshold": 0.25,
        "vad-activity-threshold": 0.3,
        "enable-text-eou": False,
        "vad-speech-proportion-threshold": 0.15,
        "vad-analysis-window": 2.0,
        "vad-consecutive-silence-threshold": 0.8,
        "vad-min-frames-for-eou": 0.5,
        "vad-recent-speech-lookback-seconds": 1.5,
        "vad-recent-speech-threshold": 0.10,
        "websocket-host": None,
        "websocket-port": 8765,
        "websocket-ping-interval": 20,
        "websocket-ping-timeout": 10,
        "log-level": "debug",
        "log-format": "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        "log-file": None,
        "log-max-size": 10,
        "log-backup-count": 3
    }

def build_argparse_from_config(config_path):
    """
    Dynamically build argparse parser from config descriptions.
    This allows --help to auto-populate with descriptions from config.json.
    """
    # Load full config to get descriptions
    try:
        with open(config_path, 'r') as f:
            full_config = json.load(f)
        descriptions = full_config.get("descriptions", {})
    except Exception:
        descriptions = {}  # Fallback if config can't be loaded
    
    # Load merged config for defaults
    config = load_config(config_path)
    
    parser = argparse.ArgumentParser(
        description="Online ASR with config file support. CLI args override config values.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ws_stt.py --websocket-host 10.0.0.2 --device remote
  python ws_stt.py --config my_config.json --quiet
  python ws_stt.py --mode-selection fast
        """
    )
    
    # Add --config arg first
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to JSON config file"
    )
    
    # Add --mode-selection arg
    parser.add_argument(
        "--mode-selection",
        type=str,
        default=None,
        help="Mode selection from config.json (e.g., 'default', 'fast', 'slow', 'reliable')"
    )
    
    # Dynamically add args for each config key using descriptions
    for key, default_value in config.items():
        if key in ["mode-selection", "modes", "descriptions"]:
            continue  # Skip non-setting keys
        
        # Determine type based on default value
        arg_type = type(default_value) if default_value is not None else str
        if arg_type == bool:
            # For booleans, use store_true with SUPPRESS to only set if provided
            parser.add_argument(
                f"--{key.replace('_', '-')}",
                action="store_true",
                default=argparse.SUPPRESS,  # Attribute not set if not provided
                help=descriptions.get(key, f"Override {key} from config")
            )
        else:
            parser.add_argument(
                f"--{key}",  # Use the key as-is for CLI argument
                type=arg_type,
                default=None,  # None means not provided, so don't override
                help=descriptions.get(key, f"Override {key} from config")
            )
    
    return parser

def merge_config_with_args(config, args):
    """
    Merge config with CLI args: CLI overrides take precedence if provided.
    Only overrides non-None CLI values.
    """
    merged = config.copy()
    for key in config:
        if key in ["mode-selection", "modes", "descriptions"]:
            continue  # Skip non-setting keys
        
        # Convert config key (with hyphens) to argparse attribute name (with underscores)
        cli_key = key.replace('-', '_')
        cli_value = getattr(args, cli_key, None)
        
        if cli_value is not None:
            merged[key] = cli_value
            # Can't use logger here as it's not set up yet - use stderr for config debug
            import sys
            print(f"[CONFIG] Overriding {key} with CLI value: {cli_value}", file=sys.stderr)
    
    return merged

def main():
    # Parse arguments first to get config path
    parser = build_argparse_from_config("config.json")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override mode-selection if provided via CLI
    if args.mode_selection is not None:
        config["mode-selection"] = args.mode_selection
        # Can't use logger here as it's not set up yet
        import sys
        print(f"[CONFIG] Overriding mode-selection with CLI value: {args.mode_selection}", file=sys.stderr)

    # Merge config with CLI overrides
    config = merge_config_with_args(config, args)

    # Setup global logging first
    setup_global_logging(
        log_level=config.get("log-level", "debug"),
        log_format=config.get("log-format"),
        log_file=config.get("log-file"),
        log_max_size=config.get("log-max-size", 10),
        log_backup_count=config.get("log-backup-count", 3)
    )
    
    # Create main logger
    main_logger = logging.getLogger("haku_stt.main")
    main_logger.info("Starting Haku STT system...")
    
    # Create log config for components
    log_config = {
        "log_level": config.get("log-level", "debug"),
        "log_format": config.get("log-format"),
        "log_file": config.get("log-file"),
        "log_max_size": config.get("log-max-size", 10),
        "log_backup_count": config.get("log-backup-count", 3)
    }
    
    # Extract settings from merged config
    asr_model = config["asr-model"]
    punct_model = config["punct-model"]
    lookahead = config["lookahead"]
    decoder = config["decoder"]
    device_id = config["device"]
    remote_port = config["remote-port"]
    chunk_size_ms = config["chunk-size"]
    show_raw = config["show-raw"]
    
    # Automatically determine quiet_mode based on log level
    log_level_str = config.get("log-level", "debug").lower()
    # quiet_mode = log_level_str in ["info", "warning", "error", "critical"]
    quiet_mode = log_level_str in ["warning", "error", "critical"]
    if quiet_mode:
        main_logger.debug(f"Enabling quiet_mode due to log level: {log_level_str}")
    
    enable_eou = not config["no-eou"]
    eou_threshold = config["eou-threshold"]
    eou_min_words = config.get("eou-min-words", 4)
    eou_confirmation_needed = config.get("eou-confirmation-needed", 2)
    vad_model_type = config["vad-model"]
    vad_silence_threshold = config["vad-silence-threshold"]
    vad_speech_threshold = config.get("vad-speech-threshold", 0.5)
    vad_activity_threshold = config["vad-activity-threshold"]
    enable_text_eou = config["enable-text-eou"]
    vad_speech_proportion_threshold = config.get("vad-speech-proportion-threshold", 0.2)
    vad_analysis_window = config.get("vad-analysis-window", 2.0)
    vad_consecutive_silence_threshold = config.get("vad-consecutive-silence-threshold", 0.8)
    vad_min_frames_for_eou = config.get("vad-min-frames-for-eou", 0.5)
    vad_lookback_seconds = config.get("vad-recent-speech-lookback-seconds", 1.5)
    vad_recent_speech_threshold = config.get("vad-recent-speech-threshold", 0.10)
    websocket_host = config["websocket-host"]
    websocket_port = config["websocket-port"]
    websocket_ping_interval = config.get("websocket-ping-interval", 20)
    websocket_ping_timeout = config.get("websocket-ping-timeout", 10)
    
    # Initialize WebSocket server if requested
    websocket_server = None
    if websocket_host and WEBSOCKET_AVAILABLE:
        websocket_server = WebSocketServer(
            host=websocket_host,
            port=websocket_port,
            log_config=log_config
        )
        
        if websocket_server.start_server():
            main_logger.info(f"WebSocket server enabled on ws://{websocket_host}:{websocket_port}")
        else:
            main_logger.error("Failed to start WebSocket server")
            websocket_server = None
    elif websocket_host and not WEBSOCKET_AVAILABLE:
        main_logger.error("WebSocket functionality not available. Install with: pip install websockets")
        return
    
    try:
        # Initialize ASR system
        main_logger.info("Initializing Online ASR with VAD-based EOU Detection system...")
            
        asr_system = OnlineASRWithPunctuation(
            asr_model_name=asr_model,
            punct_model_name=punct_model,
            lookahead_size=lookahead,
            decoder_type=decoder,
            quiet_mode=quiet_mode,
            enable_eou=enable_eou,
            websocket_server=websocket_server,
            log_config=log_config
        )
        
        # Reinitialize frame detector with the correct VAD model type and configuration
        if enable_eou:
            vad_config = {
                "vad-speech-threshold": vad_speech_threshold,
                "vad-speech-proportion-threshold": vad_speech_proportion_threshold,
                "vad-analysis-window": vad_analysis_window,
                "vad-consecutive-silence-threshold": vad_consecutive_silence_threshold,
                "vad-min-frames-for-eou": vad_min_frames_for_eou,
                "vad-recent-speech-lookback-seconds": vad_lookback_seconds,
                "vad-recent-speech-threshold": vad_recent_speech_threshold
            }
            asr_system.frame_detector = FrameLevelSpeechDetector(vad_model_type, vad_config, log_config)
        
        # Configure VAD-based EOU detector
        if asr_system.frame_detector and asr_system.frame_detector.vad_model:
            asr_system.frame_detector.speech_probability_threshold = vad_speech_threshold
            asr_system.frame_detector.speech_proportion_threshold = vad_speech_proportion_threshold
            asr_system.frame_detector.analysis_window_seconds = vad_analysis_window
            asr_system.frame_detector.consecutive_silence_threshold = int(
                vad_consecutive_silence_threshold / asr_system.frame_detector.vad_frame_duration
            )
            # Recalculate frames per analysis window
            asr_system.frame_detector.frames_per_analysis_window = int(
                vad_analysis_window / asr_system.frame_detector.vad_frame_duration
            )
            asr_system.frame_detector.speech_frame_history = deque(
                maxlen=asr_system.frame_detector.frames_per_analysis_window
            )
            
            main_logger.info(f"VAD-based EOU configured:")
            main_logger.info(f"  VAD Model: {vad_model_type.upper()}")
            main_logger.debug(f"  Frame duration: {asr_system.frame_detector.vad_frame_duration*1000:.0f}ms")
            main_logger.debug(f"  Analysis window: {vad_analysis_window}s")
            main_logger.debug(f"  Speech probability threshold: {vad_speech_threshold}")
            main_logger.debug(f"  Speech proportion threshold: {vad_speech_proportion_threshold}")
            main_logger.debug(f"  Consecutive silence threshold: {vad_consecutive_silence_threshold}s")
            main_logger.debug(f"  Frames per window: {asr_system.frame_detector.frames_per_analysis_window}")
        
        # Initialize unified EOU detector if EOU is enabled
        if enable_eou:
            # Create unified EOU configuration
            unified_eou_config = {
                "eou-threshold": eou_threshold,
                "eou-weights": {
                    "vad": config.get("eou-vad-weight", 0.6),
                    "text": config.get("eou-text-weight", 0.3) if enable_text_eou else 0.0,
                    "silence": config.get("eou-silence-weight", 0.1)
                },
                "eou-min-words": eou_min_words,
                "eou-confirmation-needed": eou_confirmation_needed,
                "enable-text-eou": enable_text_eou,
                "enable-vad-eou": True,
                "enable-silence-eou": True,
                "text-eou-threshold": 0.8  # Threshold for individual text detector
            }
            
            # Initialize unified EOU detector
            asr_system.unified_eou_detector = UnifiedEOUDetector(
                log_config=log_config,
                config=unified_eou_config
            )
            
            # Connect VAD detector to unified EOU detector
            if asr_system.frame_detector:
                asr_system.unified_eou_detector.set_vad_detector(asr_system.frame_detector)
            
            main_logger.info(f"Unified EOU detector initialized")
            main_logger.info(f"  Text EOU: {'Enabled' if enable_text_eou else 'Disabled'}")
            main_logger.info(f"  VAD EOU: Enabled")
            main_logger.info(f"  EOU Weights: {unified_eou_config['eou-weights']}")
            main_logger.debug(f"  Overall threshold: {eou_threshold}")
        else:
            main_logger.info("EOU detection disabled")
    
        main_logger.info("ASR system ready!")
        main_logger.info(f"ASR Model: {asr_model}")
        main_logger.info(f"Punctuation Model: {punct_model or 'None'}")
        main_logger.info(f"Text EOU Detection: {'Enabled' if enable_text_eou else 'Disabled'}")
        main_logger.info(f"Lookahead: {lookahead}ms")
        main_logger.info(f"Decoder: {decoder}")
        if device_id == 'remote':
            main_logger.info(f"Audio Input: Remote stream (port {remote_port})")
        else:
            main_logger.info(f"Audio Input: {'Auto-select microphone' if device_id is None else f'Device {device_id}'}")
        if websocket_server:
            main_logger.info(f"WebSocket Output: ws://{websocket_host}:{websocket_port}")
        main_logger.info(f"Log level: {config.get('log-level', 'debug').upper()}")
        
        # Send initial state via WebSocket
        if websocket_server:
            # Send "started" status ONCE during initialization
            websocket_server.send_status_update("started", {
                "asr_model": asr_model,
                "punct_model": punct_model,
                "eou_enabled": enable_eou,
                "device_type": "remote" if device_id == 'remote' else "microphone",
                "initialized": True
            })
            
            # Schedule transition to "running" state after initialization
            async def transition_to_running():
                await asyncio.sleep(0.5)  # Give time for started message to be received
                # Update WebSocket server state to running
                websocket_server.current_state = 'running'
                await websocket_server.send_state_update('running', {
                    'initialization_complete': True,
                    'timestamp': time.time()
                })
            
            if websocket_server.loop:
                try:
                    asyncio.run_coroutine_threadsafe(
                        transition_to_running(),
                        websocket_server.loop
                    )
                except Exception as e:
                    main_logger.error(f"Error transitioning to running state: {e}")
    
        # Run streaming ASR with config values
        run_streaming_asr_with_punct(
            asr_system=asr_system,
            device_id=device_id,
            chunk_size_ms=chunk_size_ms,
            show_raw=show_raw,
            quiet_mode=quiet_mode,
            remote_audio_port=remote_port
        )
        
    except KeyboardInterrupt:
        main_logger.info("Shutdown requested by user")
    except Exception as e:
        main_logger.error(f"System error: {e}", exc_info=True)
    finally:
        # Clean up WebSocket server
        if websocket_server:
            websocket_server.send_status_update("stopped", {"reason": "shutdown"})
            time.sleep(0.5)  # Give time for final message to send
            websocket_server.stop_server()
            main_logger.info("Shutting down WebSocket server...")

if __name__ == "__main__":
    main()


