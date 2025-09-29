"""
Online ASR with Punctuation and Capitalization
Combines streaming FastConformer ASR with punctuation/capitalization post-processing

This script does real-time speech recognition and applies punctuation and capitalization
to make the output more readable.
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

class OnlineASRWithPunctuation:
    def __init__(self, asr_model_name, punct_model_name=None, lookahead_size=480, decoder_type="rnnt", quiet_mode=False):
        """
        Initialize the Online ASR system with punctuation
        
        Args:
            asr_model_name: Name of the pretrained ASR model to use
            punct_model_name: Name of the punctuation model (None to disable)
            lookahead_size: Lookahead size in milliseconds
            decoder_type: "rnnt" or "ctc"
            quiet_mode: If True, suppress all logging
        """
        self.asr_model_name = asr_model_name
        self.punct_model_name = punct_model_name
        self.lookahead_size = lookahead_size
        self.decoder_type = decoder_type
        self.quiet_mode = quiet_mode
        
        # Comprehensive logging suppression for quiet mode
        if quiet_mode:
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
            
            # Redirect stderr temporarily during model loading
            self._original_stderr = sys.stderr
            
        # Initialize models
        self._setup_asr_model()
        if punct_model_name:
            self._setup_punctuation_model()
        else:
            self.punct_model = None
            
        # Text buffer for punctuation processing
        self.text_buffer = deque(maxlen=50)
        self._setup_preprocessing()
        self._reset_streaming_state()
        
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
        
    def _reset_streaming_state(self):
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
    
    def transcribe_chunk(self, audio_chunk):
        """
        Transcribe a single audio chunk with punctuation
        
        Args:
            audio_chunk: numpy array of audio samples (int16)
            
        Returns:
            tuple: (raw_text, punctuated_text)
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
        
        # Apply punctuation if enabled
        if self.punct_model and raw_text.strip():
            punctuated_text = self._apply_punctuation(raw_text)
        else:
            punctuated_text = raw_text
        
        self.step_num += 1
        
        return raw_text, punctuated_text

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
    Run streaming ASR with punctuation using microphone input
    
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
            raw_text, punct_text = asr_system.transcribe_chunk(signal)
            
            # Only print if text has changed
            if raw_text.strip() and raw_text != last_raw:
                if quiet_mode:
                    # In quiet mode, only print the final punctuated text
                    if punct_text.strip():
                        print(punct_text, flush=True)
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
        description="Online ASR with Punctuation and Capitalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available ASR models:
{chr(10).join(f"  - {model}" for model in AVAILABLE_ASR_MODELS)}

Available punctuation models:
{chr(10).join(f"  - {model}" for model in AVAILABLE_PUNCT_MODELS)}

Examples:
  # Basic usage with punctuation
  python {__file__} --asr-model stt_en_fastconformer_hybrid_large_streaming_multi --punct-model punctuation_en_bert

  # Without punctuation (raw ASR only)
  python {__file__} --asr-model stt_en_fastconformer_hybrid_large_streaming_80ms

  # Show both raw and punctuated text
  python {__file__} --asr-model stt_en_fastconformer_hybrid_large_streaming_multi --punct-model punctuation_en_bert --show-raw
  
  # Quiet mode - only show punctuated output
  python {__file__} --asr-model stt_en_fastconformer_hybrid_large_streaming_multi --punct-model punctuation_en_bert --quiet
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
        default=80,
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
    
    args = parser.parse_args()
    
    # List devices and exit if requested
    if args.list_devices:
        list_audio_devices()
        return
    
    # Set up comprehensive quiet mode before any NeMo imports
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
            print("Initializing Online ASR with Punctuation system...")
            
        asr_system = OnlineASRWithPunctuation(
            asr_model_name=args.asr_model,
            punct_model_name=args.punct_model,
            lookahead_size=args.lookahead,
            decoder_type=args.decoder,
            quiet_mode=args.quiet
        )
        
        if not args.quiet:
            print("\nASR system ready!")
            print(f"ASR Model: {args.asr_model}")
            print(f"Punctuation Model: {args.punct_model or 'None'}")
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
