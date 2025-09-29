#!/usr/bin/env python3
"""
Frame-by-frame speech detection test script.
This script uses the VAD model to analyze speech activity per frame and shows
the proportion of speech frames in the last 1 second of audio from the microphone.

This script reads the configuration from config.json to use the same VAD model
and settings as the main ws_stt script.
"""

import sys
import os
import time
import numpy as np
import pyaudio as pa
import torch
import json
from collections import deque

# Add the current directory to Python path to import from ws_stt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ws_stt import FrameLevelSpeechDetector, load_config

# Audio constants
SAMPLE_RATE = 16000  # Hz
CHUNK_DURATION_MS = 80  # 80ms chunks to match ASR system
FRAMES_PER_BUFFER = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

class FrameSpeechTester:
    def __init__(self, config_path="config.json"):
        """Initialize the frame-by-frame speech tester with config from ws_stt"""
        print("=== Frame-by-Frame Speech Detection Tester ===")
        print("This script analyzes speech activity per frame using VAD.")
        print("It shows speech proportion in the last 1 second of audio.")
        print("Using configuration from ws_stt system.\n")
        
        # Load configuration from config.json (same as ws_stt)
        print(f"Loading configuration from: {config_path}")
        try:
            self.config = load_config(config_path)
            print(f"Loaded configuration for mode: {self.config.get('mode-selection', 'default')}")
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            print("Using default VAD settings...")
            self.config = {"vad-model": "nvidia"}
        
        # Extract VAD configuration
        self.vad_model_type = self.config.get("vad-model", "nvidia")
        
        # Create VAD configuration dictionary
        vad_config = {
            "vad-speech-threshold": self.config.get("vad-speech-threshold", 0.5 if self.vad_model_type == "silero" else 0.25),
            "vad-speech-proportion-threshold": self.config.get("vad-speech-proportion-threshold", 0.2 if self.vad_model_type == "silero" else 0.15),
            "vad-analysis-window": self.config.get("vad-analysis-window", 2.5 if self.vad_model_type == "silero" else 2.0),
            "vad-consecutive-silence-threshold": self.config.get("vad-consecutive-silence-threshold", 1.0 if self.vad_model_type == "silero" else 0.8),
            "vad-min-frames-for-eou": self.config.get("vad-min-frames-for-eou", 0.64 if self.vad_model_type == "silero" else 0.5),
            "vad-recent-speech-lookback-seconds": self.config.get("vad-recent-speech-lookback-seconds", 2.0 if self.vad_model_type == "silero" else 1.5),
            "vad-recent-speech-threshold": self.config.get("vad-recent-speech-threshold", 0.15 if self.vad_model_type == "silero" else 0.10)
        }
        
        print("Configuration loaded:")
        print(f"  VAD Model: {self.vad_model_type.upper()}")
        print(f"  Speech threshold: {vad_config['vad-speech-threshold']}")
        print(f"  Speech proportion threshold: {vad_config['vad-speech-proportion-threshold']}")
        print(f"  Analysis window: {vad_config['vad-analysis-window']}s")
        print(f"  Consecutive silence threshold: {vad_config['vad-consecutive-silence-threshold']}s")
        print()
        
        print("Loading frame-level speech detector...")
        self.frame_detector = FrameLevelSpeechDetector(
            vad_model_type=self.vad_model_type,
            config=vad_config
        )
        
        if not self.frame_detector.vad_model:
            print("ERROR: Failed to load VAD model. Cannot proceed.")
            return
        
        print("Frame detector loaded successfully!\n")
        
        # Analysis parameters
        self.analysis_window_seconds = 1.0  # Analyze last 1 second
        self.frames_per_analysis_window = int(self.analysis_window_seconds / self.frame_detector.vad_frame_duration)
        
        # Override the frame detector's history to match our 1-second window
        self.frame_detector.speech_frame_history = deque(maxlen=self.frames_per_analysis_window)
        
        print("Configuration:")
        print(f"  VAD frame duration: {self.frame_detector.vad_frame_duration*1000:.0f}ms ({self.vad_model_type} VAD)")
        print(f"  Analysis window: {self.analysis_window_seconds}s")
        print(f"  Frames per analysis window: {self.frames_per_analysis_window}")
        print(f"  Speech probability threshold: {self.frame_detector.speech_probability_threshold}")
        print(f"  Chunk duration: {CHUNK_DURATION_MS}ms")
        print(f"  Samples per chunk: {FRAMES_PER_BUFFER}")
        print()
        
        # Statistics tracking
        self.total_chunks_processed = 0
        self.total_frames_analyzed = 0
        self.chunks_with_speech = 0
        self.start_time = time.time()
        
    def list_audio_devices(self):
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
    
    def create_probability_bar(self, probability, width=20):
        """
        Create a visual bar showing speech probability
        
        Args:
            probability: Speech probability (0.0 to 1.0)
            width: Width of the bar in characters
            
        Returns:
            str: Formatted bar with colors
        """
        # ANSI color codes
        GREEN = '\033[92m'
        RED = '\033[91m'
        RESET = '\033[0m'
        
        # Calculate how many characters to fill
        filled_chars = int(probability * width)
        empty_chars = width - filled_chars
        
        # Choose color based on threshold
        is_over_threshold = probability > self.frame_detector.speech_probability_threshold
        color = GREEN if is_over_threshold else RED
        
        # Create the bar
        filled_part = '█' * filled_chars
        empty_part = '░' * empty_chars
        
        return f"{color}[{filled_part}{empty_part}]{RESET} {probability:.3f}"
    
    def analyze_audio_chunk(self, audio_chunk):
        """
        Analyze a single audio chunk for speech activity with visual probability bars
        
        Args:
            audio_chunk: numpy array of audio samples (int16)
        """
        self.total_chunks_processed += 1
        
        # Convert int16 to float32 and normalize
        audio_data = audio_chunk.astype(np.float32) / 32768.0
        
        # Get raw speech probabilities for visualization
        speech_probabilities = self.get_speech_probabilities(audio_data)
        
        # Analyze with VAD to get frame-level speech detection
        speech_frames = self.frame_detector.analyze_audio_vad(audio_data)
        
        if not speech_frames:
            return
        
        self.total_frames_analyzed += len(speech_frames)
        
        # Update the rolling window with new frame results
        self.frame_detector.update_speech_frame_history(speech_frames)
        
        # Check if any frames in this chunk had speech
        speech_detected_in_chunk = any(speech_frames)
        if speech_detected_in_chunk:
            self.chunks_with_speech += 1
        
        # Get speech proportion in the last 1 second
        analysis = self.frame_detector.get_recent_speech_proportion(
            window_seconds=self.analysis_window_seconds
        )
        
        # Show probability bars for each frame
        current_time = time.time() - self.start_time
        print(f"\n[{current_time:6.1f}s] Chunk #{self.total_chunks_processed} - "
              f"1s proportion: {analysis['speech_proportion']:.3f} "
              f"({analysis['speech_frames']}/{analysis['total_frames']} frames)")
        
        # Display probability bar for each frame in this chunk
        for i, (prob, is_speech) in enumerate(zip(speech_probabilities, speech_frames)):
            frame_status = "SPEECH" if is_speech else "silence"
            bar = self.create_probability_bar(prob)
            print(f"  Frame {i+1}: {bar} ({frame_status})")
        
        # Periodically show statistics (every 20 chunks = ~1.6 seconds)
        if self.total_chunks_processed % 20 == 0:
            chunk_speech_rate = (self.chunks_with_speech / self.total_chunks_processed) * 100
            
            print(f"\n{'='*60}")
            print(f"STATS at {current_time:.1f}s:")
            print(f"  Total chunks: {self.total_chunks_processed}")
            print(f"  Chunks with speech: {self.chunks_with_speech} ({chunk_speech_rate:.1f}%)")
            print(f"  Total frames analyzed: {self.total_frames_analyzed}")
            if analysis['sufficient_data']:
                print(f"  Current 1s speech proportion: {analysis['speech_proportion']:.3f}")
            else:
                print(f"  Insufficient data for 1s analysis (need {self.frame_detector.min_frames_for_eou} frames)")
            print(f"{'='*60}\n")
    
    def get_speech_probabilities(self, audio_signal):
        """
        Get raw speech probabilities for each frame (similar to analyze_audio_vad but returns probabilities)
        
        Args:
            audio_signal: Raw audio signal (numpy array, float32, 16kHz)
            
        Returns:
            list: List of speech probabilities for each frame
        """
        try:
            if self.frame_detector.vad_model is None or len(audio_signal) == 0:
                return []
            
            # Use different approaches based on VAD model type
            if self.vad_model_type == "silero":
                return self._get_silero_probabilities(audio_signal)
            else:
                return self._get_nvidia_probabilities(audio_signal)
            
        except Exception as e:
            print(f"Error getting speech probabilities: {e}")
            return []
    
    def _get_nvidia_probabilities(self, audio_signal):
        """Get speech probabilities from NeMo VAD model"""
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
        device = next(self.frame_detector.vad_model.parameters()).device
        input_signal = input_signal.to(device)
        input_signal_length = input_signal_length.to(device)
        
        # Run VAD inference
        with torch.no_grad():
            vad_outputs = self.frame_detector.vad_model(
                input_signal=input_signal,
                input_signal_length=input_signal_length
            )
        
        # Move outputs back to CPU for processing
        if hasattr(vad_outputs, 'cpu'):
            vad_probs = vad_outputs.cpu().numpy()
        else:
            vad_probs = vad_outputs
        
        # Handle VAD output format
        if vad_probs.ndim == 3:
            vad_probs = vad_probs.squeeze(0)  # Remove batch dimension
        
        # Convert probabilities to speech probabilities
        if vad_probs.ndim == 2 and vad_probs.shape[1] == 2:
            # Extract speech probabilities (second column)
            speech_probs = vad_probs[:, 1]  # Get speech probabilities
            
            # Apply softmax to convert logits to probabilities if needed
            if np.any(speech_probs < 0) or np.any(speech_probs > 1):
                # These look like logits, apply softmax
                exp_probs = np.exp(vad_probs - np.max(vad_probs, axis=1, keepdims=True))
                softmax_probs = exp_probs / np.sum(exp_probs, axis=1, keepdims=True)
                speech_probs = softmax_probs[:, 1]  # Speech probabilities after softmax
            
            return speech_probs.tolist()
        elif vad_probs.ndim == 1:
            # Single dimension - treat as speech probabilities directly
            if np.any(vad_probs < 0) or np.any(vad_probs > 1):
                speech_probs = 1.0 / (1.0 + np.exp(-vad_probs))  # Sigmoid
            else:
                speech_probs = vad_probs
            return speech_probs.tolist()
        else:
            return []
    
    def _get_silero_probabilities(self, audio_signal):
        """Get speech probabilities from Silero VAD model"""
        # Ensure audio is the right format
        if not isinstance(audio_signal, np.ndarray):
            audio_signal = np.array(audio_signal)
        
        if audio_signal.dtype != np.float32:
            audio_signal = audio_signal.astype(np.float32)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_signal)
        
        # Move to same device as model
        device = next(self.frame_detector.vad_model.parameters()).device
        audio_tensor = audio_tensor.to(device)
        
        # Silero VAD processes audio in 32ms chunks (512 samples at 16kHz)
        speech_probs = []
        
        with torch.no_grad():
            # Process audio in chunks
            for i in range(0, len(audio_tensor), self.frame_detector.window_size_samples):
                chunk = audio_tensor[i:i + self.frame_detector.window_size_samples]
                
                # Pad if chunk is too short
                if len(chunk) < self.frame_detector.window_size_samples:
                    padding = torch.zeros(self.frame_detector.window_size_samples - len(chunk), device=device)
                    chunk = torch.cat([chunk, padding])
                
                # Get speech probability for this chunk
                speech_prob = self.frame_detector.vad_model(chunk, 16000).item()
                speech_probs.append(speech_prob)
        
        # Reset model states after processing
        self.frame_detector.vad_model.reset_states()
        
        return speech_probs
    
    def run_microphone_test(self, device_id=None):
        """Run the frame-by-frame speech detection test using microphone"""
        
        # Initialize PyAudio
        p = pa.PyAudio()
        
        try:
            # Select audio device
            input_devices = self.list_audio_devices()
            
            if not input_devices:
                print('ERROR: No audio input device found.')
                return
            
            if device_id is None:
                device_id = -1
                while device_id not in input_devices:
                    try:
                        device_id = int(input('Please enter input device ID: '))
                    except ValueError:
                        print("Please enter a valid device ID number")
                        continue
            
            if device_id not in input_devices:
                print(f"Error: Device {device_id} not found in available input devices")
                return
            
            print(f"Using device {device_id}: {p.get_device_info_by_index(device_id)['name']}")
            
            # Define callback function
            def stream_callback(in_data, frame_count, time_info, status):
                if status:
                    print(f"Stream status: {status}")
                
                # Convert audio data and analyze
                signal = np.frombuffer(in_data, dtype=np.int16)
                self.analyze_audio_chunk(signal)
                
                return (in_data, pa.paContinue)
            
            # Open audio stream
            stream = p.open(
                format=pa.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=device_id,
                stream_callback=stream_callback,
                frames_per_buffer=FRAMES_PER_BUFFER
            )
            
            print(f'\nListening for speech... (Press Ctrl+C to stop)')
            print(f'Chunk size: {CHUNK_DURATION_MS}ms ({FRAMES_PER_BUFFER} samples)')
            print(f'Will show speech detection confirmations and 1-second proportions')
            print('=' * 60)
            
            # Start streaming
            stream.start_stream()
            
            try:
                while stream.is_active():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print('\n\nStopping...')
            finally:
                stream.stop_stream()
                stream.close()
                
                # Final statistics
                total_time = time.time() - self.start_time
                print(f"\n=== FINAL STATISTICS ===")
                print(f"Total runtime: {total_time:.1f}s")
                print(f"Total chunks processed: {self.total_chunks_processed}")
                print(f"Total frames analyzed: {self.total_frames_analyzed}")
                print(f"Chunks with speech: {self.chunks_with_speech}")
                print(f"Chunk speech rate: {(self.chunks_with_speech/self.total_chunks_processed)*100:.1f}%")
                
                # Final speech proportion
                final_analysis = self.frame_detector.get_recent_speech_proportion(
                    window_seconds=self.analysis_window_seconds
                )
                if final_analysis['sufficient_data']:
                    print(f"Final 1s speech proportion: {final_analysis['speech_proportion']:.3f}")
                print("========================")
        
        finally:
            p.terminate()

def main():
    """Main function"""
    try:
        # Allow config file to be specified as first argument
        config_path = "config.json"
        device_id = None
        
        # Parse command line arguments
        if len(sys.argv) > 1:
            # Check if first argument is a config file (ends with .json)
            if sys.argv[1].endswith('.json'):
                config_path = sys.argv[1]
                print(f"Using config file: {config_path}")
                # Check for device ID as second argument
                if len(sys.argv) > 2:
                    try:
                        device_id = int(sys.argv[2])
                        print(f"Using device ID from command line: {device_id}")
                    except ValueError:
                        print(f"Invalid device ID: {sys.argv[2]}. Will prompt for selection.")
            else:
                # First argument is device ID
                try:
                    device_id = int(sys.argv[1])
                    print(f"Using device ID from command line: {device_id}")
                except ValueError:
                    print(f"Invalid device ID: {sys.argv[1]}. Will prompt for selection.")
        
        tester = FrameSpeechTester(config_path)
        
        if not tester.frame_detector.vad_model:
            return
        
        tester.run_microphone_test(device_id)
        
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Frame-by-Frame Speech Detection Test")
    print("Usage: python test_frame_speech.py [device_id]")
    print()
    main()