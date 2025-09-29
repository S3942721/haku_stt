#!/usr/bin/env python3
"""
EOU Threshold Tuning Tool
Interactive tool for tuning End-of-Utterance detection thresholds.

This script inherits functionality from ws_stt.py and provides an interactive
interface for testing and tuning different EOU detection parameters in real-time.
"""

import sys
import os
import time
import json
import argparse
import threading
from collections import defaultdict, deque
import numpy as np

# Import everything from ws_stt
from ws_stt import *

class EOUTuningInterface:
    """Interactive interface for tuning EOU parameters"""
    
    def __init__(self):
        self.current_mode = "unified"  # unified, vad_only, text_only
        self.live_stats = {
            "vad_triggers": 0,
            "text_triggers": 0,
            "unified_triggers": 0,
            "false_positives": 0,
            "missed_eou": 0,
            "total_chunks": 0
        }
        self.recent_metrics = deque(maxlen=100)  # Store recent metric values
        
    def print_header(self):
        """Print the tuning interface header"""
        print("\n" + "="*80)
        print("🎛️  EOU THRESHOLD TUNING TOOL")
        print("="*80)
        print("Interactive tool for tuning End-of-Utterance detection parameters")
        print("Press 'h' for help, 'q' to quit")
        print("="*80)
    
    def print_help(self):
        """Print help information"""
        help_text = """
📖 HELP - EOU Tuning Tool Commands:

🎯 Mode Selection:
  1 - Unified EOU mode (all methods combined)
  2 - VAD-only mode (VAD detection only) 
  3 - Text-only mode (text detection only)

🔧 VAD Tuning (press key):
  v - Adjust VAD speech probability threshold
  p - Adjust VAD speech proportion threshold  
  w - Adjust VAD analysis window duration
  s - Adjust VAD consecutive silence threshold
  l - Adjust VAD lookback seconds
  r - Adjust VAD recent speech threshold

📝 Text Tuning:
  t - Adjust text EOU threshold
  m - Adjust minimum words required
  c - Adjust confirmation count needed

🎚️ Unified Tuning:
  u - Adjust unified EOU threshold
  V - Adjust VAD weight in unified system
  T - Adjust text weight in unified system  
  S - Adjust silence weight in unified system

📊 Statistics & Info:
  i - Show current parameter values
  z - Reset statistics
  d - Toggle detailed logging (verbose/debug)
  
🎮 Control:
  space - Mark false positive (increment FP counter)
  enter - Mark missed EOU (increment missed counter)
  h - Show this help
  q - Quit tuning tool

💡 Tips:
- Speak naturally and observe the real-time metrics
- Use space bar when EOU triggers incorrectly (false positive)
- Press enter when EOU should have triggered but didn't (missed)
- Adjust thresholds based on the live feedback
- Higher thresholds = less sensitive, Lower = more sensitive
"""
        print(help_text)
    
    def print_current_params(self, asr_system):
        """Print current parameter values"""
        print("\n📋 CURRENT PARAMETER VALUES:")
        print("-" * 50)
        
        if asr_system.unified_eou_detector:
            config = asr_system.unified_eou_detector.config
            weights = config.get("eou-weights", {})
            
            print(f"🎚️  Unified EOU:")
            print(f"   Overall Threshold: {config.get('eou-threshold', 'N/A'):.3f}")
            print(f"   VAD Weight: {weights.get('vad', 0):.2f}")
            print(f"   Text Weight: {weights.get('text', 0):.2f}") 
            print(f"   Silence Weight: {weights.get('silence', 0):.2f}")
        
        if asr_system.frame_detector:
            print(f"\n🎤 VAD Parameters:")
            print(f"   Speech Probability Threshold: {asr_system.frame_detector.speech_probability_threshold:.3f}")
            print(f"   Speech Proportion Threshold: {asr_system.frame_detector.speech_proportion_threshold:.3f}")
            print(f"   Analysis Window: {asr_system.frame_detector.analysis_window_seconds:.1f}s")
            print(f"   Consecutive Silence Threshold: {asr_system.frame_detector.consecutive_silence_threshold_seconds:.1f}s")
            print(f"   Lookback Seconds: {asr_system.frame_detector.lookback_seconds:.1f}s")
            print(f"   Recent Speech Threshold: {asr_system.frame_detector.recent_speech_threshold:.3f}")
        
        if (asr_system.unified_eou_detector and 
            asr_system.unified_eou_detector.text_detector):
            text_det = asr_system.unified_eou_detector.text_detector
            print(f"\n📝 Text EOU Parameters:")
            print(f"   Text Threshold: {text_det.threshold:.3f}")
            print(f"   Minimum Words: {text_det.min_words_for_eou}")
            print(f"   Confirmation Needed: {text_det.confirmation_needed}")
        
        print(f"\n📊 Live Statistics:")
        print(f"   VAD Triggers: {self.live_stats['vad_triggers']}")
        print(f"   Text Triggers: {self.live_stats['text_triggers']}")
        print(f"   Unified Triggers: {self.live_stats['unified_triggers']}")
        print(f"   False Positives: {self.live_stats['false_positives']}")
        print(f"   Missed EOU: {self.live_stats['missed_eou']}")
        print(f"   Total Chunks: {self.live_stats['total_chunks']}")
        
        if self.recent_metrics:
            recent = list(self.recent_metrics)[-10:]  # Last 10 measurements
            avg_confidence = np.mean([m.get('confidence', 0) for m in recent])
            print(f"   Recent Avg Confidence: {avg_confidence:.3f}")

class EOUTuningASR(OnlineASRWithPunctuation):
    """Extended ASR system with tuning capabilities"""
    
    def __init__(self, tuning_interface, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tuning_interface = tuning_interface
        self.detailed_logging = True  # Start with detailed logging
        
    def transcribe_chunk_with_tuning(self, audio_chunk):
        """Enhanced transcribe_chunk with tuning metrics"""
        # Call parent method
        raw_text, punct_text, is_eou, complete_utterance = self.transcribe_chunk(audio_chunk)
        
        # Update statistics
        self.tuning_interface.live_stats['total_chunks'] += 1
        
        # Collect detailed metrics if we have the unified detector
        if self.unified_eou_detector and raw_text.strip():
            # Get the last EOU result if available
            if hasattr(self.unified_eou_detector, 'last_eou_result'):
                result = self.unified_eou_detector.last_eou_result
                self.tuning_interface.recent_metrics.append({
                    'confidence': result.get('confidence', 0),
                    'method_scores': result.get('method_scores', {}),
                    'triggered_by': result.get('triggered_by', []),
                    'text': raw_text[:50]  # First 50 chars
                })
                
                # Update trigger counts
                if 'vad' in result.get('triggered_by', []):
                    self.tuning_interface.live_stats['vad_triggers'] += 1
                if 'text' in result.get('triggered_by', []):
                    self.tuning_interface.live_stats['text_triggers'] += 1
                if result.get('is_eou', False):
                    self.tuning_interface.live_stats['unified_triggers'] += 1
        
        return raw_text, punct_text, is_eou, complete_utterance
    
    def print_live_metrics(self):
        """Print live metrics in a compact format with proper line separation"""
        if not self.tuning_interface.recent_metrics:
            return
            
        recent = list(self.tuning_interface.recent_metrics)[-1]
        method_scores = recent.get('method_scores', {})
        confidence = recent.get('confidence', 0)
        triggered = recent.get('triggered_by', [])
        
        # Compact live display with clear line separation
        vad_score = method_scores.get('vad', 0)
        text_score = method_scores.get('text', 0)
        silence_score = method_scores.get('silence', 0)
        
        # Clear the current line and print on a new line
        print(f"\n📊 VAD:{vad_score:.1f} TXT:{text_score:.1f} SIL:{silence_score:.1f} "
              f"CONF:{confidence:.3f} TRIG:{','.join(triggered) or 'none':<8} "
              f"FP:{self.tuning_interface.live_stats['false_positives']} "
              f"MISS:{self.tuning_interface.live_stats['missed_eou']}")
        
        # Add separator line for clarity
        print("-" * 60)

class InteractiveEOUTuner:
    """Main tuning application"""
    
    def __init__(self):
        self.interface = EOUTuningInterface()
        self.asr_system = None
        self.input_thread = None
        self.running = False
        
    def setup_asr_system(self, config_path="config.json"):
        """Initialize ASR system with tuning capabilities"""
        # Load config
        config = load_config(config_path)
        
        # Force verbose logging for tuning
        config["log-level"] = "verbose"
        config["enable-text-eou"] = True  # Enable both methods for comparison
        
        # Setup logging
        setup_global_logging(
            log_level=config.get("log-level", "verbose"),
            log_format=config.get("log-format"),
            log_file=config.get("log-file"),
            log_max_size=config.get("log-max-size", 10),
            log_backup_count=config.get("log-backup-count", 3)
        )
        
        # Create log config
        log_config = {
            "log_level": config.get("log-level", "verbose"),
            "log_format": config.get("log-format"),
            "log_file": config.get("log-file"),
            "log_max_size": config.get("log-max-size", 10),
            "log_backup_count": config.get("log-backup-count", 3)
        }
        
        # Initialize tuning ASR system
        self.asr_system = EOUTuningASR(
            tuning_interface=self.interface,
            asr_model_name=config["asr-model"],
            punct_model_name=config["punct-model"],
            lookahead_size=config["lookahead"],
            decoder_type=config["decoder"],
            quiet_mode=False,
            enable_eou=True,
            websocket_server=None,
            log_config=log_config
        )
        
        # Setup VAD detector
        vad_config = {
            "vad-speech-threshold": config.get("vad-speech-threshold", 0.25),
            "vad-speech-proportion-threshold": config.get("vad-speech-proportion-threshold", 0.15),
            "vad-analysis-window": config.get("vad-analysis-window", 2.0),
            "vad-consecutive-silence-threshold": config.get("vad-consecutive-silence-threshold", 0.8),
            "vad-min-frames-for-eou": config.get("vad-min-frames-for-eou", 0.5),
            "vad-recent-speech-lookback-seconds": config.get("vad-recent-speech-lookback-seconds", 1.5),
            "vad-recent-speech-threshold": config.get("vad-recent-speech-threshold", 0.10)
        }
        
        self.asr_system.frame_detector = FrameLevelSpeechDetector(
            config["vad-model"], vad_config, log_config
        )
        
        # Setup unified EOU detector
        unified_eou_config = {
            "eou-threshold": config.get("eou-threshold", 0.7),
            "eou-weights": {
                "vad": config.get("eou-vad-weight", 0.6),
                "text": config.get("eou-text-weight", 0.3),
                "silence": config.get("eou-silence-weight", 0.1)
            },
            "eou-min-words": config.get("eou-min-words", 4),
            "eou-confirmation-needed": config.get("eou-confirmation-needed", 2),
            "enable-text-eou": True,
            "enable-vad-eou": True,
            "enable-silence-eou": True,
            "text-eou-threshold": 0.8
        }
        
        self.asr_system.unified_eou_detector = UnifiedEOUDetector(
            log_config=log_config,
            config=unified_eou_config
        )
        
        # Connect VAD detector
        self.asr_system.unified_eou_detector.set_vad_detector(self.asr_system.frame_detector)
        
        print(f"✅ ASR System initialized with tuning capabilities")
        print(f"📊 VAD Model: {config['vad-model']}")
        print(f"🎯 EOU Methods: VAD + Text")
        
    def adjust_parameter(self, param_name, current_value, param_type=float, min_val=0.0, max_val=1.0, step=0.05):
        """Interactive parameter adjustment"""
        print(f"\n🔧 Adjusting {param_name}")
        print(f"Current value: {current_value}")
        print(f"Range: {min_val} - {max_val}, Step: {step}")
        print("Controls: +/- to adjust, Enter to confirm, Esc to cancel")
        
        new_value = current_value
        
        while True:
            try:
                key = input(f"{param_name}: {new_value:.3f} (+/-/Enter/Esc): ").strip().lower()
                
                if key == '' or key == 'enter':
                    return new_value
                elif key == 'esc' or key == 'escape':
                    return current_value
                elif key == '+':
                    new_value = min(max_val, new_value + step)
                elif key == '-':
                    new_value = max(min_val, new_value - step)
                elif key.replace('.', '').replace('-', '').isdigit():
                    # Direct value input
                    new_value = param_type(key)
                    new_value = max(min_val, min(max_val, new_value))
                else:
                    print("Invalid input. Use +, -, a number, Enter, or Esc")
                    
            except ValueError:
                print("Invalid number format")
            except KeyboardInterrupt:
                return current_value
    
    def handle_user_input(self):
        """Handle user input in separate thread"""
        def getch():
            """Cross-platform single character input"""
            try:
                import msvcrt  # Windows
                return msvcrt.getch().decode('utf-8', errors='ignore')
            except ImportError:
                # Unix/Linux - try multiple approaches
                try:
                    import tty, termios
                    if hasattr(tty, 'setraw'):
                        fd = sys.stdin.fileno()
                        old_settings = termios.tcgetattr(fd)
                        try:
                            tty.setraw(sys.stdin.fileno())
                            ch = sys.stdin.read(1)
                        except KeyboardInterrupt:
                            ch = '\x03'  # Ctrl+C
                        finally:
                            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        return ch
                    else:
                        raise ImportError("tty.setraw not available")
                except (ImportError, AttributeError, OSError):
                    # Fallback to regular input for systems without tty support
                    try:
                        return input("Command: ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        return 'q'
        
        print("\n🎮 Interactive tuning active. Press keys to adjust parameters...")
        print("💡 If single-key input is not working, you can type full commands like 'v' and press Enter.")
        
        while self.running:
            try:
                key = getch()
                if isinstance(key, bytes):
                    key = key.decode('utf-8', errors='ignore')
                
                # Handle multi-character input (from fallback mode)
                if len(key) > 1:
                    key = key[0]  # Take first character
                
                key = key.lower()
                
                if key == 'q':
                    print("\n🛑 Quit requested")
                    self.running = False
                    break
                elif key == 'h':
                    self.interface.print_help()
                elif key == 'i':
                    self.interface.print_current_params(self.asr_system)
                elif key == 'z':
                    # Reset statistics
                    for stat in self.interface.live_stats:
                        self.interface.live_stats[stat] = 0
                    self.interface.recent_metrics.clear()
                    print("\n📊 Statistics reset")
                elif key == ' ':
                    # Mark false positive
                    self.interface.live_stats['false_positives'] += 1
                    print(f"\n❌ False positive marked (Total: {self.interface.live_stats['false_positives']})")
                elif key == '\r' or key == '\n':
                    # Mark missed EOU
                    self.interface.live_stats['missed_eou'] += 1
                    print(f"\n⚠️  Missed EOU marked (Total: {self.interface.live_stats['missed_eou']})")
                elif key == 'd':
                    # Toggle detailed logging
                    self.asr_system.detailed_logging = not self.asr_system.detailed_logging
                    new_level = "verbose" if self.asr_system.detailed_logging else "info"
                    setup_global_logging(log_level=new_level)
                    print(f"\n🔧 Logging level: {new_level}")
                
                # Parameter adjustment keys
                elif key == 'v' and self.asr_system.frame_detector:
                    # VAD speech threshold
                    new_val = self.adjust_parameter(
                        "VAD Speech Probability Threshold",
                        self.asr_system.frame_detector.speech_probability_threshold,
                        float, 0.0, 1.0, 0.05
                    )
                    self.asr_system.frame_detector.speech_probability_threshold = new_val
                    print(f"✅ VAD speech threshold set to {new_val:.3f}")
                
                elif key == 'p' and self.asr_system.frame_detector:
                    # VAD speech proportion threshold
                    new_val = self.adjust_parameter(
                        "VAD Speech Proportion Threshold",
                        self.asr_system.frame_detector.speech_proportion_threshold,
                        float, 0.0, 1.0, 0.05
                    )
                    self.asr_system.frame_detector.speech_proportion_threshold = new_val
                    print(f"✅ VAD speech proportion threshold set to {new_val:.3f}")
                
                elif key == 'w' and self.asr_system.frame_detector:
                    # VAD analysis window
                    new_val = self.adjust_parameter(
                        "VAD Analysis Window (seconds)",
                        self.asr_system.frame_detector.analysis_window_seconds,
                        float, 0.5, 10.0, 0.5
                    )
                    self.asr_system.frame_detector.analysis_window_seconds = new_val
                    # Recalculate frames per window
                    self.asr_system.frame_detector.frames_per_analysis_window = int(
                        new_val / self.asr_system.frame_detector.vad_frame_duration
                    )
                    print(f"✅ VAD analysis window set to {new_val:.1f}s")
                
                elif key == 's' and self.asr_system.frame_detector:
                    # VAD consecutive silence threshold
                    new_val = self.adjust_parameter(
                        "VAD Consecutive Silence Threshold (seconds)",
                        self.asr_system.frame_detector.consecutive_silence_threshold_seconds,
                        float, 0.1, 5.0, 0.1
                    )
                    self.asr_system.frame_detector.consecutive_silence_threshold_seconds = new_val
                    self.asr_system.frame_detector.consecutive_silence_threshold = int(
                        new_val / self.asr_system.frame_detector.vad_frame_duration
                    )
                    print(f"✅ VAD consecutive silence threshold set to {new_val:.1f}s")
                
                elif key == 'l' and self.asr_system.frame_detector:
                    # VAD lookback seconds
                    new_val = self.adjust_parameter(
                        "VAD Lookback Seconds",
                        self.asr_system.frame_detector.lookback_seconds,
                        float, 0.5, 10.0, 0.5
                    )
                    self.asr_system.frame_detector.lookback_seconds = new_val
                    print(f"✅ VAD lookback seconds set to {new_val:.1f}s")
                
                elif key == 'r' and self.asr_system.frame_detector:
                    # VAD recent speech threshold
                    new_val = self.adjust_parameter(
                        "VAD Recent Speech Threshold",
                        self.asr_system.frame_detector.recent_speech_threshold,
                        float, 0.0, 1.0, 0.05
                    )
                    self.asr_system.frame_detector.recent_speech_threshold = new_val
                    print(f"✅ VAD recent speech threshold set to {new_val:.3f}")
                
                elif key == 't' and self.asr_system.unified_eou_detector:
                    # Text EOU threshold
                    text_det = self.asr_system.unified_eou_detector.text_detector
                    if text_det:
                        new_val = self.adjust_parameter(
                            "Text EOU Threshold",
                            text_det.threshold,
                            float, 0.0, 1.0, 0.05
                        )
                        text_det.threshold = new_val
                        print(f"✅ Text EOU threshold set to {new_val:.3f}")
                
                elif key == 'm' and self.asr_system.unified_eou_detector:
                    # Text minimum words
                    text_det = self.asr_system.unified_eou_detector.text_detector
                    if text_det:
                        new_val = self.adjust_parameter(
                            "Text Minimum Words",
                            text_det.min_words_for_eou,
                            int, 1, 20, 1
                        )
                        text_det.min_words_for_eou = int(new_val)
                        print(f"✅ Text minimum words set to {int(new_val)}")
                
                elif key == 'c' and self.asr_system.unified_eou_detector:
                    # Text confirmation count
                    text_det = self.asr_system.unified_eou_detector.text_detector
                    if text_det:
                        new_val = self.adjust_parameter(
                            "Text Confirmation Count",
                            text_det.confirmation_needed,
                            int, 1, 10, 1
                        )
                        text_det.confirmation_needed = int(new_val)
                        print(f"✅ Text confirmation count set to {int(new_val)}")
                
                elif key == 'u' and self.asr_system.unified_eou_detector:
                    # Unified EOU threshold
                    new_val = self.adjust_parameter(
                        "Unified EOU Threshold",
                        self.asr_system.unified_eou_detector.config["eou-threshold"],
                        float, 0.0, 1.0, 0.05
                    )
                    self.asr_system.unified_eou_detector.config["eou-threshold"] = new_val
                    print(f"✅ Unified EOU threshold set to {new_val:.3f}")
                
                elif key.upper() == 'V' and self.asr_system.unified_eou_detector:
                    # VAD weight in unified system
                    weights = self.asr_system.unified_eou_detector.config["eou-weights"]
                    new_val = self.adjust_parameter(
                        "VAD Weight in Unified System",
                        weights["vad"],
                        float, 0.0, 1.0, 0.1
                    )
                    weights["vad"] = new_val
                    self.asr_system.unified_eou_detector._validate_weights()
                    print(f"✅ VAD weight set to {new_val:.2f}")
                
                elif key.upper() == 'T' and self.asr_system.unified_eou_detector:
                    # Text weight in unified system  
                    weights = self.asr_system.unified_eou_detector.config["eou-weights"]
                    new_val = self.adjust_parameter(
                        "Text Weight in Unified System",
                        weights["text"],
                        float, 0.0, 1.0, 0.1
                    )
                    weights["text"] = new_val
                    self.asr_system.unified_eou_detector._validate_weights()
                    print(f"✅ Text weight set to {new_val:.2f}")
                
                elif key.upper() == 'S' and self.asr_system.unified_eou_detector:
                    # Silence weight in unified system
                    weights = self.asr_system.unified_eou_detector.config["eou-weights"]
                    new_val = self.adjust_parameter(
                        "Silence Weight in Unified System", 
                        weights["silence"],
                        float, 0.0, 1.0, 0.1
                    )
                    weights["silence"] = new_val
                    self.asr_system.unified_eou_detector._validate_weights()
                    print(f"✅ Silence weight set to {new_val:.2f}")
                    
            except Exception as e:
                print(f"\n❌ Error handling input: {e}")
                continue
    
    def run_tuning_session(self, device_id=None):
        """Run the interactive tuning session"""
        self.interface.print_header()
        self.interface.print_help()
        self.interface.print_current_params(self.asr_system)
        
        # Start input handler thread
        self.running = True
        self.input_thread = threading.Thread(target=self.handle_user_input, daemon=True)
        self.input_thread.start()
        
        # Run ASR with live metrics
        try:
            self.run_asr_with_live_feedback(device_id)
        except KeyboardInterrupt:
            print("\n🛑 Tuning session interrupted")
        finally:
            self.running = False
            if self.input_thread and self.input_thread.is_alive():
                self.input_thread.join(timeout=1.0)
    
    def run_asr_with_live_feedback(self, device_id):
        """Run ASR with live feedback display"""
        # Use microphone for simplicity in tuning
        import pyaudio as pa
        
        p = pa.PyAudio()
        
        try:
            # List and select audio device
            input_devices = []
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                if dev.get('maxInputChannels'):
                    input_devices.append(i)
            
            if not input_devices:
                print("❌ No audio input devices found")
                return
            
            if device_id is None:
                device_id = input_devices[0]  # Use first available
            
            print(f"🎤 Using audio device: {p.get_device_info_by_index(device_id)['name']}")
            
            # Calculate audio parameters
            chunk_size_ms = self.asr_system.lookahead_size + ENCODER_STEP_LENGTH
            frames_per_buffer = int(SAMPLE_RATE * chunk_size_ms / 1000) - 1
            
            print(f"🔊 Chunk size: {chunk_size_ms}ms, Frames: {frames_per_buffer}")
            print(f"🎯 Ready for tuning! Speak naturally and watch the metrics...")
            print(f"📊 Live metrics will appear below:")
            
            # Callback function for audio processing
            def audio_callback(in_data, frame_count, time_info, status):
                if not self.running:
                    return (in_data, pa.paComplete)
                
                # Convert and process audio
                signal = np.frombuffer(in_data, dtype=np.int16)
                raw_text, punct_text, is_eou, complete_utterance = self.asr_system.transcribe_chunk_with_tuning(signal)
                
                # Display live metrics
                self.asr_system.print_live_metrics()
                
                # Handle complete utterances
                if complete_utterance:
                    print(f"\n{'='*80}")
                    print(f"🎯 COMPLETE UTTERANCE: {complete_utterance}")
                    print(f"{'='*80}")
                    print()  # Add extra space for clarity
                
                return (in_data, pa.paContinue)
            
            # Open and start audio stream
            stream = p.open(
                format=pa.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=device_id,
                stream_callback=audio_callback,
                frames_per_buffer=frames_per_buffer
            )
            
            stream.start_stream()
            
            # Keep running until stopped
            while self.running and stream.is_active():
                time.sleep(0.1)
            
            stream.stop_stream()
            stream.close()
            
        finally:
            p.terminate()
            print("\n🎤 Audio stream stopped")

def main():
    """Main entry point for EOU tuning tool"""
    parser = argparse.ArgumentParser(
        description="Interactive EOU Threshold Tuning Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eou_tuning_tool.py
  python eou_tuning_tool.py --config custom_config.json --device 1
        """
    )
    
    parser.add_argument(
        "--config",
        default="config.json", 
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio device ID (will auto-select if not provided)"
    )
    
    args = parser.parse_args()
    
    # Initialize and run tuner
    tuner = InteractiveEOUTuner()
    
    try:
        print("🚀 Initializing EOU Tuning Tool...")
        tuner.setup_asr_system(args.config)
        tuner.run_tuning_session(args.device)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("👋 EOU Tuning Tool finished")

if __name__ == "__main__":
    main()