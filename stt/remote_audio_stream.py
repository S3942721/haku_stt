#!/usr/bin/env python3
"""
Remote Audio Stream Receiver for ASR Integration
Receives RTP audio stream from Pepper robot for speech recognition
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject, GLib
import numpy as np
import threading
import queue
import time
import signal

class RemoteAudioStream:
    def __init__(self, listen_port=5004, sample_rate=16000, verbose=False):
        self.listen_port = listen_port
        self.sample_rate = sample_rate
        self.verbose = verbose
        self.pipeline = None
        self.loop = None
        self.loop_thread = None
        
        # Audio data queue for ASR consumption
        self.audio_queue = queue.Queue(maxsize=100)
        self.is_running = False
        self.is_connected = False
        
        # PyAudio-compatible buffering system
        self.target_chunk_size = None  # Will be set when read_audio is called
        self.audio_accumulator = np.array([], dtype=np.int16)  # Persistent accumulator
        self.chunk_ready_event = threading.Event()
        self.lock = threading.Lock()
        
        # Statistics for debugging
        self.packets_received = 0
        self.bytes_received = 0
        self.last_audio_time = 0
        self.volume_history = []
        self.connection_start_time = 0
        
        # Audio format debugging
        self.audio_format_info = {}
        self.sample_count_history = []
        
        # Initialize GStreamer
        Gst.init(None)
        
    def create_pipeline(self):
        """Create GStreamer pipeline for receiving audio data (not playing)"""
        # Enhanced pipeline with format debugging
        pipeline_desc = (
            f"udpsrc port={self.listen_port} "
            "caps=\"application/x-rtp,media=(string)audio,clock-rate=(int)44100,encoding-name=(string)L16,payload=(int)96,channels=(int)1\" ! "
            "rtpjitterbuffer "
            "latency=100 "  # Lower latency for real-time processing
            "drop-on-latency=true "
            "max-dropout-time=500 "
            "max-misorder-time=50 ! "
            "rtpL16depay ! "
            "audioconvert ! "
            f"audioresample ! "
            f"audio/x-raw,rate={self.sample_rate},channels=1,format=S16LE ! "
            "appsink name=sink "
            "emit-signals=true "
            "max-buffers=20 "
            "drop=true "
            "sync=false"
        )
        
        if self.verbose:
            print(f"AUDIO: Creating pipeline: {pipeline_desc}")
        
        try:
            self.pipeline = Gst.parse_launch(pipeline_desc)
            
            # Get the appsink element
            self.appsink = self.pipeline.get_by_name("sink")
            if not self.appsink:
                print("ERROR: Could not get appsink element")
                return False
                
            # Connect to new-sample signal
            self.appsink.connect("new-sample", self.on_new_sample)
            
            # Set up message handling
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self.on_message)
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to create pipeline: {e}")
            return False
    
    def analyze_audio_format(self, audio_data, caps):
        """Analyze and log audio format information for debugging"""
        try:
            if caps:
                structure = caps.get_structure(0)
                format_info = {
                    'rate': structure.get_int('rate')[1] if structure.get_int('rate')[0] else 'unknown',
                    'channels': structure.get_int('channels')[1] if structure.get_int('channels')[0] else 'unknown',
                    'format': structure.get_string('format') or 'unknown'
                }
                
                # Only log if format changed
                if format_info != self.audio_format_info:
                    self.audio_format_info = format_info
                    if self.verbose:
                        print(f"AUDIO FORMAT: {format_info}")
            
            # Track sample count distribution
            sample_count = len(audio_data)
            self.sample_count_history.append(sample_count)
            if len(self.sample_count_history) > 100:
                self.sample_count_history.pop(0)
            
            # Log format details periodically
            if self.packets_received % 50 == 0 and self.verbose:
                unique_counts = list(set(self.sample_count_history[-50:]))
                avg_count = np.mean(self.sample_count_history[-50:]) if self.sample_count_history else 0
                print(f"AUDIO SAMPLES: avg={avg_count:.1f}, range={min(unique_counts) if unique_counts else 0}-{max(unique_counts) if unique_counts else 0}, "
                      f"dtype={audio_data.dtype}, shape={audio_data.shape}")
                
        except Exception as e:
            if self.verbose:
                print(f"FORMAT DEBUG ERROR: {e}")
    
    def validate_and_normalize_audio(self, audio_data):
        """Validate and normalize audio data to match PyAudio format exactly"""
        try:
            # Ensure it's a numpy array
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)
            
            # Debug original format
            original_dtype = audio_data.dtype
            original_shape = audio_data.shape
            original_range = [int(np.min(audio_data)), int(np.max(audio_data))] if len(audio_data) > 0 else [0, 0]
            
            # Ensure correct data type (int16 for PyAudio compatibility)
            if audio_data.dtype != np.int16:
                if self.verbose and self.packets_received % 50 == 0:
                    print(f"AUDIO: Converting {original_dtype} to int16")
                
                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    # Convert from float [-1, 1] to int16 [-32768, 32767]
                    audio_data = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)
            
            # Ensure 1D array (mono audio)
            if audio_data.ndim > 1:
                if audio_data.shape[1] == 1:
                    # Convert from (N, 1) to (N,)
                    audio_data = audio_data.flatten()
                elif audio_data.shape[0] == 1:
                    # Convert from (1, N) to (N,)
                    audio_data = audio_data.flatten()
                else:
                    # Multi-channel - take first channel
                    audio_data = audio_data[:, 0] if audio_data.shape[1] > 1 else audio_data[0, :]
                    if self.verbose:
                        print(f"AUDIO: Multi-channel detected, using first channel")
            
            # Validate range
            if len(audio_data) > 0:
                data_min, data_max = np.min(audio_data), np.max(audio_data)
                if data_min < -32768 or data_max > 32767:
                    if self.verbose:
                        print(f"AUDIO: Clipping out-of-range values: [{data_min}, {data_max}] -> [-32768, 32767]")
                    audio_data = np.clip(audio_data, -32768, 32767)
            
            # Log transformation if significant change occurred
            if (self.verbose and self.packets_received % 50 == 0 and 
                (original_dtype != np.int16 or original_shape != audio_data.shape)):
                new_range = [int(np.min(audio_data)), int(np.max(audio_data))] if len(audio_data) > 0 else [0, 0]
                print(f"AUDIO TRANSFORM: {original_dtype}{original_shape}[{original_range[0]}:{original_range[1]}] -> "
                      f"{audio_data.dtype}{audio_data.shape}[{new_range[0]}:{new_range[1]}]")
            
            return audio_data
            
        except Exception as e:
            print(f"ERROR: Audio validation failed: {e}")
            # Return original data if validation fails
            return audio_data if isinstance(audio_data, np.ndarray) else np.array(audio_data, dtype=np.int16)
    
    def on_new_sample(self, sink):
        """Handle new audio sample from GStreamer - feed into PyAudio-like buffer"""
        try:
            sample = sink.emit("pull-sample")
            if sample:
                buffer = sample.get_buffer()
                caps = sample.get_caps()
                
                # Extract audio data
                success, map_info = buffer.map(Gst.MapFlags.READ)
                if success:
                    # Convert to numpy array (GStreamer gives us int16 data)
                    raw_audio_data = np.frombuffer(map_info.data, dtype=np.int16)
                    buffer.unmap(map_info)
                    
                    # Analyze format for debugging
                    self.analyze_audio_format(raw_audio_data, caps)
                    
                    # Validate and normalize to match PyAudio format
                    audio_data = self.validate_and_normalize_audio(raw_audio_data)
                    
                    # Update statistics
                    self.packets_received += 1
                    self.bytes_received += len(audio_data) * 2  # 16-bit samples
                    self.last_audio_time = time.time()
                    
                    # Calculate volume (RMS)
                    if len(audio_data) > 0:
                        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                        self.volume_history.append(rms)
                        if len(self.volume_history) > 50:  # Keep last 50 samples
                            self.volume_history.pop(0)
                        
                        # Verbose logging with format details
                        if self.verbose and self.packets_received % 50 == 0:
                            avg_volume = np.mean(self.volume_history) if self.volume_history else 0
                            duration = time.time() - self.connection_start_time if self.connection_start_time else 0
                            print(f"AUDIO: Packet #{self.packets_received} | "
                                  f"RMS: {rms:.0f} | Avg: {avg_volume:.0f} | "
                                  f"Samples: {len(audio_data)} | Duration: {duration:.1f}s | "
                                  f"Type: {audio_data.dtype} | Shape: {audio_data.shape}")
                    
                    # Add to PyAudio-compatible accumulator
                    self.add_to_accumulator(audio_data)
                    
                    if not self.is_connected:
                        self.is_connected = True
                        self.connection_start_time = time.time()
                        print(f"AUDIO: Remote stream connected - format: {self.audio_format_info}")
                
        except Exception as e:
            if self.verbose:
                print(f"ERROR: Failed to process audio sample: {e}")
                import traceback
                traceback.print_exc()
        
        return Gst.FlowReturn.OK

    def add_to_accumulator(self, audio_data):
        """Add audio data to accumulator and create chunks when ready"""
        with self.lock:
            # Add new data to accumulator
            self.audio_accumulator = np.concatenate([self.audio_accumulator, audio_data])
            
            # If we have a target chunk size and enough data, create chunks
            if self.target_chunk_size is not None:
                while len(self.audio_accumulator) >= self.target_chunk_size:
                    # Extract exactly target_chunk_size samples
                    chunk = self.audio_accumulator[:self.target_chunk_size].copy()
                    self.audio_accumulator = self.audio_accumulator[self.target_chunk_size:]
                    
                    # Add to queue for consumption
                    try:
                        self.audio_queue.put_nowait(chunk)
                        self.chunk_ready_event.set()
                    except queue.Full:
                        # Queue is full, drop oldest chunk
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.put_nowait(chunk)
                        except queue.Empty:
                            pass

    def read_audio_pyaudio_compatible(self, chunk_size, timeout=1.0):
        """Read audio data in PyAudio-compatible way - always returns exact chunk_size"""
        if not self.is_running:
            return None
        
        # Set target chunk size for accumulator
        if self.target_chunk_size != chunk_size:
            with self.lock:
                self.target_chunk_size = chunk_size
                if self.verbose:
                    print(f"AUDIO: Set target chunk size to {chunk_size} samples")
                # Process any existing data in accumulator
                while len(self.audio_accumulator) >= self.target_chunk_size:
                    chunk = self.audio_accumulator[:self.target_chunk_size].copy()
                    self.audio_accumulator = self.audio_accumulator[self.target_chunk_size:]
                    try:
                        self.audio_queue.put_nowait(chunk)
                        self.chunk_ready_event.set()
                    except queue.Full:
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.put_nowait(chunk)
                        except queue.Empty:
                            pass
        
        try:
            # Wait for chunk to be ready
            audio_data = self.audio_queue.get(timeout=timeout)
            
            # Ensure exact chunk size (should always be true now, but safety check)
            if len(audio_data) != chunk_size:
                if self.verbose:
                    print(f"WARNING: Chunk size mismatch - expected {chunk_size}, got {len(audio_data)}")
                if len(audio_data) < chunk_size:
                    # Pad with zeros
                    padding = np.zeros(chunk_size - len(audio_data), dtype=np.int16)
                    audio_data = np.concatenate([audio_data, padding])
                else:
                    # Truncate
                    audio_data = audio_data[:chunk_size]
            
            # Final validation
            audio_data = self.validate_and_normalize_audio(audio_data)
            
            # Debug chunk info periodically
            if self.verbose and hasattr(self, '_chunk_count'):
                self._chunk_count += 1
            else:
                self._chunk_count = 1
                
            if self.verbose and self._chunk_count % 20 == 0:
                print(f"PYAUDIO CHUNK READ: size={len(audio_data)}, dtype={audio_data.dtype}, "
                      f"range=[{np.min(audio_data)}, {np.max(audio_data)}], "
                      f"rms={np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)):.0f}")
            
            return audio_data
            
        except queue.Empty:
            if self.verbose:
                print(f"AUDIO: Timeout waiting for chunk (target_size={self.target_chunk_size}, queue_size={self.audio_queue.qsize()})")
            return None

    def read_audio(self, chunk_size=None, timeout=1.0):
        """Read audio data - delegates to PyAudio-compatible method if chunk_size specified"""
        if chunk_size is not None:
            return self.read_audio_pyaudio_compatible(chunk_size, timeout)
        else:
            # Original behavior for compatibility
            if not self.is_running:
                return None
                
            try:
                audio_data = self.audio_queue.get(timeout=timeout)
                audio_data = self.validate_and_normalize_audio(audio_data)
                return audio_data
            except queue.Empty:
                return None
    
    def on_message(self, bus, message):
        """Handle GStreamer messages"""
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"AUDIO ERROR: {err}")
            if self.verbose:
                print(f"AUDIO DEBUG: {debug}")
            self.stop()
        elif message.type == Gst.MessageType.EOS:
            print("AUDIO: End of stream")
            self.stop()
        elif message.type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                if self.verbose:
                    print(f"AUDIO: Pipeline state changed from {old_state.value_nick} to {new_state.value_nick}")
        elif message.type == Gst.MessageType.STREAM_START:
            print("AUDIO: Stream started - waiting for audio data...")
    
    def start(self):
        """Start receiving audio stream"""
        if self.is_running:
            return True
            
        if not self.create_pipeline():
            return False
            
        print(f"AUDIO: Starting remote audio stream receiver on port {self.listen_port}")
        print(f"AUDIO: Target sample rate: {self.sample_rate} Hz")
        
        # Start the pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("ERROR: Failed to start pipeline")
            return False
        
        # Create and start GLib main loop in separate thread
        self.loop = GLib.MainLoop()
        self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.loop_thread.start()
        
        self.is_running = True
        return True
    
    def _run_loop(self):
        """Run GLib main loop in separate thread"""
        try:
            self.loop.run()
        except Exception as e:
            print(f"ERROR: GLib main loop error: {e}")
    
    def stop(self):
        """Stop audio stream receiver"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.is_connected = False
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            
        if self.loop and self.loop.is_running():
            self.loop.quit()
            
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=2.0)
        
        # Print final statistics
        if self.verbose and self.packets_received > 0:
            duration = time.time() - self.connection_start_time if self.connection_start_time else 0
            avg_volume = np.mean(self.volume_history) if self.volume_history else 0
            print(f"AUDIO: Session ended - {self.packets_received} packets, "
                  f"{self.bytes_received} bytes, {duration:.1f}s, avg volume: {avg_volume:.0f}")
    
    def is_stream_active(self):
        """Check if audio stream is actively receiving data"""
        if not self.is_running or not self.is_connected:
            return False
            
        # Consider stream active if we received data in the last 2 seconds
        return (time.time() - self.last_audio_time) < 2.0
    
    def get_status(self):
        """Get current stream status"""
        return {
            'running': self.is_running,
            'connected': self.is_connected,
            'active': self.is_stream_active(),
            'packets_received': self.packets_received,
            'bytes_received': self.bytes_received,
            'queue_size': self.audio_queue.qsize(),
            'avg_volume': np.mean(self.volume_history) if self.volume_history else 0,
            'sample_rate': self.sample_rate,
            'port': self.listen_port
        }

def test_remote_audio_stream():
    """Test the remote audio stream receiver"""
    print("Testing remote audio stream receiver...")
    
    stream = RemoteAudioStream(verbose=True)
    
    if not stream.start():
        print("Failed to start audio stream")
        return False
    
    print("Listening for audio... Press Ctrl+C to stop")
    
    try:
        while True:
            audio_data = stream.read_audio(timeout=1.0)
            if audio_data is not None:
                rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                print(f"Received audio: {len(audio_data)} samples, RMS: {rms:.0f}")
            else:
                print("No audio data received")
            
            status = stream.get_status()
            if status['packets_received'] % 20 == 0 and status['packets_received'] > 0:
                print(f"Status: {status}")
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop()
    
    return True

if __name__ == "__main__":
    test_remote_audio_stream()
