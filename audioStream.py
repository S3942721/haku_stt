# -*- coding: utf-8 -*-

import sys
import time
import socket
import numpy as np
from naoqi import ALModule, ALProxy
import traceback

# Don't import GStreamer at module level - it intercepts command line arguments
# We'll import it only when needed in the methods

# Audio streaming constants
SAMPLE_RATE = 16000  # Hz - 16kHz for audio streaming
CHUNK_SIZE = 256     # samples per packet (16ms duration at 16kHz)

class AudioStreamModule(ALModule):
    """
    Audio streaming module using GStreamer pipeline for real-time audio transmission
    Simple and efficient approach similar to pepper_audio_sender
    """

    def __init__(self, strModuleName, strNaoIp, port, target_host, target_port):
        
        global global_speaking_state
        global_speaking_state = False  # Initialize speaking state
        
        self.led_service = ALProxy('ALLeds')
        try:
            ALModule.__init__(self, strModuleName)
            
            self.BIND_PYTHON(self.getName(), "callback")
            self.strNaoIp = strNaoIp
            self.port = port
            
            # Network settings
            self.target_host = target_host
            self.target_port = target_port
            
            # GStreamer pipeline
            self.pipeline = None
            self.gstreamer_available = None  # Lazy check
            
            # Audio streaming state
            self.isStarted = False
            self.isStreamingEnabled = False
            self.isSpeaking = False  # Track speaking state
            
            # Statistics
            self.stream_start_time = 0
            self.packets_sent = 0
            self.network_errors = 0
            
            print("AUDIO: AudioStreamModule initialized - target: {}:{}".format(target_host, target_port))
            
            # Memory setup
            self.memory = ALProxy("ALMemory", self.strNaoIp, self.port)
            self.memory.subscribeToEvent("ControlUDPAudioStreaming", self.getName(), "control_streaming")
            self.memory.subscribeToEvent("Speaking", self.getName(), "on_speaking_event")
            
        except BaseException as err:
            print("ERR: AudioStreamModule: loading error: %s" % str(err))

    def check_gstreamer_availability(self):
        """Check if GStreamer is available - only when needed"""
        if self.gstreamer_available is not None:
            return self.gstreamer_available
            
        try:
            # Add GStreamer path for Python 2.7
            sys.path.append('/usr/lib/python2.7/site-packages/gst-0.10')
            
            import gst
            import gobject
            gobject.threads_init()
            
            # Store references for later use
            self.gst = gst
            self.gobject = gobject
            self.gstreamer_available = True
            
            print("AUDIO: GStreamer available")
            return True
            
        except ImportError as e:
            print("WARN: GStreamer not available: %s" % e)
            self.gstreamer_available = False
            return False

    def create_gstreamer_pipeline(self):
        """Create GStreamer pipeline for audio streaming with consistent chunk sizes"""
        if not self.check_gstreamer_availability():
            print("ERR: Cannot create pipeline - GStreamer not available")
            return False
            
        try:
            # Create GStreamer pipeline with fixed buffer sizes for consistency
            pipeline_desc = (
                "alsasrc device=default "
                "blocksize=2048 "  # Fixed block size for consistency
                "latency-time=80000 "  # 80ms latency to match ASR expectations
                "buffer-time=320000 ! "  # 320ms buffer
                "audio/x-raw-int,rate=44100,channels=1,width=16,depth=16 ! "
                "audioconvert ! "
                "audioresample ! "
                "queue "
                "max-size-buffers=10 "  # Smaller queue for lower latency
                "max-size-time=80000000 "  # 80ms buffer in queue
                "leaky=downstream ! "
                "rtpL16pay "
                "mtu=1200 "
                "pt=96 "
                "max-ptime=20 "  # Maximum 20ms per RTP packet
                "min-ptime=16 ! "  # Minimum 16ms per RTP packet
                "udpsink host=%s port=%d "
                "sync=false "
                "async=false"
            ) % (self.target_host, self.target_port)
            
            print("AUDIO: Creating consistent GStreamer pipeline: %s" % pipeline_desc)
            
            self.pipeline = self.gst.parse_launch(pipeline_desc)
            
            # Set up message handling
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self.on_gstreamer_message)
            
            return True
            
        except Exception as e:
            print("ERR: Failed to create GStreamer pipeline: %s" % e)
            return False

    def on_gstreamer_message(self, bus, message):
        """Handle GStreamer pipeline messages"""
        if message.type == self.gst.MESSAGE_ERROR:
            err, debug = message.parse_error()
            print("AUDIO: GStreamer error: %s" % err)
            print("AUDIO: Debug: %s" % debug)
            self.network_errors += 1
            self.stop_gstreamer_pipeline()
        elif message.type == self.gst.MESSAGE_EOS:
            print("AUDIO: End of stream")
            self.stop_gstreamer_pipeline()
        elif message.type == self.gst.MESSAGE_STATE_CHANGED:
            old_state, new_state, pending_state = message.parse_state_changed()
            if message.src == self.pipeline:
                print("AUDIO: Pipeline state changed from %s to %s" % 
                      (old_state.value_name, new_state.value_name))

    def start_gstreamer_pipeline(self):
        """Start the GStreamer audio pipeline"""
        if not self.check_gstreamer_availability():
            print("ERR: Cannot start pipeline - GStreamer not available")
            return False
            
        if not self.pipeline:
            if not self.create_gstreamer_pipeline():
                return False
        
        try:
            print("AUDIO: Starting GStreamer pipeline to %s:%d" % (self.target_host, self.target_port))
            self.pipeline.set_state(self.gst.STATE_PLAYING)
            self.stream_start_time = time.time()
            return True
        except Exception as e:
            print("ERR: Failed to start GStreamer pipeline: %s" % e)
            return False

    def stop_gstreamer_pipeline(self):
        """Stop the GStreamer audio pipeline"""
        if self.pipeline and self.gstreamer_available:
            try:
                self.pipeline.set_state(self.gst.STATE_NULL)
                
                # Print session statistics
                if self.stream_start_time > 0:
                    duration = time.time() - self.stream_start_time
                    print("AUDIO: Streaming session ended:")
                    print("  - Duration: {:.2f} seconds".format(duration))
                    print("  - Target: {}:{}".format(self.target_host, self.target_port))
                    print("  - Network errors: {}".format(self.network_errors))
                
                self.pipeline = None
                self.stream_start_time = 0
                
            except Exception as e:
                print("ERR: Failed to stop GStreamer pipeline: %s" % e)

    def test_network_connection(self):
        """Test if target device is reachable"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(1)
            sock.sendto("test", (self.target_host, self.target_port))
            sock.close()
            return True
        except:
            return False

    def control_streaming(self, _, value):
        """Control audio streaming based on boolean value"""
        print("CONTROL: Audio streaming control signal received - value: {}".format(value))
        if value:
            self.enable_stream()
        else:
            self.disable_stream()

    def on_speaking_event(self, _, is_speaking):
        """Handle speaking state changes to pause/resume audio streaming"""
        is_speaking = bool(is_speaking)
        
        print("AUDIO: Speaking event received - value: {}, current state: {}".format(
            is_speaking, self.isSpeaking))
        
        # Only process if state actually changed
        if self.isSpeaking != is_speaking:
            self.isSpeaking = is_speaking
            global_speaking_state = self.isSpeaking  # Update global speaking state
            
            if self.isSpeaking:
                # Subscribe to end of speech event
                self.memory.subscribeToEvent("ALAnimatedSpeech/EndOfAnimatedSpeech", self.getName(), "stopped_speaking")
                self.led_service.fadeRGB("FaceLeds", 0xFF0000, 0.1)
                
                print("AUDIO: Speaking detected - pausing audio stream")
                # Temporarily pause the pipeline without stopping it completely
                if self.pipeline and self.gstreamer_available and self.isStreamingEnabled:
                    try:
                        # Pause the pipeline instead of stopping it
                        self.pipeline.set_state(self.gst.STATE_PAUSED)
                        print("AUDIO: Pipeline paused successfully")
                    except Exception as e:
                        print("ERR: Failed to pause GStreamer pipeline: %s" % e)
            else:
                print("AUDIO: Speaking ended via direct event - resuming audio stream")
                self.resume_audio_stream()
    
    def resume_audio_stream(self):
        """Resume audio streaming after speaking ends"""
        try:
            # Unsubscribe from the event if subscribed
            try:
                self.memory.unsubscribeToEvent("ALAnimatedSpeech/EndOfAnimatedSpeech", self.getName())
            except:
                pass  # May not be subscribed
            
            self.led_service.fadeRGB("FaceLeds", 0x00FF00, 0.1)
            
            if self.pipeline and self.gstreamer_available and self.isStreamingEnabled:
                try:
                    # Resume the pipeline
                    self.pipeline.set_state(self.gst.STATE_PLAYING)
                    print("AUDIO: Pipeline resumed successfully")
                except Exception as e:
                    print("ERR: Failed to resume GStreamer pipeline: %s" % e)
        except Exception as e:
            print("ERR: Error in resume_audio_stream: %s" % e)

    def stopped_speaking(self, _, value=None):
        """Handle end of animated speech event to resume audio streaming"""
        print("AUDIO: End of animated speech detected - resuming audio stream")
        
        self.isSpeaking = False
        global_speaking_state = self.isSpeaking  # Update global speaking state
        
        self.resume_audio_stream()

    def enable_stream(self):
        """Enable audio streaming using GStreamer"""
        if self.isStreamingEnabled:
            print("CONTROL: Audio streaming already enabled")
            return
            
        if not self.check_gstreamer_availability():
            print("ERR: Cannot enable streaming - GStreamer not available")
            return
        
        # Test network connectivity
        print("AUDIO: Testing network connectivity to {}:{}".format(self.target_host, self.target_port))
        if not self.test_network_connection():
            print("WARN: Network connectivity test failed")
        else:
            print("AUDIO: Network connectivity test successful")
        
        # Start GStreamer pipeline
        if self.start_gstreamer_pipeline():
            self.isStreamingEnabled = True
            self.network_errors = 0
            
            # If currently speaking, start in paused state
            if self.isSpeaking:
                print("AUDIO: Starting in paused state due to active speaking")
                try:
                    self.pipeline.set_state(self.gst.STATE_PAUSED)
                except Exception as e:
                    print("ERR: Failed to pause pipeline on start: %s" % e)
            
            print("CONTROL: Audio streaming ENABLED - {}:{} (GStreamer RTP/L16)".format(
                self.target_host, self.target_port))
        else:
            print("ERR: Failed to enable audio streaming")

    def disable_stream(self):
        """Disable audio streaming"""
        if not self.isStreamingEnabled:
            print("CONTROL: Audio streaming already disabled")
            return
            
        self.isStreamingEnabled = False
        self.stop_gstreamer_pipeline()
        print("CONTROL: Audio streaming DISABLED")

    def set_target(self, host, port):
        """Change target host and port for streaming"""
        was_streaming = self.isStreamingEnabled
        
        if was_streaming:
            self.disable_stream()
        
        self.target_host = host
        self.target_port = port
        print("INF: Audio stream target changed to {}:{}".format(host, port))
        
        if was_streaming:
            self.enable_stream()

    def get_streaming_status(self):
        """Get current streaming status including speaking state"""
        return {
            'streaming': self.isStreamingEnabled,
            'speaking': self.isSpeaking,
            'gstreamer_available': self.check_gstreamer_availability(),
            'target': "{}:{}".format(self.target_host, self.target_port),
            'network_errors': self.network_errors,
            'format': 'RTP/L16' if self.check_gstreamer_availability() else 'unavailable'
        }

    # Legacy methods for compatibility with existing code
    def start(self):
        """Legacy start method - not needed for GStreamer approach"""
        pass
        
    def stop(self):
        """Legacy stop method"""
        self.disable_stream()
        was_streaming = self.isStreamingEnabled
        
        if was_streaming:
            self.disable_stream()
        
        self.target_host = host
        self.target_port = port
        print("INF: Audio stream target changed to {}:{}".format(host, port))
        
        if was_streaming:
            self.enable_stream()

    def get_streaming_status(self):
        """Get current streaming status"""
        return {
            'streaming': self.isStreamingEnabled,
            'gstreamer_available': self.check_gstreamer_availability(),
            'target': "{}:{}".format(self.target_host, self.target_port),
            'network_errors': self.network_errors,
            'format': 'RTP/L16' if self.check_gstreamer_availability() else 'unavailable'
        }

    # Legacy methods for compatibility with existing code
    def start(self):
        """Legacy start method - not needed for GStreamer approach"""
        pass
        
    def stop(self):
        """Legacy stop method"""
        self.disable_stream()
        
    def processRemote(self, nbOfChannels, nbrOfSamplesByChannel, aTimeStamp, buffer):
        """Legacy processRemote method - not needed for GStreamer approach"""
        # GStreamer handles audio capture directly from ALSA
        pass
        try:
            # Convert to float for filtering
            data_float = data.astype(np.float32)
            filtered_data = np.zeros_like(data_float)
            
            # Simple 3-tap FIR filter to reduce aliasing
            # Coefficients for basic low-pass filter
            a0, a1, a2 = 0.25, 0.5, 0.25
            
            for i in range(len(data_float)):
                if i == 0:
                    # Use history for first sample
                    filtered_data[i] = (a0 * self.filter_history[2] + 
                                      a1 * self.filter_history[1] + 
                                      a2 * data_float[i])
                elif i == 1:
                    filtered_data[i] = (a0 * self.filter_history[1] + 
                                      a1 * data_float[i-1] + 
                                      a2 * data_float[i])
                else:
                    filtered_data[i] = (a0 * data_float[i-2] + 
                                      a1 * data_float[i-1] + 
                                      a2 * data_float[i])
            
            # Update filter history for next call
            if len(data_float) >= 2:
                self.filter_history = [data_float[-2], data_float[-1], data_float[-1]]
            elif len(data_float) == 1:
                self.filter_history = [self.filter_history[1], self.filter_history[2], data_float[0]]
            
            return filtered_data.astype(np.int16)
            
        except Exception as e:
            print("WARN: Filter error, using unfiltered data: {}".format(e))
            return data

    def proper_downsample_48k_to_16k(self, audio_48k):
        """Properly downsample from 48kHz to 16kHz with anti-aliasing"""
        try:
            # Apply anti-aliasing filter first
            filtered_data = self.apply_simple_lowpass_filter(audio_48k)
            
            # Downsample by factor of 3 (48000/16000 = 3)
            downsampled = filtered_data[::3]
            
            # Ensure proper int16 range
            downsampled = np.clip(downsampled, -32768, 32767).astype(np.int16)
            
            return downsampled
            
        except Exception as e:
            print("ERR: Downsampling failed: {}".format(e))
            # Fallback to simple decimation if filter fails
            return audio_48k[::3].astype(np.int16)

    def processRemote(self, nbOfChannels, nbrOfSamplesByChannel, aTimeStamp, buffer):
        """Process audio data with RTP streaming and proper timing"""
        if not self.isStreamingEnabled:
            return
            
        try:
            # Convert audio buffer to numpy array
            aSoundDataInterlaced = np.fromstring(str(buffer), dtype=np.int16)
            aSoundData = np.reshape(aSoundDataInterlaced, (nbOfChannels, nbrOfSamplesByChannel), 'F')
            
            # Use front microphone (channel 0)
            frontMicData = aSoundData[0]
            
            # Proper downsampling from 48kHz to 16kHz with anti-aliasing
            downsampledData = self.proper_downsample_48k_to_16k(frontMicData)
            
            # Add to buffer
            self.audioBuffer.extend(downsampledData)
            self.audio_samples_processed += len(downsampledData)
            
            # Send RTP packets based on precise timing
            current_time = time.time()
            time_since_last_packet = current_time - self.last_packet_time
            
            # Send packet if we have enough data AND enough time has passed
            if (len(self.audioBuffer) >= self.bufferSize and 
                time_since_last_packet >= PACKET_INTERVAL):
                
                chunk = self.audioBuffer[:self.bufferSize]
                self.audioBuffer = self.audioBuffer[self.bufferSize:]
                
                self.send_rtp_audio_packet(chunk, current_time)
                self.audio_chunks_processed += 1
                
            elif time_since_last_packet >= (PACKET_INTERVAL * 1.5):
                # We're behind schedule - send partial packet or skip
                if len(self.audioBuffer) >= self.bufferSize // 2:
                    # Send partial packet padded with zeros
                    chunk = self.audioBuffer[:self.bufferSize]
                    if len(chunk) < self.bufferSize:
                        # Pad with zeros to maintain packet size
                        padding = np.zeros(self.bufferSize - len(chunk), dtype=np.int16)
                        chunk = np.concatenate([chunk, padding])
                    
                    self.audioBuffer = self.audioBuffer[len(self.audioBuffer[:self.bufferSize]):]
                    self.send_rtp_audio_packet(chunk, current_time)
                    self.audio_chunks_processed += 1
                else:
                    # Skip this packet interval
                    self.packets_skipped += 1
                    self.last_packet_time = current_time
            
            # Debug logging every 100 chunks (~1.6 seconds)
            if self.audio_chunks_processed % 100 == 0 and self.audio_chunks_processed > 0:
                print("AUDIO: Processed {} chunks, {} samples, buffer: {}, RTP seq: {}, skipped: {}".format(
                    self.audio_chunks_processed, self.audio_samples_processed, 
                    len(self.audioBuffer), self.sequence_number, self.packets_skipped))
                
        except Exception as e:
            print("ERR: AudioStreamModule processRemote error: %s" % str(e))
            traceback.print_exc()

    def send_rtp_audio_packet(self, audio_data, send_time):
        """Send RTP audio packet with proper L16 payload"""
        try:
            # Ensure exactly 256 samples
            audio_array = np.array(audio_data, dtype=np.int16)
            if len(audio_array) != CHUNK_SIZE:
                print("WARN: Audio chunk size mismatch: expected {}, got {}".format(
                    CHUNK_SIZE, len(audio_array)))
                # Pad or truncate to exact size
                if len(audio_array) < CHUNK_SIZE:
                    padding = np.zeros(CHUNK_SIZE - len(audio_array), dtype=np.int16)
                    audio_array = np.concatenate([audio_array, padding])
                else:
                    audio_array = audio_array[:CHUNK_SIZE]
            
            # Validate audio data range
            audio_array = np.clip(audio_array, -32768, 32767)
            
            # Create RTP header
            # Note: RTP timestamp represents the sampling instant of the first sample
            rtp_header = self.create_rtp_header(self.sequence_number, self.rtp_timestamp)
            
            # Convert audio to bytes (network byte order - big-endian for L16)
            # L16 payload format uses big-endian 16-bit samples
            audio_bytes = audio_array.astype('>i2').tostring()  # Big-endian int16
            
            # Verify payload size (256 samples * 2 bytes = 512 bytes)
            if len(audio_bytes) != 512:
                print("ERR: Audio payload size incorrect: {} bytes (expected 512)".format(len(audio_bytes)))
                return
            
            # Create complete RTP packet
            rtp_packet = rtp_header + audio_bytes
            total_packet_size = RTP_HEADER_SIZE + len(audio_bytes)
            
            # Verify total packet size
            if len(rtp_packet) != total_packet_size:
                print("ERR: RTP packet size incorrect: {} bytes (expected {})".format(
                    len(rtp_packet), total_packet_size))
                return
            
            # Send RTP packet
            try:
                bytes_sent = self.sock.sendto(rtp_packet, (self.target_host, self.target_port))
                
                if bytes_sent != len(rtp_packet):
                    print("RTP: WARNING - Partial send: {} of {} bytes".format(
                        bytes_sent, len(rtp_packet)))
                
                # Update RTP session state
                self.last_packet_time = send_time
                self.packets_sent += 1
                self.bytes_sent += bytes_sent
                self.sequence_number = (self.sequence_number + 1) & 0xFFFF  # 16-bit rollover
                self.rtp_timestamp = (self.rtp_timestamp + self.timestamp_increment) & 0xFFFFFFFF  # 32-bit rollover
                
                # Detailed logging every 10 packets
                if self.sequence_number % 10 == 0:
                    rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2)) if len(audio_array) > 0 else 0
                    print("RTP: seq:{} ts:{} -> {}:{} | {}B | RMS:{:.0f} | time:{:.3f}s".format(
                        self.sequence_number, self.rtp_timestamp, self.target_host, self.target_port, 
                        len(rtp_packet), rms, send_time))
                
                # Network stats every 50 packets
                if self.sequence_number % 50 == 0:
                    elapsed = send_time - self.last_network_log_time
                    if elapsed > 0:
                        actual_rate = 50.0 / elapsed
                        expected_rate = 1.0 / PACKET_INTERVAL
                        rate_accuracy = (actual_rate / expected_rate * 100)
                        print("RTP: Rate {:.1f}/{:.1f} pkt/s ({:.1f}%) | SSRC:0x{:08X} | {} errs".format(
                            actual_rate, expected_rate, rate_accuracy, self.ssrc, self.network_errors))
                    self.last_network_log_time = send_time
                
            except socket.error as e:
                self.network_errors += 1
                print("RTP: Socket error #{}: {}".format(self.network_errors, e))
                
                # Recreate socket after multiple errors
                if self.network_errors % 10 == 0:
                    print("RTP: Recreating socket after {} errors".format(self.network_errors))
                    try:
                        self.sock.close()
                        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        print("RTP: Socket recreated successfully")
                    except Exception as recreate_error:
                        print("RTP: Failed to recreate socket: {}".format(recreate_error))
            
            # Volume display every 50 packets
            if self.sequence_number % 50 == 0:
                rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2)) if len(audio_array) > 0 else 0
                volume_bar = '#' * min(20, int(rms // 1000))
                print("Volume: [{}{}] {:.0f} RMS (seq: {})".format(
                    volume_bar, ' ' * (20 - len(volume_bar)), rms, self.sequence_number))
                
        except Exception as e:
            print("ERR: Failed to send RTP audio packet: %s" % str(e))
            traceback.print_exc()

    def set_target(self, host, port):
        """Change target host and port for RTP streaming"""
        self.target_host = host
        self.target_port = port
        print("INF: RTP Audio stream target changed to {}:{}".format(host, port))

    def get_sequence(self):
        """Get current RTP sequence number"""
        return self.sequence_number

    def get_streaming_status(self):
        """Get current RTP streaming status"""
        return {
            'streaming': self.isStreamingEnabled,
            'started': self.isStarted,
            'rtp_sequence': self.sequence_number,
            'rtp_timestamp': self.rtp_timestamp,
            'rtp_ssrc': self.ssrc,
            'target': "{}:{}".format(self.target_host, self.target_port),
            'buffer_size': len(self.audioBuffer),
            'payload_type': 'L16/{}'.format(SAMPLE_RATE)
        }
