"""
ASR to LLM Integration Script
Combines real-time speech recognition with LLM communication.
When a complete utterance is detected, it's sent to the LLM for processing.
"""

import asyncio
import json
import os
import sys
import argparse
import logging
from typing import Optional, Dict, Any, List
import time
import threading
import queue
from datetime import datetime
import re

# GStreamer imports for robot audio
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject, GLib
import signal

# AWS SDK imports for Bedrock
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    print("Warning: boto3 not available. AWS Bedrock integration disabled.")

# HTTP/WebSocket client libraries
import aiohttp
import websockets

# Import the ASR system
from llm_vad_eou_streaming_with_punctuation_quiet import OnlineASRWithPunctuation, AVAILABLE_ASR_MODELS, AVAILABLE_PUNCT_MODELS
import pyaudio as pa
import numpy as np

# Import the lambda client
from lambda_llm_client import LambdaLLMClient, LambdaWebSocketClient

class BedrockLLMClient:
    """Client for communicating with AWS Bedrock directly"""
    
    def __init__(self, model_id: str = None, region: str = "us-east-1"):
        if not AWS_AVAILABLE:
            raise ImportError("boto3 is required for Bedrock integration. Install with: pip install boto3")
        
        self.model_id = model_id or os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
        self.region = region
        self.session_id = None
        self.conversation_history = []
        
        # Initialize Bedrock client
        try:
            self.bedrock_client = boto3.client('bedrock-runtime', region_name=region)
            print(f"Bedrock client initialized for region: {region}")
        except NoCredentialsError:
            print("Error: AWS credentials not found. Please configure AWS credentials.")
            raise
        except Exception as e:
            print(f"Error initializing Bedrock client: {e}")
            raise
    
    def send_message_sync(self, message: str, deep_search: bool = False) -> str:
        """Send message to Bedrock and return the response"""
        if not self.session_id:
            self.session_id = f"asr_session_{int(time.time())}"
        
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        try:
            # Prepare the request for Claude
            system_prompt = """You are a helpful assistant. Respond naturally to the user's speech input. 
Keep your responses conversational and concise unless more detail is specifically requested."""
            
            # Format messages for Claude API
            formatted_messages = []
            for msg in self.conversation_history:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Prepare inference parameters
            inference_config = {
                "temperature": 0.7,
                "topP": 0.9,
                "maxTokens": 1000
            }
            
            # Make request to Bedrock
            response = self.bedrock_client.converse(
                modelId=self.model_id,
                messages=formatted_messages,
                system=[{"text": system_prompt}],
                inferenceConfig=inference_config
            )
            
            # Extract response text
            assistant_message = ""
            if 'output' in response and 'message' in response['output']:
                content = response['output']['message']['content']
                if isinstance(content, list) and len(content) > 0:
                    assistant_message = content[0].get('text', '')
                elif isinstance(content, str):
                    assistant_message = content
            
            # Add assistant response to conversation history
            if assistant_message:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
            
            return assistant_message or "I didn't generate a response."
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            print(f"Bedrock API error [{error_code}]: {error_message}")
            return f"Sorry, I encountered an error: {error_message}"
        except Exception as e:
            print(f"Unexpected Bedrock error: {e}")
            return f"Sorry, I encountered an unexpected error: {e}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.session_id = None

class WebSocketLLMClient:
    """Client for communicating with WebSocket LLM service (like the JavaScript version)"""
    
    def __init__(self, api_endpoint: str, model_id: str = None, region: str = "us-east-1"):
        self.api_endpoint = api_endpoint
        self.model_id = model_id or os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
        self.region = region
        self.session_id = None
        self.conversation_history = []
        
    async def send_message_websocket(self, message: str, deep_search: bool = False) -> str:
        """Send message via WebSocket and return the complete response"""
        if not self.session_id:
            self.session_id = f"asr_session_{int(time.time())}"
        
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        try:
            # Convert HTTP/HTTPS to WebSocket URLs
            uri = self.api_endpoint
            if uri.startswith('https://'):
                uri = uri.replace('https://', 'wss://')
            elif uri.startswith('http://'):
                uri = uri.replace('http://', 'ws://')
            
            print(f"Connecting to WebSocket: {uri}")
            
            # WebSocket connection with headers for authentication
            headers = {
                'Origin': 'https://localhost:3000',
                'User-Agent': 'ASR-LLM-Integration/1.0'
            }
            
            async with websockets.connect(
                uri
            ) as websocket:
                # Send the completion request (exactly matching JavaScript format)
                request_data = {
                    "action": "completion",
                    "history": self.conversation_history.copy(),  # Send full conversation history
                    "deepSearch": deep_search,
                    "sessionId": self.session_id
                }
                
                print(f"Sending WebSocket request:")
                print(json.dumps(request_data, indent=2))
                
                await websocket.send(json.dumps(request_data))
                
                complete_response = ""
                timeout_count = 0
                max_timeout = 30  # 30 second timeout
                
                # Receive response chunks with timeout
                try:
                    while timeout_count < max_timeout:
                        try:
                            message_data = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            timeout_count = 0  # Reset timeout on successful receive
                            
                            data = json.loads(message_data)
                            print(f"Received WebSocket message: {data}")
                            
                            if data.get("action") == "completion":
                                if "content" in data:
                                    content_chunk = data["content"]
                                    complete_response += content_chunk
                                    print(f"Content chunk: {content_chunk}")
                                
                                if data.get("isFinished", False):
                                    print("Response finished")
                                    break
                            elif "error" in data:
                                print(f"WebSocket error from server: {data['error']}")
                                return f"Server error: {data['error']}"
                                    
                        except asyncio.TimeoutError:
                            timeout_count += 1
                            continue
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}")
                            continue
                            
                except Exception as recv_error:
                    print(f"Error receiving WebSocket message: {recv_error}")
                
                # Add assistant response to conversation history
                if complete_response.strip():
                    self.conversation_history.append({
                        "role": "assistant", 
                        "content": complete_response
                    })
                
                return complete_response or "No response received from WebSocket"
                
        except websockets.exceptions.InvalidStatusCode as e:
            error_msg = f"WebSocket connection failed with status {e.status_code}"
            if e.status_code == 403:
                error_msg += "\n\nPossible solutions for 403 error:"
                error_msg += "\n1. Check if your API Gateway allows WebSocket connections"
                error_msg += "\n2. Verify CORS settings in API Gateway"
                error_msg += "\n3. Ensure your AWS credentials have proper permissions"
                error_msg += "\n4. Check if the WebSocket route requires authentication"
                error_msg += "\n5. Try using --use-bedrock instead for direct AWS access"
                error_msg += f"\n\nRequest that failed:"
                error_msg += f"\n{json.dumps({'action': 'completion', 'history': self.conversation_history}, indent=2)}"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"WebSocket error: {e}"
            print(error_msg)
            return error_msg
    
    def send_message_sync(self, message: str, deep_search: bool = False) -> str:
        """Synchronous wrapper for sending messages"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.send_message_websocket(message, deep_search))
        finally:
            loop.close()
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.session_id = None

class SimpleLLMClient:
    """Simple HTTP-based LLM client for testing"""
    
    def __init__(self, api_endpoint: str, model_id: str = None):
        self.api_endpoint = api_endpoint
        self.model_id = model_id
        self.conversation_history = []
    
    def send_message_sync(self, message: str, deep_search: bool = False) -> str:
        """Send message via HTTP and return response"""
        self.conversation_history.append({"role": "user", "content": message})
        
        # For testing, just echo back the message
        response = f"Echo: {message}"
        self.conversation_history.append({"role": "assistant", "content": response})
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

class RobotAudioReceiver:
    """Receives audio stream from Pepper robot via RTP and provides it as audio chunks"""
    
    def __init__(self, listen_port=5004, sample_rate=16000, chunk_size_ms=480):
        self.listen_port = listen_port
        self.sample_rate = sample_rate
        self.chunk_size_ms = chunk_size_ms
        self.pipeline = None
        self.audio_queue = queue.Queue(maxsize=100)  # Buffer for audio chunks
        self.running = False
        self.volume_level = 0.0
        
        # Audio buffering for proper chunk sizes
        self.samples_per_chunk = int(sample_rate * chunk_size_ms / 1000)
        self.audio_buffer = np.array([], dtype=np.int16)
        self.buffer_lock = threading.Lock()
        
        # Debug counters
        self.total_samples_received = 0
        self.total_buffers_received = 0
        self.buffer_overruns = 0
        self.last_sample_time = 0
        
        # Initialize GStreamer
        Gst.init(None)
        
    def create_pipeline(self):
        """Create GStreamer pipeline for receiving audio from robot"""
        # Pipeline that receives RTP audio and converts it to 16kHz mono for ASR
        pipeline_desc = (
            f"udpsrc port={self.listen_port} "
            "caps=\"application/x-rtp,media=(string)audio,clock-rate=(int)44100,encoding-name=(string)L16,payload=(int)96,channels=(int)1\" ! "
            "rtpjitterbuffer "
            "latency=200 "  # 200ms jitter buffer
            "drop-on-latency=true "
            "max-dropout-time=1000 "
            "max-misorder-time=100 ! "
            "rtpL16depay ! "
            "audioconvert ! "
            f"audioresample ! audio/x-raw,rate={self.sample_rate},channels=1,format=S16LE ! "
            "level name=volumelevel interval=50000000 ! "  # 50ms intervals for volume monitoring
            "appsink name=appsink emit-signals=true sync=false "
            "max-buffers=50 drop=false"  # Increased buffer size, don't drop
        )
        
        print(f"Creating robot audio pipeline: {pipeline_desc}")
        
        try:
            self.pipeline = Gst.parse_launch(pipeline_desc)
        except Exception as e:
            print(f"Error creating pipeline: {e}")
            return False
            
        # Set up message handling
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)
        bus.connect("message::element", self.on_level_message)
        
        # Set up app sink callback
        appsink = self.pipeline.get_by_name("appsink")
        appsink.connect("new-sample", self.on_new_sample)
        
        return True
    
    def on_new_sample(self, appsink):
        """Handle new audio sample from GStreamer"""
        sample = appsink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                # Convert to numpy array
                audio_data = np.frombuffer(map_info.data, dtype=np.int16)
                
                # Update debug counters
                self.total_buffers_received += 1
                self.total_samples_received += len(audio_data)
                self.last_sample_time = time.time()
                
                with self.buffer_lock:
                    # Append to internal buffer
                    self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])
                    
                    # Extract chunks of the correct size
                    while len(self.audio_buffer) >= self.samples_per_chunk:
                        chunk = self.audio_buffer[:self.samples_per_chunk]
                        self.audio_buffer = self.audio_buffer[self.samples_per_chunk:]
                        
                        # Add chunk to queue (non-blocking, drop if full)
                        try:
                            self.audio_queue.put_nowait(chunk)
                        except queue.Full:
                            # Drop oldest sample and add new one
                            self.buffer_overruns += 1
                            try:
                                self.audio_queue.get_nowait()
                                self.audio_queue.put_nowait(chunk)
                            except queue.Empty:
                                pass
                
                buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK
    
    def on_level_message(self, bus, message):
        """Handle volume level messages from the level element"""
        if message.type == Gst.MessageType.ELEMENT:
            structure = message.get_structure()
            if structure and structure.get_name() == "level":
                # Get RMS values (volume levels)
                rms = structure.get_value("rms")
                if rms and len(rms) > 0:
                    # Convert from dB to linear scale (0-1) with better scaling
                    db_level = rms[0]
                    if db_level != float('-inf'):
                        # Improved dB to linear conversion with wider dynamic range
                        # Typical speech is around -20dB to -10dB, loud speech around -6dB
                        # Adjust the range to make better use of 0-100% scale
                        linear_level = min(1.0, max(0.0, (db_level + 40) / 30))  # Changed from +60/60 to +40/30
                        self.volume_level = linear_level
    
    def on_message(self, bus, message):
        """Handle GStreamer messages"""
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"\nRobot audio error: {err}")
            print(f"Debug: {debug}")
            self.stop()
        elif message.type == Gst.MessageType.EOS:
            print("\nRobot audio stream ended")
            self.stop()
        elif message.type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                print(f"\nRobot audio pipeline state: {old_state.value_nick} -> {new_state.value_nick}")
        elif message.type == Gst.MessageType.STREAM_START:
            print("\nRobot audio stream started! 🎵")
            print("Now receiving audio data from robot...")
        elif message.type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"\nRobot audio warning: {warn}")
    
    def start(self):
        """Start receiving audio from robot"""
        if not self.create_pipeline():
            return False
            
        print(f"Starting robot audio receiver on port {self.listen_port}")
        print(f"Expected chunk size: {self.samples_per_chunk} samples ({self.chunk_size_ms}ms)")
        print("Waiting for audio stream from Pepper robot...")
        
        # Start the pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Failed to start robot audio pipeline")
            return False
        
        self.running = True
        return True
    
    def stop(self):
        """Stop receiving audio"""
        self.running = False
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
    
    def get_audio_chunk(self, timeout=1.0):
        """Get next audio chunk (blocking with timeout)"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def is_running(self):
        """Check if receiver is running"""
        return self.running and self.pipeline is not None
    
    def get_debug_stats(self):
        """Get debugging statistics"""
        return {
            'total_buffers': self.total_buffers_received,
            'total_samples': self.total_samples_received,
            'buffer_overruns': self.buffer_overruns,
            'queue_size': self.audio_queue.qsize(),
            'queue_max': self.audio_queue.maxsize,
            'buffer_length': len(self.audio_buffer) if hasattr(self, 'audio_buffer') else 0,
            'last_sample_time': self.last_sample_time,
            'volume_level': self.volume_level
        }

class ASRLLMIntegration:
    """Main class that integrates ASR with LLM communication"""
    
    def __init__(self, 
                 asr_model_name: str,
                 llm_client,
                 punct_model_name: str = None,
                 lookahead_size: int = 480,
                 decoder_type: str = "rnnt",
                 quiet_mode: bool = True,
                 enable_eou: bool = True,
                 retry_on_error: bool = True,
                 robot_audio_port: Optional[int] = None,
                 robot_chunk_size_ms: Optional[int] = None):
        
        self.llm_client = llm_client
        self.quiet_mode = quiet_mode
        self.retry_on_error = retry_on_error
        self.robot_audio_port = robot_audio_port
        self.robot_chunk_size_ms = robot_chunk_size_ms or lookahead_size
        self.utterance_queue = queue.Queue()
        self.llm_thread = None
        self.running = False
        self.robot_receiver = None
        
        # Initialize ASR system
        print("Initializing ASR system...")
        self.asr_system = OnlineASRWithPunctuation(
            asr_model_name=asr_model_name,
            punct_model_name=punct_model_name,
            lookahead_size=lookahead_size,
            decoder_type=decoder_type,
            quiet_mode=quiet_mode,
            enable_eou=enable_eou
        )
        
        print("ASR-LLM Integration ready!")
    
    def _llm_worker(self):
        """Worker thread that processes utterances and sends them to LLM"""
        while self.running:
            try:
                # Get utterance from queue (with timeout to allow checking self.running)
                utterance = self.utterance_queue.get(timeout=1.0)
                
                if utterance is None:  # Shutdown signal
                    break
                
                print(f"\n{'='*60}")
                print(f"[USER] {utterance}")
                print(f"{'='*60}")
                print("[LLM] Processing...")
                
                # Send to LLM with retry logic
                response = None
                retry_count = 0
                max_retries = 2 if self.retry_on_error else 0
                
                while retry_count <= max_retries and response is None:
                    try:
                        response = self.llm_client.send_message_sync(utterance, deep_search=False)
                        
                        # Check if we got an error response
                        if "WebSocket connection failed" in response or "server rejected" in response:
                            if retry_count < max_retries:
                                print(f"[LLM] WebSocket error, retrying... (attempt {retry_count + 1}/{max_retries + 1})")
                                time.sleep(2)  # Wait before retry
                                response = None
                                retry_count += 1
                                continue
                            else:
                                print(f"[LLM] WebSocket failed after {max_retries + 1} attempts")
                                response = "Sorry, I'm having trouble connecting to the LLM service. Please check the WebSocket endpoint configuration."
                        
                    except Exception as e:
                        if retry_count < max_retries:
                            print(f"[LLM] Error occurred, retrying... (attempt {retry_count + 1}/{max_retries + 1}): {e}")
                            time.sleep(2)
                            retry_count += 1
                            continue
                        else:
                            print(f"[LLM] Failed after {max_retries + 1} attempts: {e}")
                            response = f"Sorry, I encountered an error: {e}"
                            break
                
                # Clean up response (remove thinking tags, etc.)
                cleaned_response = self._clean_llm_response(response)
                
                print(f"\n[LLM] {cleaned_response}")
                print(f"{'='*60}")
                print("Listening for next utterance...")
                
                self.utterance_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"LLM worker error: {e}")
                continue
    
    def _clean_llm_response(self, response: str) -> str:
        """Clean up LLM response by removing thinking tags and formatting"""
        # Remove <Thinking>...</Thinking> blocks
        response = re.sub(r'<Thinking>.*?</Thinking>', '', response, flags=re.DOTALL)
        
        # Extract content from <Response>...</Response> tags
        response_match = re.search(r'<Response>(.*?)</Response>', response, flags=re.DOTALL)
        if response_match:
            response = response_match.group(1)
        
        # Remove tool use tags
        response = re.sub(r'<ToolUse[^>]*>', '', response)
        response = re.sub(r'</ToolUse>', '', response)
        
        # Clean up whitespace
        response = response.strip()
        
        return response
    
    def start_llm_worker(self):
        """Start the LLM worker thread"""
        self.running = True
        self.llm_thread = threading.Thread(target=self._llm_worker, daemon=True)
        self.llm_thread.start()
    
    def stop_llm_worker(self):
        """Stop the LLM worker thread"""
        self.running = False
        self.utterance_queue.put(None)  # Shutdown signal
        if self.llm_thread:
            self.llm_thread.join()
    
    def process_utterance(self, utterance: str):
        """Add utterance to queue for LLM processing"""
        if utterance.strip():
            self.utterance_queue.put(utterance.strip())
    
    def run_streaming_asr_robot(self):
        """Run streaming ASR with robot audio input"""
        if not self.robot_audio_port:
            print("Error: Robot audio port not specified")
            return
        
        # Start LLM worker thread
        self.start_llm_worker()
        
        # Initialize robot audio receiver with appropriate chunk size
        self.robot_receiver = RobotAudioReceiver(
            listen_port=self.robot_audio_port,
            sample_rate=16000,
            chunk_size_ms=self.robot_chunk_size_ms
        )
        
        if not self.robot_receiver.start():
            print("Failed to start robot audio receiver")
            self.stop_llm_worker()
            return
        
        print(f"Receiving audio from robot on port {self.robot_audio_port}")
        print(f"Using chunk size: {self.robot_chunk_size_ms}ms ({self.robot_receiver.samples_per_chunk} samples)")
        print("Processing speech from robot... (Press Ctrl+C to stop)")
        print("\nAudio Reception Status:")
        print("=" * 60)
        
        # Store last transcription to avoid repetition
        last_utterance = ""
        last_volume_display = time.time()
        last_debug_display = time.time()
        
        # Audio monitoring variables
        total_chunks_received = 0
        total_samples_received = 0
        max_volume_seen = 0.0
        last_audio_time = time.time()
        
        try:
            while self.robot_receiver.is_running():
                # Get audio chunk from robot
                audio_chunk = self.robot_receiver.get_audio_chunk(timeout=0.1)
                
                if audio_chunk is not None:
                    total_chunks_received += 1
                    total_samples_received += len(audio_chunk)
                    last_audio_time = time.time()
                    
                    # Calculate chunk volume for debugging with better scaling
                    chunk_rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
                    # Use logarithmic scaling for better dynamic range
                    # Scale based on typical speech levels (adjust these values as needed)
                    chunk_volume_normalized = min(1.0, chunk_rms / 5000.0)  # Increased from 1000 to 5000
                    max_volume_seen = max(max_volume_seen, chunk_volume_normalized)
                    
                    # Verify chunk size before processing
                    if len(audio_chunk) < 100:  # Very small chunk, skip
                        print(f"[DEBUG] Skipping small chunk: {len(audio_chunk)} samples")
                        continue
                    
                    try:
                        # Process with ASR
                        raw_text, punct_text, is_eou, complete_utterance = self.asr_system.transcribe_chunk(audio_chunk)
                        
                        # Handle complete utterance
                        if is_eou and complete_utterance and complete_utterance != last_utterance:
                            print(f"\n{'='*60}")
                            print(f"COMPLETE UTTERANCE DETECTED:")
                            print(f"{'='*60}")
                            print(f"{complete_utterance}")
                            print(f"{'='*60}")
                            
                            # Send to LLM
                            self.process_utterance(complete_utterance)
                            last_utterance = complete_utterance
                        
                        # Show incremental updates if not in quiet mode
                        elif not self.quiet_mode and punct_text.strip():
                            print(f"\r[Robot Audio] {punct_text}", end='', flush=True)
                            
                    except Exception as e:
                        print(f"\n[ASR ERROR] Failed to process audio chunk (size: {len(audio_chunk)}): {e}")
                        continue
                    
                    # Show real-time volume and status
                    current_time = time.time()
                    if current_time - last_volume_display > 0.2:  # Update every 200ms
                        # Create volume bars for both GStreamer level and chunk level
                        # Use adjusted GStreamer volume scaling
                        gst_volume_adjusted = min(1.0, self.robot_receiver.volume_level * 2.0)  # Scale up GST volume
                        gst_volume_percent = int(gst_volume_adjusted * 100)
                        chunk_volume_percent = int(chunk_volume_normalized * 100)
                        
                        # Volume bars
                        bar_length = 30
                        
                        # GStreamer volume bar
                        gst_filled = int(bar_length * gst_volume_adjusted)
                        gst_bar = '█' * gst_filled + '░' * (bar_length - gst_filled)
                        
                        # Chunk volume bar
                        chunk_filled = int(bar_length * chunk_volume_normalized)
                        chunk_bar = '█' * chunk_filled + '░' * (bar_length - chunk_filled)
                        
                        # More nuanced color coding based on volume level
                        max_vol = max(gst_volume_adjusted, chunk_volume_normalized)
                        if max_vol > 0.9:
                            color = '\033[95m'  # Magenta - Very loud
                        elif max_vol > 0.7:
                            color = '\033[92m'  # Green - Good volume
                        elif max_vol > 0.4:
                            color = '\033[93m'  # Yellow - Medium volume
                        elif max_vol > 0.1:
                            color = '\033[94m'  # Blue - Low volume
                        elif max_vol > 0.02:
                            color = '\033[96m'  # Cyan - Very low volume
                        else:
                            color = '\033[91m'  # Red - No/minimal volume
                        
                        reset_color = '\033[0m'
                        
                        # Show raw RMS value for debugging
                        # Clear line and show volume status
                        print(f"\r{' ' * 140}", end='')  # Clear line
                        print(f"\r{color}[GST: {gst_bar} {gst_volume_percent:3d}%] [CHK: {chunk_bar} {chunk_volume_percent:3d}%]{reset_color} | RMS: {chunk_rms:6.0f} | Chunks: {total_chunks_received} | Samples: {total_samples_received}", end='', flush=True)
                        
                        last_volume_display = current_time
                    
                    # Show detailed debug info every 5 seconds
                    if current_time - last_debug_display > 5.0:
                        print(f"\n{'-' * 80}")
                        print(f"AUDIO DEBUG INFO:")
                        print(f"  Total chunks received: {total_chunks_received}")
                        print(f"  Total samples received: {total_samples_received}")
                        print(f"  Average chunk size: {total_samples_received / max(1, total_chunks_received):.1f} samples")
                        print(f"  Max volume seen: {max_volume_seen * 100:.1f}%")
                        print(f"  Current chunk RMS: {chunk_rms:.0f}")
                        print(f"  Current GStreamer volume: {self.robot_receiver.volume_level * 100:.1f}%")
                        print(f"  Audio buffer status: {self.robot_receiver.audio_queue.qsize()}/{self.robot_receiver.audio_queue.maxsize} chunks")
                        print(f"  Pipeline running: {self.robot_receiver.is_running()}")
                        print(f"  Buffer overruns: {self.robot_receiver.buffer_overruns}")
                        print(f"{'-' * 80}")
                        last_debug_display = current_time
                
                else:
                    # No audio received - check if we've been waiting too long
                    current_time = time.time()
                    if current_time - last_audio_time > 3.0:  # No audio for 3 seconds
                        if current_time - last_volume_display > 1.0:
                            print(f"\r\033[91m[NO AUDIO] Waiting for audio stream... (last received {current_time - last_audio_time:.1f}s ago)\033[0m", end='', flush=True)
                            last_volume_display = current_time
                
                # Check for interruption
                if not self.running:
                    break
                    
        except KeyboardInterrupt:
            print('\n\nStopping robot audio processing...')
        finally:
            # Show final statistics
            print(f"\n{'='*60}")
            print(f"FINAL AUDIO STATISTICS:")
            print(f"  Total chunks processed: {total_chunks_received}")
            print(f"  Total samples processed: {total_samples_received}")
            print(f"  Max volume detected: {max_volume_seen * 100:.1f}%")
            print(f"  Max volume RMS value: {max_volume_seen * 5000:.0f}")
            if total_chunks_received > 0:
                print(f"  Average chunk size: {total_samples_received / total_chunks_received:.1f} samples")
                print(f"  Total audio duration: {total_samples_received / 16000:.2f} seconds")
                print(f"  Buffer overruns: {self.robot_receiver.buffer_overruns}")
            print(f"{'='*60}")
            
            if self.robot_receiver:
                self.robot_receiver.stop()
            self.stop_llm_worker()
            print("Robot audio processing stopped")

    def run_streaming_asr(self, device_id: Optional[int] = None, chunk_size_ms: Optional[int] = None):
        """Run streaming ASR with either microphone or robot audio input"""
        
        # If robot audio port is specified, use robot audio instead of microphone
        if self.robot_audio_port:
            return self.run_streaming_asr_robot()
        
        # Calculate chunk size
        if chunk_size_ms is None:
            chunk_size_ms = self.asr_system.lookahead_size + 80  # ENCODER_STEP_LENGTH
        
        print(f"Using chunk size: {chunk_size_ms}ms")
        
        # Initialize PyAudio
        p = pa.PyAudio()
        
        try:
            # Select audio device
            input_devices = []
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                if dev.get('maxInputChannels'):
                    input_devices.append(i)
            
            if not input_devices:
                print('ERROR: No audio input device found.')
                return
            
            if device_id is None:
                print('Available audio input devices:')
                for i in input_devices:
                    dev = p.get_device_info_by_index(i)
                    print(f"  {i}: {dev.get('name')} (channels: {dev.get('maxInputChannels')})")
                
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
            
            # Calculate frames per buffer
            frames_per_buffer = int(16000 * chunk_size_ms / 1000) - 1
            
            # Store last transcription to avoid repetition
            last_utterance = ""
            
            # Define callback function
            def stream_callback(in_data, frame_count, time_info, status):
                nonlocal last_utterance
                
                if status:
                    print(f"Stream status: {status}")
                
                # Convert audio data and transcribe
                signal = np.frombuffer(in_data, dtype=np.int16)
                raw_text, punct_text, is_eou, complete_utterance = self.asr_system.transcribe_chunk(signal)
                
                # Handle complete utterance
                if is_eou and complete_utterance and complete_utterance != last_utterance:
                    print(f"\n{'='*60}")
                    print(f"COMPLETE UTTERANCE DETECTED:")
                    print(f"{'='*60}")
                    print(f"{complete_utterance}")
                    print(f"{'='*60}")
                    
                    # Send to LLM
                    self.process_utterance(complete_utterance)
                    last_utterance = complete_utterance
                
                # Show incremental updates if not in quiet mode
                elif not self.quiet_mode and punct_text.strip():
                    print(f"\r[Listening] {punct_text}", end='', flush=True)
                
                return (in_data, pa.paContinue)
            
            # Open audio stream
            stream = p.open(
                format=pa.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=device_id,
                stream_callback=stream_callback,
                frames_per_buffer=frames_per_buffer
            )
            
            print('\nListening for speech... (Press Ctrl+C to stop)')
            print('When you finish speaking, the utterance will be sent to the LLM.')
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
                print("Audio stream stopped")
        
        finally:
            p.terminate()
            self.stop_llm_worker()

def main():
    parser = argparse.ArgumentParser(
        description="ASR to LLM Integration - Real-time speech recognition with LLM communication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using AWS Bedrock directly (recommended)
  python asr_llm_integration.py --use-bedrock --model-id anthropic.claude-3-sonnet-20240229-v1:0

  # Using Lambda functionality directly (includes Knowledge Base)
  python asr_llm_integration.py --use-lambda --model-id anthropic.claude-3-sonnet-20240229-v1:0

  # Using WebSocket endpoint
  python asr_llm_integration.py --llm-endpoint wss://your-api-gateway.execute-api.region.amazonaws.com/prod

  # Using robot audio with AWS Bedrock
  python audio_direct_from_robot.py --use-bedrock --robot-port 5004

  # Using microphone with WebSocket endpoint
  python audio_direct_from_robot.py --llm-endpoint wss://your-api-gateway.execute-api.region.amazonaws.com/prod

  # Testing mode (echo responses)
  python asr_llm_integration.py --test-mode
  python audio_direct_from_robot.py --test-mode --robot-port 5004
        """
    )
    
    # LLM Configuration
    llm_group = parser.add_mutually_exclusive_group(required=True)
    llm_group.add_argument(
        "--llm-endpoint",
        help="LLM API endpoint (WebSocket WS/WSS URL)"
    )
    llm_group.add_argument(
        "--use-bedrock",
        action="store_true",
        help="Use AWS Bedrock directly (requires AWS credentials)"
    )
    llm_group.add_argument(
        "--use-lambda",
        action="store_true",
        help="Use Lambda functionality directly (includes Knowledge Base access)"
    )
    llm_group.add_argument(
        "--test-mode",
        action="store_true",
        help="Use test mode (echo responses)"
    )
    
    parser.add_argument(
        "--model-id",
        default=None,
        help="LLM model ID (defaults to environment variable BEDROCK_MODEL_ID)"
    )
    
    parser.add_argument(
        "--aws-region",
        default="us-east-1",
        help="AWS region for Bedrock (default: us-east-1)"
    )
    
    parser.add_argument(
        "--deep-search",
        action="store_true",
        help="Enable deep search mode for LLM"
    )
    
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Disable retry on LLM communication errors"
    )
    
    # ASR Configuration
    parser.add_argument(
        "--asr-model",
        default="stt_en_fastconformer_hybrid_large_streaming_multi",
        choices=AVAILABLE_ASR_MODELS,
        help="ASR model name to use"
    )
    
    parser.add_argument(
        "--punct-model",
        default=None,
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
    
    # Audio Configuration  
    parser.add_argument(
        "--device",
        type=int,
        help="Audio device ID (will prompt if not provided, ignored if --robot-port is used)"
    )
    
    parser.add_argument(
        "--robot-port",
        type=int,
        help="Port to receive robot audio stream (if specified, microphone input is disabled)"
    )
    
    parser.add_argument(
        "--robot-chunk-size",
        type=int,
        help="Robot audio chunk size in milliseconds (defaults to lookahead size)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Chunk size in milliseconds (auto-calculated if not provided)"
    )
    
    # EOU Configuration
    parser.add_argument(
        "--no-eou",
        action="store_true",
        help="Disable end-of-utterance detection"
    )
    
    parser.add_argument(
        "--vad-speech-threshold",
        type=float,
        default=0.5,
        help="VAD probability threshold for speech detection (0.0-1.0)"
    )
    
    parser.add_argument(
        "--vad-speech-proportion-threshold",
        type=float,
        default=0.2,
        help="Speech proportion threshold for EOU detection (0.0-1.0, lower = more sensitive)"
    )
    
    parser.add_argument(
        "--vad-analysis-window",
        type=float,
        default=2.0,
        help="Analysis window in seconds for speech proportion calculation"
    )
    
    # Utility
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (show incremental transcriptions)"
    )
    
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    
    args = parser.parse_args()
    
    # List devices and exit if requested
    if args.list_devices:
        p = pa.PyAudio()
        print('Available audio input devices:')
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get('maxInputChannels'):
                print(f"  {i}: {dev.get('name')} (channels: {dev.get('maxInputChannels')})")
        p.terminate()
        return
    
    try:
        # Initialize LLM client based on selected mode
        print("Initializing LLM client...")
        
        if args.use_bedrock:
            if not AWS_AVAILABLE:
                print("Error: boto3 not installed. Install with: pip install boto3")
                return
            llm_client = BedrockLLMClient(
                model_id=args.model_id,
                region=args.aws_region
            )
            print("Using AWS Bedrock for LLM communication")
        elif args.use_lambda:
            if not AWS_AVAILABLE:
                print("Error: boto3 not installed. Install with: pip install boto3")
                return
            llm_client = LambdaWebSocketClient(
                region=args.aws_region,
                model_id=args.model_id
            )
            print("Using Lambda functionality (includes Knowledge Base access)")
            print("Note: Requires KB_ID environment variable for Knowledge Base access")
        elif args.test_mode:
            llm_client = SimpleLLMClient("http://test")
            print("Using test mode (echo responses)")
        else:
            llm_client = WebSocketLLMClient(
                api_endpoint=args.llm_endpoint,
                model_id=args.model_id,
                region=args.aws_region
            )
            print(f"Using WebSocket endpoint: {args.llm_endpoint}")
            print("Note: WebSocket requests will be sent in the format:")
            print('{"action": "completion", "history": [{"role": "user", "content": "message"}]}')
        
        # Initialize ASR-LLM integration
        integration = ASRLLMIntegration(
            asr_model_name=args.asr_model,
            llm_client=llm_client,
            punct_model_name=args.punct_model,
            lookahead_size=args.lookahead,
            decoder_type=args.decoder,
            quiet_mode=not args.verbose,
            enable_eou=not args.no_eou,
            retry_on_error=not args.no_retry,
            robot_audio_port=args.robot_port,
            robot_chunk_size_ms=args.robot_chunk_size
        )
        
        # Configure VAD parameters
        if integration.asr_system.frame_detector:
            integration.asr_system.frame_detector.speech_probability_threshold = args.vad_speech_threshold
            integration.asr_system.frame_detector.speech_proportion_threshold = args.vad_speech_proportion_threshold
            integration.asr_system.frame_detector.analysis_window_seconds = args.vad_analysis_window
        
        print("\nASR-LLM Integration ready!")
        print(f"ASR Model: {args.asr_model}")
        print(f"Punctuation Model: {args.punct_model or 'None'}")
        print(f"EOU Detection: {'Enabled' if not args.no_eou else 'Disabled'}")
        print(f"Retry on Error: {'Enabled' if not args.no_retry else 'Disabled'}")
        
        if args.use_lambda:
            print("\nLambda functionality includes:")
            print("- Direct Bedrock access with streaming")
            print("- Knowledge Base search integration")
            print("- Tool calling support")
            print("- RMIT University research assistant prompts")
        
        if args.llm_endpoint:
            print("\n" + "="*60)
            print("WEBSOCKET REQUEST FORMAT:")
            print("="*60)
            print("Each request will be sent as:")
            print(json.dumps({
                "action": "completion",
                "history": [
                    {"role": "user", "content": "your speech input"}
                ]
            }, indent=2))
            print("="*60 + "\n")
            
            print("WEBSOCKET TROUBLESHOOTING TIPS:")
            print("="*60)
            print("If you get 403 errors:")
            print("1. Check API Gateway WebSocket route configuration")
            print("2. Verify CORS settings allow WebSocket connections")
            print("3. Ensure proper authentication is set up")
            print("4. Try using --use-bedrock instead for direct access")
            print("="*60 + "\n")
        
        # Run the integration
        integration.run_streaming_asr(
            device_id=args.device,
            chunk_size_ms=args.chunk_size
        )
        
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
