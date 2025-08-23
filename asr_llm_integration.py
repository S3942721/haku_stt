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
                'Origin': 'https://localhost:3000',  # Add origin header
                'User-Agent': 'ASR-LLM-Integration/1.0'
            }
            
            # Try to get AWS credentials for WebSocket authentication
            try:
                import boto3
                from botocore.auth import SigV4Auth
                from botocore.awsrequest import AWSRequest
                from urllib.parse import urlparse
                
                # Parse the WebSocket URL
                parsed = urlparse(uri)
                
                # Create a session to get credentials
                session = boto3.Session()
                credentials = session.get_credentials()
                
                if credentials:
                    # Create a signed request for WebSocket upgrade
                    # Note: This is a simplified approach - proper WebSocket signing is more complex
                    print("Found AWS credentials, attempting authenticated connection...")
                
            except Exception as auth_error:
                print(f"Could not set up AWS authentication: {auth_error}")
            
            async with websockets.connect(
                uri, 
                extra_headers=headers,
                timeout=10
            ) as websocket:
                # Send the completion request (matching JavaScript format)
                request_data = {
                    "action": "completion",
                    "history": self.conversation_history,
                    "deepSearch": deep_search,
                    "sessionId": self.session_id
                }
                
                print(f"Sending request to WebSocket...")
                await websocket.send(json.dumps(request_data))
                
                complete_response = ""
                timeout_count = 0
                max_timeout = 30  # 30 second timeout
                
                # Receive response chunks with timeout
                try:
                    while timeout_count < max_timeout:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            timeout_count = 0  # Reset timeout on successful receive
                            
                            data = json.loads(message)
                            print(f"Received WebSocket message: {data}")
                            
                            if data.get("action") == "completion":
                                if "content" in data:
                                    complete_response += data["content"]
                                    print(f"Content chunk: {data['content']}")
                                
                                if data.get("isFinished", False):
                                    print("Response finished")
                                    break
                                    
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
                 retry_on_error: bool = True):
        
        self.llm_client = llm_client
        self.quiet_mode = quiet_mode
        self.retry_on_error = retry_on_error
        self.utterance_queue = queue.Queue()
        self.llm_thread = None
        self.running = False
        
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
    
    def run_streaming_asr(self, device_id: Optional[int] = None, chunk_size_ms: Optional[int] = None):
        """Run streaming ASR with LLM integration"""
        
        # Start LLM worker thread
        self.start_llm_worker()
        
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
  # Using AWS Bedrock directly (recommended - avoids WebSocket auth issues)
  python asr_llm_integration.py --use-bedrock --model-id anthropic.claude-3-sonnet-20240229-v1:0

  # Using WebSocket endpoint (may require proper authentication setup)
  python asr_llm_integration.py --llm-endpoint wss://your-api-gateway.execute-api.region.amazonaws.com/prod

  # Testing mode (echo responses)
  python asr_llm_integration.py --test-mode

  # With custom ASR model and no retries
  python asr_llm_integration.py --use-bedrock --asr-model stt_en_fastconformer_hybrid_large_streaming_80ms --no-retry
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
        help="Audio device ID (will prompt if not provided)"
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
            print("Note: WebSocket may require proper authentication setup in API Gateway")
        
        # Initialize ASR-LLM integration
        integration = ASRLLMIntegration(
            asr_model_name=args.asr_model,
            llm_client=llm_client,
            punct_model_name=args.punct_model,
            lookahead_size=args.lookahead,
            decoder_type=args.decoder,
            quiet_mode=not args.verbose,
            enable_eou=not args.no_eou,
            retry_on_error=not args.no_retry
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
        
        if args.llm_endpoint:
            print("\n" + "="*60)
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
