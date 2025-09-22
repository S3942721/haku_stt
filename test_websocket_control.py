#!/usr/bin/env python3
"""
Simple WebSocket client test script for controlling ASR processing.

This script connects to the ASR WebSocket server and allows you to send
pause, resume, purge, and reset commands interactively.

Usage:
    python test_websocket_control.py [--host HOST] [--port PORT]

Commands:
    pause  - Pause processing and purge current utterance
    resume - Resume processing
    purge  - Purge current utterance and reset ASR
    reset  - Complete system reset
    quit   - Exit the test script
"""

import asyncio
import websockets
import json
import argparse
import sys

class ASRControlClient:
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}"
        self.websocket = None
        self.running = False
        
    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            print(f"Connecting to ASR WebSocket server at {self.uri}...")
            self.websocket = await websockets.connect(self.uri)
            print("Connected successfully!")
            self.running = True
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            print("Disconnected from server")
        self.running = False
    
    async def send_command(self, action):
        """Send a control command to the server"""
        if not self.websocket:
            print("Error: Not connected to server")
            return False
        
        command = {
            "type": "control",
            "action": action
        }
        
        try:
            await self.websocket.send(json.dumps(command))
            print(f"Sent command: {action}")
            
            # Wait for acknowledgment
            response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            response_data = json.loads(response)
            
            if response_data.get("type") == "command_ack":
                print(f"✓ Command acknowledged: {response_data.get('action')} - {response_data.get('status')}")
            elif response_data.get("type") == "error":
                print(f"✗ Error: {response_data.get('error')}")
            else:
                print(f"Server response: {response_data}")
            
            return True
            
        except asyncio.TimeoutError:
            print("✗ Timeout waiting for server response")
            return False
        except Exception as e:
            print(f"✗ Error sending command: {e}")
            return False
    
    async def listen_for_messages(self):
        """Listen for incoming messages from the server"""
        try:
            while self.running and self.websocket:
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    msg_type = data.get("type", "unknown")
                    
                    if msg_type == "partial":
                        print(f"[PARTIAL] {data.get('text', '')}")
                    elif msg_type == "complete":
                        print(f"[COMPLETE] {data.get('text', '')}")
                    elif msg_type == "status":
                        status = data.get('status', '')
                        details = data.get('details', {})
                        print(f"[STATUS] {status}: {details}")
                    elif msg_type in ["command_ack", "error"]:
                        # These are handled in send_command
                        pass
                    else:
                        print(f"[{msg_type.upper()}] {data}")
                        
                except asyncio.TimeoutError:
                    # Normal timeout, continue listening
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed by server")
                    break
                    
        except Exception as e:
            print(f"Error in message listener: {e}")
    
    async def interactive_session(self):
        """Run an interactive command session"""
        print("\n" + "="*50)
        print("ASR WebSocket Control Test Client")
        print("="*50)
        print("Available commands:")
        print("  pause  - Pause processing and purge current utterance")
        print("  resume - Resume processing")
        print("  purge  - Purge current utterance and reset ASR")
        print("  reset  - Complete system reset")
        print("  status - Request current status")
        print("  quit   - Exit the test script")
        print("="*50)
        
        # Start message listener task
        listener_task = asyncio.create_task(self.listen_for_messages())
        
        try:
            while self.running:
                try:
                    # Get user input (non-blocking)
                    command = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: input("\nEnter command: ").strip().lower()
                    )
                    
                    if command == "quit":
                        print("Exiting...")
                        break
                    elif command in ["pause", "resume", "purge", "reset"]:
                        await self.send_command(command)
                    elif command == "status":
                        # Send a status request (this is a custom command for testing)
                        await self.send_command("status")
                    elif command == "help":
                        print("\nAvailable commands: pause, resume, purge, reset, status, quit")
                    elif command == "":
                        continue
                    else:
                        print(f"Unknown command: {command}")
                        print("Type 'help' for available commands")
                        
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    
        finally:
            self.running = False
            listener_task.cancel()
            try:
                await listener_task
            except asyncio.CancelledError:
                pass

async def main():
    parser = argparse.ArgumentParser(
        description="WebSocket client for testing ASR control commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Connect to local server on default port
  python test_websocket_control.py
  
  # Connect to remote server
  python test_websocket_control.py --host 192.168.1.100 --port 8765
  
  # Quick test commands
  python test_websocket_control.py --command pause
  python test_websocket_control.py --command resume
        """
    )
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="WebSocket server host (default: localhost)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket server port (default: 8765)"
    )
    
    parser.add_argument(
        "--command",
        choices=["pause", "resume", "purge", "reset"],
        help="Send a single command and exit"
    )
    
    args = parser.parse_args()
    
    # Create client
    client = ASRControlClient(host=args.host, port=args.port)
    
    try:
        # Connect to server
        if not await client.connect():
            sys.exit(1)
        
        if args.command:
            # Send single command and exit
            print(f"Sending command: {args.command}")
            success = await client.send_command(args.command)
            
            # Wait a moment for any status updates
            await asyncio.sleep(1.0)
            
            if success:
                print("Command sent successfully")
                sys.exit(0)
            else:
                print("Command failed")
                sys.exit(1)
        else:
            # Run interactive session
            await client.interactive_session()
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    try:
        # Check if websockets is available
        import websockets
    except ImportError:
        print("Error: websockets module not found")
        print("Install with: pip install websockets")
        sys.exit(1)
    
    asyncio.run(main())
