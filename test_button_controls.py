#!/usr/bin/env python3
"""
Test script for WebSocket button controls
This script demonstrates how to send button commands to the STT system via WebSocket
"""

import asyncio
import websockets
import json
import time

async def test_button_commands():
    """Test various button commands via WebSocket"""
    uri = "ws://localhost:8765"
    
    try:
        print("Connecting to WebSocket STT server...")
        async with websockets.connect(uri) as websocket:
            print("Connected! Testing button commands...")
            
            # Test mode switching
            print("\n1. Testing mode switch to push-to-talk...")
            mode_switch_cmd = {
                "type": "button",
                "button": "mode_switch",
                "action": "toggle"
            }
            await websocket.send(json.dumps(mode_switch_cmd))
            response = await websocket.recv()
            print(f"Response: {response}")
            
            await asyncio.sleep(2)
            
            # Test push to talk activation
            print("\n2. Testing push to talk activation...")
            ptt_press_cmd = {
                "type": "button",
                "button": "push_to_talk",
                "action": "press"
            }
            await websocket.send(json.dumps(ptt_press_cmd))
            response = await websocket.recv()
            print(f"Response: {response}")
            
            await asyncio.sleep(3)  # Simulate talking
            
            # Test push to talk release
            print("\n3. Testing push to talk release...")
            ptt_release_cmd = {
                "type": "button",
                "button": "push_to_talk",
                "action": "release"
            }
            await websocket.send(json.dumps(ptt_release_cmd))
            response = await websocket.recv()
            print(f"Response: {response}")
            
            await asyncio.sleep(2)
            
            # Test manual EOU
            print("\n4. Testing manual EOU trigger...")
            manual_eou_cmd = {
                "type": "button",
                "button": "manual_eou",
                "action": "press"
            }
            await websocket.send(json.dumps(manual_eou_cmd))
            response = await websocket.recv()
            print(f"Response: {response}")
            
            await asyncio.sleep(2)
            
            # Test keep listening override
            print("\n5. Testing keep listening override...")
            keep_listening_cmd = {
                "type": "button",
                "button": "keep_listening",
                "action": "press"
            }
            await websocket.send(json.dumps(keep_listening_cmd))
            response = await websocket.recv()
            print(f"Response: {response}")
            
            await asyncio.sleep(3)  # Simulate EOU condition while held
            
            # Release keep listening
            print("\n6. Releasing keep listening override...")
            keep_listening_release_cmd = {
                "type": "button",
                "button": "keep_listening",
                "action": "release"
            }
            await websocket.send(json.dumps(keep_listening_release_cmd))
            response = await websocket.recv()
            print(f"Response: {response}")
            
            await asyncio.sleep(2)
            
            # Test not listen (mute)
            print("\n7. Testing not listen (mute)...")
            not_listen_cmd = {
                "type": "button",
                "button": "not_listen",
                "action": "press"
            }
            await websocket.send(json.dumps(not_listen_cmd))
            response = await websocket.recv()
            print(f"Response: {response}")
            
            await asyncio.sleep(2)
            
            # Release not listen
            print("\n8. Releasing not listen...")
            not_listen_release_cmd = {
                "type": "button",
                "button": "not_listen", 
                "action": "release"
            }
            await websocket.send(json.dumps(not_listen_release_cmd))
            response = await websocket.recv()
            print(f"Response: {response}")
            
            # Switch back to standard mode
            print("\n9. Switching back to standard mode...")
            await websocket.send(json.dumps(mode_switch_cmd))
            response = await websocket.recv()
            print(f"Response: {response}")
            
            print("\nButton command testing completed!")
            
    except websockets.exceptions.ConnectionRefused:
        print("ERROR: Could not connect to WebSocket server")
        print("Make sure the STT system is running with WebSocket enabled")
        print("Example: python ws_stt.py --websocket-host localhost")
    except Exception as e:
        print(f"ERROR: {e}")

async def listen_for_messages():
    """Listen for incoming messages from the STT system"""
    uri = "ws://localhost:8765"
    
    try:
        print("Starting message listener...")
        async with websockets.connect(uri) as websocket:
            print("Listening for STT messages (Ctrl+C to stop)...")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "unknown")
                    
                    if msg_type == "partial":
                        print(f"[PARTIAL] {data.get('text', '')}")
                    elif msg_type == "complete":
                        print(f"[COMPLETE] {data.get('text', '')}")
                    elif msg_type == "status":
                        status = data.get('status', 'unknown')
                        details = data.get('details', {})
                        print(f"[STATUS] {status} - {details}")
                    elif msg_type == "button_ack":
                        button = data.get('button', 'unknown')
                        action = data.get('action', 'unknown')
                        print(f"[BUTTON ACK] {button} {action}")
                    else:
                        print(f"[{msg_type.upper()}] {data}")
                        
                except json.JSONDecodeError:
                    print(f"[RAW] {message}")
                    
    except KeyboardInterrupt:
        print("\nMessage listener stopped")
    except websockets.exceptions.ConnectionRefused:
        print("ERROR: Could not connect to WebSocket server")
    except Exception as e:
        print(f"ERROR: {e}")

def main():
    print("WebSocket Button Control Test")
    print("============================")
    print()
    print("Choose test mode:")
    print("1. Test button commands")
    print("2. Listen for STT messages")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(test_button_commands())
    elif choice == "2":
        asyncio.run(listen_for_messages())
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()