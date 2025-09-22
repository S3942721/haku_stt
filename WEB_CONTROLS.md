# Web-Based Button Controls for WebSocket STT System

This document describes the web-based button control system for the WebSocket Speech-to-Text system.

## Overview

The button control functionality is now provided through a web interface (`web_controls.html`) that connects to the STT system via WebSocket. This provides:

- **Cross-platform compatibility**: Works on any device with a web browser
- **No dependencies**: No need to install keyboard libraries or handle permissions
- **Remote control**: Can control STT system from any device on the network
- **Visual feedback**: Real-time display of button states and STT output
- **Touch support**: Works on mobile devices and tablets

## Components

### 1. Main STT System (`ws_stt.py`)
- Handles speech-to-text processing
- Manages WebSocket server for receiving commands
- Processes button commands received via WebSocket
- **No keyboard input handling**

### 2. Web Controls (`web_controls.html`)
- HTML/JavaScript web interface
- Connects to STT system via WebSocket
- Provides buttons for all control functions
- Displays real-time STT output
- Supports both mouse/touch and keyboard shortcuts

## Usage

### Starting the STT System

```bash
# Start the main STT system with WebSocket server
python3 ws_stt.py --websocket-host 0.0.0.0 --websocket-port 8765
```

### Using Web Controls

1. Open `web_controls.html` in any web browser
2. Enter the WebSocket URI (e.g., `ws://localhost:8765`)
3. Click "Connect"
4. Use the button controls

### Remote Access

To use controls from another device:

1. Start STT system with `--websocket-host 0.0.0.0`
2. Open `web_controls.html` on remote device
3. Enter `ws://[STT_SERVER_IP]:8765` as WebSocket URI
4. Connect and control remotely

## Button Controls

The web interface provides these control options:

### Mouse/Touch Controls

| Button | Description |
|--------|-------------|
| **Manual EOU** | Click to trigger manual end-of-utterance |
| **Keep Listening** | Hold to override EOU detection |
| **Mute** | Hold to stop processing audio |
| **Push-to-Talk** | Hold to activate microphone (in PTT mode) |
| **Mode Switch** | Toggle between standard/push-to-talk modes |

### Keyboard Shortcuts (optional)

When the web page has focus:

| Key | Button Command | Description |
|-----|----------------|-------------|
| **Space** | `manual_eou` | Trigger manual end-of-utterance |
| **Ctrl** | `keep_listening` | Override EOU detection, keep listening |
| **Shift** | `not_listen` | Mute audio input |
| **Alt** | `push_to_talk` | Push-to-talk button (in PTT mode) |
| **M** | `mode_switch` | Switch between standard/push-to-talk modes |

### Button Actions

Each button (except `mode_switch`) supports:
- **Press**: Button pressed down (mouse down, key down)
- **Release**: Button released (mouse up, key up)
- **Toggle**: Switch state (for mode switch)

## WebSocket Protocol

### Button Commands

```json
{
  "type": "button",
  "button": "manual_eou",
  "action": "press"
}
```

### Button Acknowledgments

```json
{
  "type": "button_ack",
  "button": "manual_eou", 
  "action": "press",
  "status": "queued",
  "timestamp": 1234567890.123
}
```

### STT Output Messages

The web interface receives and displays STT output:

```json
{
  "type": "partial",
  "text": "hello world",
  "confidence": 0.95,
  "timestamp": 1234567890.123
}
```

```json
{
  "type": "complete", 
  "text": "Hello world.",
  "confidence": 0.98,
  "timestamp": 1234567890.123
}
```

## Features

### Real-time Display
- Live STT output with partial and complete transcriptions
- Connection status indicator
- Current mode display
- Button state visual feedback

### Cross-Platform
- Works on desktop browsers (Chrome, Firefox, Safari, Edge)
- Mobile browser support with touch controls
- No installation or dependencies required

### Network Control
- Can connect to remote STT servers
- Multiple clients can connect simultaneously
- WebSocket reconnection handling

## Dependencies

### STT System (`ws_stt.py`)
- Requires `websockets` for WebSocket server
- No keyboard-specific dependencies

### Web Controls (`web_controls.html`)
- Any modern web browser
- JavaScript enabled
- No additional installations required

## Testing

### Test with Local STT System

1. Start STT system:
```bash
python3 ws_stt.py --websocket-host localhost
```

2. Open `web_controls.html` in browser
3. Connect to `ws://localhost:8765`
4. Test button controls and observe STT output

### Test with Remote STT System

1. Start STT system on server:
```bash
python3 ws_stt.py --websocket-host 0.0.0.0
```

2. Open `web_controls.html` on client device
3. Connect to `ws://[SERVER_IP]:8765`
4. Control STT system remotely

## Troubleshooting

### Connection Issues
- Ensure STT system is running with WebSocket enabled
- Check WebSocket host/port configuration
- Verify firewall settings for WebSocket port
- Use browser developer tools to check WebSocket errors

### Browser Compatibility
- Use modern browsers (Chrome 88+, Firefox 85+, Safari 14+, Edge 88+)
- Enable JavaScript in browser
- Check for ad blockers that might block WebSocket connections

### Network Issues
- For remote connections, ensure STT server binds to `0.0.0.0`
- Check network connectivity between client and server
- Verify WebSocket port is not blocked by firewall

## Migration Notes

### From Keyboard Controller Version
If you were using the separate keyboard controller:

1. **Old way**:
```bash
# Terminal 1: STT system
python3 ws_stt.py --websocket-host localhost

# Terminal 2: Keyboard controls  
python3 keyboard_controller.py
```

2. **New way**:
```bash
# Terminal: STT system
python3 ws_stt.py --websocket-host localhost

# Browser: Open web_controls.html
```

### Configuration Changes
- No changes needed in `config.json`
- All button control settings still work
- WebSocket configuration remains the same
- Better cross-platform compatibility