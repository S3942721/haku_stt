# STT Web Controls - Quick Start Guide

## What Changed

The keyboard controls have been replaced with a **web-based interface** for better compatibility and ease of use.

## Quick Start

### 1. Start the STT System
```bash
python3 ws_stt.py --websocket-host 0.0.0.0 --websocket-port 8765
```

### 2. Open Web Controls
Open `web_controls.html` in any web browser:

**Option A: Local file**
- Double-click `web_controls.html` 
- Or drag it into your browser

**Option B: HTTP server (recommended)**
```bash
# Start simple HTTP server
python3 -m http.server 8080

# Then open: http://localhost:8080/web_controls.html
```

### 3. Connect to STT System
1. In the web interface, enter: `ws://localhost:8765`
2. Click "Connect"
3. You should see "Connected" status

### 4. Use Controls
- **Manual EOU**: Click to trigger end-of-utterance
- **Keep Listening**: Hold to override automatic EOU
- **Mute**: Hold to stop audio processing
- **Push-to-Talk**: Hold to talk (in PTT mode)
- **Mode Switch**: Toggle between Standard/PTT modes

## Features

✅ **Cross-platform**: Works on any device with a web browser  
✅ **No installation**: No keyboard libraries or dependencies needed  
✅ **Remote control**: Control STT from any device on the network  
✅ **Mobile friendly**: Touch controls for phones/tablets  
✅ **Real-time output**: See STT transcription in the web interface  
✅ **Visual feedback**: Button states and connection status  

## For Remote Access

### Server Setup
```bash
# Start STT system accessible from network
python3 ws_stt.py --websocket-host 0.0.0.0 --websocket-port 8765

# Optional: Start HTTP server for web interface
python3 -m http.server 8080 --bind 0.0.0.0
```

### Client Access
1. Open web browser on any device
2. Navigate to: `http://[SERVER_IP]:8080/web_controls.html`
3. Connect to: `ws://[SERVER_IP]:8765`
4. Control STT system remotely!

## Keyboard Shortcuts (Optional)

When the web page has focus, you can use:
- **Space**: Manual EOU
- **Ctrl**: Keep Listening (hold)
- **Shift**: Mute (hold)  
- **Alt**: Push-to-Talk (hold)
- **M**: Mode Switch

## Troubleshooting

**Connection Issues:**
- Ensure STT system is running with WebSocket enabled
- Check that the WebSocket URI is correct
- Use browser developer tools (F12) to check for errors

**Browser Compatibility:**
- Use modern browsers (Chrome, Firefox, Safari, Edge)
- Ensure JavaScript is enabled
- Try disabling ad blockers

**Network Issues:**
- For remote access, use `--websocket-host 0.0.0.0`
- Check firewall settings for ports 8765 and 8080
- Verify network connectivity

## Documentation

- `WEB_CONTROLS.md` - Complete web interface documentation
- `BUTTON_CONTROLS.md` - Button functionality details
- `config.json` - STT system configuration

---

**Migration Note**: The separate `keyboard_controller.py` script has been removed in favor of this web interface for better cross-platform compatibility.