# Haku STT Remote Server

A real-time speech-to-text system with WebSocket control interface, designed to run as a remote server for external applications.

## Quick Start

### Basic Local Usage (Webcam Microphone)
```bash
python3 ws_stt.py --device 4
```
*Note: Device 4 is typically the webcam microphone - adjust the number based on your system*

### Remote Server Setup
```bash
# Enable WebSocket server for remote control
python3 ws_stt.py --device 4 --websocket-host 0.0.0.0 --websocket-port 8765
```

## Configuration Options

### Command Line Arguments
The system accepts various command line arguments that override the `config.json` settings:

```bash
# Basic server setup
python3 ws_stt.py --websocket-host 0.0.0.0 --websocket-port 8765

# Audio input options
--device 4                   # Use device ID 4 (webcam mic)
--device remote              # Use remote RTP audio stream
--device browser             # Use browser WebSocket audio (only set up in HTTPS)

# Model configuration
--asr-model stt_en_fastconformer_hybrid_large_streaming_multi
--punct-model punctuation_en_bert
--lookahead 480

# Logging and output
--log-level info             # Reduce verbosity
--show-raw                   # Show raw unpunctuated text
--quiet                      # Minimal output

# EOU (End of Utterance) settings
--eou-threshold 0.7          # EOU detection sensitivity
--no-eou                     # Disable automatic EOU detection
```

### Audio Input Sources

1. **Local Microphone** (Default)
   ```bash
   python3 ws_stt.py --device 4
   ```

2. **Remote Audio Stream** (RTP/GStreamer)
   ```bash
   python3 ws_stt.py --device remote --remote-port 5004
   ```

3. **Browser WebSocket Audio**
   ```bash
   python3 ws_stt.py --device browser --browser-audio-port 8787
   ```

## Remote Server Configuration

### Network Setup
To run as a remote server accessible from other machines:

```bash
# Allow connections from any IP
python3 ws_stt.py --websocket-host 0.0.0.0 --websocket-port 8765 --device 4
```

### Config.json Modifications
For permanent settings, edit `config.json`:

```json
{
  "websocket-host": "0.0.0.0",
  "websocket-port": 8765,
  "device": 4,
  "log-level": "info"
}
```

### Common Configuration Scenarios

#### 1. Production Server (Low Logging)
```bash
python3 ws_stt.py --websocket-host 0.0.0.0 --log-level warning --device 4
```

#### 2. Development Server (Verbose Logging)
```bash
python3 ws_stt.py --websocket-host 0.0.0.0 --log-level debug --device 4
```

#### 3. High-Performance Mode
```bash
python3 ws_stt.py --websocket-host 0.0.0.0 --asr-model stt_en_fastconformer_hybrid_large_streaming_80ms --device 4
```

#### 4. Remote Audio Input
```bash
python3 ws_stt.py --websocket-host 0.0.0.0 --device remote --remote-port 5004
```

## WebSocket Control Interface

### Connection URL
```
ws://your-server-ip:8765
```

### Web Control Panel
Open `web_controls.html` in a browser and connect to your server:
```
http://file:///path/to/stt/web_controls.html
```

### API Commands
Send JSON commands via WebSocket:

```javascript
// Manual end-of-utterance
{"type": "button", "button": "manual_eou", "action": "press"}

// System control
{"type": "control", "action": "pause"}
{"type": "control", "action": "resume"}
{"type": "control", "action": "reset"}

// Audio control
{"type": "button", "button": "not_listen", "action": "press"}   // Mute
{"type": "button", "button": "not_listen", "action": "release"} // Unmute
```

## Device ID Discovery

To find your audio device ID:
```bash
python3 -c "
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f'Device {i}: {info[\"name\"]}')
p.terminate()
"
```

## Firewall Configuration

### Allow WebSocket Port
```bash
# Ubuntu/Debian
sudo ufw allow 8765

# CentOS/RHEL
sudo firewall-cmd --add-port=8765/tcp --permanent
sudo firewall-cmd --reload
```

### Allow Remote Audio Port (if using)
```bash
sudo ufw allow 5004
sudo ufw allow 8787  # For browser audio
```

## Troubleshooting

### Common Issues

1. **"Device not found" Error**
   - Run device discovery command above
   - Try different device numbers (0, 1, 2, 3, 4...)

2. **WebSocket Connection Refused**
   - Check firewall settings
   - Ensure `--websocket-host 0.0.0.0`
   - Verify port not in use: `netstat -an | grep 8765`

3. **Audio Quality Issues**
   - Adjust chunk size: `--chunk-size 80`
   - Try different ASR model: `--asr-model stt_en_fastconformer_hybrid_large_streaming_80ms`

4. **High CPU Usage**
   - Reduce log level: `--log-level warning`
   - Disable text EOU: `--no-eou` or set `"enable-text-eou": false` in config

5. **Permission Errors**
   - Run with sudo if needed for audio device access
   - Check audio device permissions: `ls -l /dev/snd/`

### Performance Tuning

#### For Real-time Performance
```bash
python3 ws_stt.py --websocket-host 0.0.0.0 --device 4 \
  --asr-model stt_en_fastconformer_hybrid_large_streaming_80ms \
  --chunk-size 80 --log-level warning
```

#### For Accuracy Over Speed
```bash
python3 ws_stt.py --websocket-host 0.0.0.0 --device 4 \
  --asr-model stt_en_fastconformer_hybrid_large_streaming_1040ms \
  --lookahead 1040 --log-level warning
```

## Dependencies

### Required Python Packages
- `websockets` - WebSocket server functionality
- `pyaudio` - Audio input from microphones
- `torch` - PyTorch for ML models
- `nemo_toolkit` - NVIDIA NeMo for ASR
- `transformers` - HuggingFace transformers
- `onnxruntime` - ONNX model inference

### Optional Dependencies
- `gstreamer` - For remote audio streaming (remote_audio_stream.py)
- `silero-vad` - Alternative VAD model

### Installation
```bash
pip install websockets pyaudio torch nemo_toolkit[asr] transformers onnxruntime
```

## Example Integration

### Python Client
```python
import asyncio
import websockets
import json

async def stt_client():
    uri = "ws://your-server-ip:8765"
    async with websockets.connect(uri) as websocket:
        # Listen for transcriptions
        async for message in websocket:
            data = json.loads(message)
            if data["type"] == "complete":
                print(f"Transcription: {data['text']}")
            elif data["type"] == "partial":
                print(f"Partial: {data['text']}", end='\r')

asyncio.run(stt_client())
```

### JavaScript Client
```javascript
const ws = new WebSocket('ws://your-server-ip:8765');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'complete') {
        console.log('Final:', data.text);
    } else if (data.type === 'partial') {
        console.log('Partial:', data.text);
    }
};

// Send manual EOU
ws.send(JSON.stringify({
    type: 'button',
    button: 'manual_eou',
    action: 'press'
}));
```

## Documentation

- `docs/WEB_CONTROLS.md` - Detailed WebSocket API documentation
- `docs/BUTTON_CONTROLS.md` - Button control system guide
- `docs/WEB_CONTROLS_QUICKSTART.md` - Quick start guide for web controls

## License

This software uses various open-source components. Please check individual model licenses when using in production.