# Haku STT Service

This directory contains the main speech-recognition service. Start here for runtime files; use the [repository README](../README.md) for environment setup, architecture, model configuration, integration, and troubleshooting.

## Run

Activate your prepared Python environment, then run from this directory:

```bash
# Receive Pepper's RTP/UDP stream.
python3 ws_stt.py --device remote --remote-port 5004 \
  --websocket-host 0.0.0.0 --websocket-port 8765 --log-level info

# Alternatively, use a local microphone (replace 0 with its discovered ID).
python3 ws_stt.py --device 0 --websocket-host 0.0.0.0 --websocket-port 8765

# Alternatively, receive raw 16 kHz mono PCM16 over WebSocket.
python3 ws_stt.py --device browser --browser-audio-port 8787 \
  --websocket-host 0.0.0.0 --websocket-port 8765
```

Use one audio mode at a time. The control/transcription connection is `ws://STT_HOST:8765`; the browser audio port is separate. The control page does not stream microphone audio.

## Configuration

[config.json](config.json) supplies the mode structure and CLI setting descriptions. The selected mode enables additional EOU and Whisper models. Edit `mode-selection` in the file to change presets: the current `--mode-selection` CLI flag does not remerge the selected preset. Boolean CLI flags enable values; edit the file to disable them. Older `--quiet` examples are unsupported by this checked-in configuration.

Use `--config PATH` for a compatible alternative configuration, while keeping this directory as the working directory so the CLI parser sees the intended setting keys.

## Operator controls

In a separate terminal in this directory:

```bash
python3 -m http.server 8080
```

Open `http://localhost:8080/web_controls.html` and connect to `ws://localhost:8765`. Replace localhost with the service host for remote operation.

- [WebSocket protocol](docs/WEB_CONTROLS.md)
- [Button controls and operating modes](docs/BUTTON_CONTROLS.md)
- [Operator quick start](docs/WEB_CONTROLS_QUICKSTART.md)

The bundled mode-switch button needs the workaround documented in the button guide.

## Runtime files

| File | Responsibility |
| --- | --- |
| [ws_stt.py](ws_stt.py) | Streaming ASR, models, EOU logic, configuration, and JSON WebSocket controls |
| [remote_audio_stream.py](remote_audio_stream.py) | Robot RTP receiver, GStreamer conversion, and audio queue |
| [websocket_audio_stream.py](websocket_audio_stream.py) | Binary PCM receiver and audio queue |
| [web_controls.html](web_controls.html) | Browser controls and transcription display |

The parent repository's [licence](../LICENSE) applies to service code; model/dependency terms are separate.
