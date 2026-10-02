# Haku Speech-to-Text

A streaming speech-recognition service for robot conversations. Haku STT combines NVIDIA NeMo FastConformer recognition, punctuation/capitalisation, voice activity detection, and end-of-utterance detection to deliver partial text and completed utterances over WebSocket.

Audio can come from a local microphone, Pepper's RTP stream, or browser-supplied PCM frames. The service runs on a workstation and integrates with the Haku web controller; it does not itself execute robot commands or generate language-model responses.

## Features

- Cache-aware streaming ASR with configurable lookahead and decoder settings.
- Punctuation and capitalisation post-processing.
- NVIDIA or Silero voice activity detection, with optional text-based end-of-utterance checks.
- Optional Whisper analysis of buffered utterances.
- WebSocket transcription broadcasts, processing controls, and button overrides.
- A standalone HTML control page for local and remote operation.
- Configuration profiles and rotating file logging.

## Repository layout

| Path | Purpose |
| --- | --- |
| [stt/](stt/) | Main service, configuration, control page, and audio receivers |
| [stt/ws_stt.py](stt/ws_stt.py) | Service entry point and recognition/control logic |
| [stt/config.json](stt/config.json) | Mode selection, models, thresholds, and network settings |
| [stt/remote_audio_stream.py](stt/remote_audio_stream.py) | GStreamer RTP receiver and resampling |
| [stt/websocket_audio_stream.py](stt/websocket_audio_stream.py) | Binary PCM audio WebSocket receiver |
| [stt/docs/](stt/docs/) | Protocol, button controls, and operator quick start |
| [development/](development/) | Earlier recognition/integration experiments; not the primary service |
| [testing/](testing/) | Manual/model-dependent checks and WebSocket clients |
| [setup.bash](setup.bash) | Activates an already-existing Conda environment named `nemo`; does not install dependencies |

## Environment setup

Use Python 3.10 in a separate environment. A CUDA-capable NVIDIA GPU is useful for the supplied ASR and Whisper models; the code checks CUDA availability and also has CPU paths, but real-time CPU performance is not established here. Plan memory around your chosen models rather than a fixed hardware minimum.

```bash
git clone https://github.com/S3942721/haku_stt.git
cd haku_stt
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a PyTorch build appropriate for your host/CUDA environment first. On Debian/Ubuntu, PyAudio typically needs PortAudio headers and the service's audio tooling needs system libraries:

```bash
sudo apt install build-essential python3-dev portaudio19-dev libsndfile1 ffmpeg
```

The main imports require NeMo's **ASR and NLP** components, PyAudio, NumPy, OmegaConf, ONNX Runtime, Transformers, Hugging Face Hub, and WebSockets. The integrated environment used NeMo 2.4.0 and PyTorch 2.5.1; the following is a starting point for dependency resolution after installing PyTorch:

```bash
python -m pip install 'nemo_toolkit[asr,nlp]==2.4.0' \
  pyaudio numpy omegaconf onnxruntime transformers huggingface_hub websockets
```

This standalone repository does not provide a tested dependency lock or an installer. NeMo, PyTorch, Transformers, and native audio dependencies must be compatible; this command is not a guarantee for every platform. The parent integration repository's `environment.yml` records the original Linux environment, but contains machine-specific build pins, an absolute prefix, and mixed CUDA packages, so review it before using it as an installation recipe.

For robot RTP audio, also install GStreamer, the RTP/audio plugins, and Python GObject bindings in the interpreter running STT. For example, Debian/Ubuntu system packages include:

```bash
sudo apt install python3-gi gir1.2-gstreamer-1.0 \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

System `python3-gi` may not be visible inside a venv/Conda interpreter; use bindings compatible with that interpreter. Confirm `import gi` and the required GStreamer plugins work before selecting remote audio. First startup downloads model assets and therefore needs network access and cache space.

## Start the service

Run from the repository's **`stt/` directory** so `config.json` is loaded:

```bash
cd stt
python3 ws_stt.py --device remote --remote-port 5004 \
  --websocket-host 0.0.0.0 --websocket-port 8765 --log-level info
```

This example receives Pepper's audio and exposes control/transcription on `ws://STT_HOST:8765`. Point the robot handler's audio-stream destination at this machine's UDP port `5004`, and the web controller's `STT_SERVER_HOST` / `STT_SERVER_PORT` at this machine and `8765`.

### Local microphone

Device IDs are machine-specific; there is no universal webcam/USB microphone ID. Discover available inputs:

```bash
python3 - <<'PY'
import pyaudio
p = pyaudio.PyAudio()
for index in range(p.get_device_count()):
    info = p.get_device_info_by_index(index)
    if info['maxInputChannels'] > 0:
        print(index, info['name'])
p.terminate()
PY
```

Then start, replacing `0` with the desired ID:

```bash
python3 ws_stt.py --device 0 --websocket-host 0.0.0.0 --websocket-port 8765
```

### Browser audio

```bash
python3 ws_stt.py --device browser --browser-audio-port 8787 \
  --websocket-host 0.0.0.0 --websocket-port 8765
```

Port `8787` accepts binary **16 kHz, mono, little-endian signed PCM16** frames; port `8765` carries JSON controls and transcriptions. The receiver does not decode WebM/Opus or other encoded browser recordings. The standalone `web_controls.html` page controls processing; it is not a microphone-audio sender. Browser microphone capture requires a secure context such as localhost or HTTPS, with secure WebSocket forwarding for an HTTPS deployment.

## Configuration

`config.json` contains `mode-selection`, `modes.default`, named mode overrides, `vad-defaults`, and setting descriptions. Keep that structure: a flat JSON object containing only ports/device values is not accepted by `load_config()`.

The supplied selected mode is `fast-eou-buffer-accurate-medium`, enabling Silero VAD, text EOU, and buffered Whisper analysis with `distil-whisper/distil-large-v3`. It adds model downloads and memory use beyond the streaming recogniser.

To select a profile reliably, **edit `mode-selection` in the file**. The current `--mode-selection` implementation changes a metadata value after settings have already been merged; it does not reload that mode's settings. Set `mode-selection` to `default` in a copied config if you want a starting point with text EOU and Whisper disabled.

From the `stt/` directory:

```bash
cp config.json local-config.json
# Edit mode-selection and settings in local-config.json.
python3 ws_stt.py --config local-config.json --device remote --log-level info
```

CLI flags override loaded settings. Examples include `--asr-model`, `--punct-model`, `--lookahead`, `--decoder`, `--device`, `--remote-port`, `--browser-audio-port`, `--websocket-host`, `--websocket-port`, `--log-level`, `--log-file`, and `--no-eou`. `--device` selects an audio source, not a CUDA device.

Boolean CLI flags only set a value to true; disable an already-enabled feature in the JSON file. The parser is generated from the working-directory `config.json`, even when a custom file is passed through `--config`; run from this directory and use compatible configuration keys. `--quiet` appears in older examples but is **not** a generated option in the checked-in configuration; use `--log-level info` or `warning` instead.

## Web controls and protocol

In another terminal, from `stt/`:

```bash
python3 -m http.server 8080
```

Open `http://localhost:8080/web_controls.html` and connect to `ws://localhost:8765`. For a remote operator, replace localhost with the STT host. Serving this control-only page does not enable browser microphone capture.

```json
{"type":"control","action":"get_state"}
```

Supported control actions are `pause`, `resume`, `purge`, `reset`, `stop`, `ping`, and `get_state`. Button messages can request manual EOU, hold listening open, mute processing, or control push-to-talk. The service sends `partial`, `complete`, `status`, acknowledgement, and error messages.

See the [protocol guide](stt/docs/WEB_CONTROLS.md), [button reference](stt/docs/BUTTON_CONTROLS.md), and [operator quick start](stt/docs/WEB_CONTROLS_QUICKSTART.md). Those guides also describe the current mode-switch UI limitation.

## Network and verification

| Port | Protocol | Purpose |
| --- | --- | --- |
| `8765` | WebSocket / TCP | JSON control and transcription |
| `5004` | RTP / UDP | L16, mono, 44.1 kHz robot audio, resampled to 16 kHz |
| `8787` | WebSocket / TCP | Raw browser PCM audio, only in browser mode |
| `8080` | HTTP / TCP | Optional standalone control-page server |

The service has no application authentication or built-in TLS. Limit exposure to the intended network and use a suitable proxy for secure browser connections.

A useful smoke check is to connect the control page, request state, speak a short phrase, observe partial text and a completed utterance, and check pause/resume and manual EOU. In robot mode, check that RTP packets arrive even before investigating recognition output. Test STT independently before enabling the full LLM conversation.

## Troubleshooting

- **Import/build errors:** check the active interpreter, compatible NeMo/PyTorch/NLP packages, and native PortAudio/GStreamer bindings.
- **No device / no audio:** discover input IDs, check device access/sample rate, and confirm another process has not taken the device. Fix permissions in the normal user environment rather than running the model service as root.
- **No RTP input:** confirm robot destination IP, UDP `5004`, GStreamer L16 plugins, and sender/receiver caps.
- **No complete messages:** inspect VAD/EOU thresholds and the active Whisper settings. Whisper-enabled paths can suppress final output when buffered recognition yields no valid text.
- **Slow inference / out of memory:** start with the `default` file profile and fewer optional models, inspect CUDA availability, and then add models deliberately.
- **Config falls back to defaults:** confirm the working directory and required mode structure; startup prints loading errors.
- **Mode selector has no effect:** change `mode-selection` in the JSON file before launch; see the CLI limitation above.
- **Web controls disconnected:** confirm the correct host and TCP `8765`; use a reachable address rather than `0.0.0.0` in the client URI.

Scripts under `testing/` include manual experiments and model-dependent checks. They are not a dependency-free automated test suite; inspect each script's imports and hardware assumptions before running it.

## Integration and licence

See [Haku Control System](https://github.com/S3942721/haku_control_system) for deployment with the robot and [Robot Web Controller](https://github.com/S3942721/robo-web-controller) for conversation routing.

Repository code is distributed under [GPL-3.0](LICENSE). NeMo, ASR/Whisper/EOU model weights, and other dependencies have separate licences and usage terms.
