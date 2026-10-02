# STT Web Controls Quick Start

Prepare dependencies using the [repository README](../../README.md). The page controls recognition and displays text; it does not capture or stream browser microphone audio.

## 1. Start recognition

From the repository's `stt/` directory, in your prepared Python environment:

```bash
# Replace 0 with your discovered microphone device ID.
python3 ws_stt.py --device 0 --websocket-host 0.0.0.0 \
  --websocket-port 8765 --log-level info
```

For Pepper's RTP input, replace `--device 0` with `--device remote --remote-port 5004` and configure the robot to stream to this machine. Allow model initialisation to finish before connecting.

## 2. Open the page

In another terminal, from the same directory:

```bash
python3 -m http.server 8080
```

Open `http://localhost:8080/web_controls.html`. For another device on the network, use `http://STT_HOST:8080/web_controls.html`. You may also open the HTML file directly in a browser if its file-origin policy permits the WebSocket connection.

## 3. Connect and verify

1. Enter `ws://localhost:8765`, or `ws://STT_HOST:8765` remotely.
2. Click **Connect** and confirm a connected status.
3. Speak a short phrase and look for partial text followed by a complete utterance.
4. Use **Manual EOU** if you need to finalise accumulated text.
5. Hold **Mute**, then release it and verify audio processing resumes.

| Control | Shortcut while the page is focused |
| --- | --- |
| Manual EOU | Space |
| Keep Listening | Ctrl, held |
| Mute | Shift, held |
| Push-to-Talk | Alt, held; requires push-to-talk mode |
| Mode button | M; currently affected by the limitation below |

The current mode button sends a button toggle rather than the server's special operating-mode command. Use a WebSocket client to send `{"type":"button","button":"mode_switch","action":"mode_switch"}` and confirm a `mode_changed` status before testing push-to-talk. See the [button guide](BUTTON_CONTROLS.md).

## Connection problems

- Use a reachable hostname/IP, not the bind address `0.0.0.0`.
- Confirm the service is listening on TCP `8765`; the page server is a separate listener on `8080`.
- Check microphone selection or incoming robot UDP `5004` packets if connected but no text appears.
- Inspect startup configuration and EOU/Whisper settings if partial text appears without complete messages.
- If serving the page over HTTPS, forward the service through `wss://`; plain `ws://` may be blocked.

The server and page are intended for a trusted network and provide no application authentication. See the [protocol guide](WEB_CONTROLS.md) for pause/resume, state requests, and client integration.
