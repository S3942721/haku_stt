# STT WebSocket Protocol

The main service accepts JSON commands and broadcasts recognition output on `ws://STT_HOST:8765`. This is separate from the binary browser-audio endpoint on port `8787`. Run from the service directory as described in the [repository README](../../README.md).

## Connect

```bash
python3 ws_stt.py --device 0 --websocket-host 0.0.0.0 --websocket-port 8765
```

Replace `0` with a discovered microphone device ID, or use `--device remote` for Pepper RTP input. The server sends a `status` message with `status: connected` when a client registers. The standalone [control page](../web_controls.html) connects to this endpoint and shows subsequent output.

## Processing controls

Send one JSON object per WebSocket text message:

```json
{"type":"control","action":"pause"}
```

| Action | Behaviour |
| --- | --- |
| `pause` | Pause processing and clear the current utterance/buffers |
| `resume` | Resume processing |
| `purge` / `reset` | Clear buffers and recognition state, then return to running |
| `stop` | Stop processing without exiting the server |
| `get_state` | Request the current service state and details |
| `ping` | Request status; add `"healthCheck":true` to receive `pong` |

Recognised controls receive an immediate `command_ack` with `status: queued`; this acknowledges acceptance, not completed execution. State changes are reported separately.

```json
{"type":"command_ack","action":"pause","status":"queued","timestamp":1234567890.123}
```

## Button commands

```json
{"type":"button","button":"manual_eou","action":"press"}
```

Buttons are `manual_eou`, `keep_listening`, `not_listen`, `push_to_talk`, and `mode_switch`. Ordinary actions are `press`, `release`, and `toggle`. A real operating-mode change uses the special action:

```json
{"type":"button","button":"mode_switch","action":"mode_switch"}
```

The bundled page currently sends `toggle` for its mode button, which changes a button state rather than selecting push-to-talk mode. Use the special message above from your client; see the [button reference](BUTTON_CONTROLS.md).

Button messages receive `button_ack` with `status: queued`. The acknowledgement is sent before the button controller validates the button/action, so it does not prove the requested operation took effect.

## Server messages

| Type | Fields / meaning |
| --- | --- |
| `partial` | `text`, `timestamp`, `confidence`; interim recognition |
| `complete` | `text`, `timestamp`, `confidence`; completed utterance |
| `status` | `status`, optional `details`, `timestamp`; service or mode update |
| `command_ack` | `action`, `status`, `timestamp`; accepted processing control |
| `button_ack` | `button`, `action`, `status`, `timestamp`; queued button message |
| `pong` | `status`, `healthCheck`, `timestamp`; health-check reply |
| `error` | `error`, `timestamp`; invalid JSON or unsupported command |

```json
{"type":"partial","text":"hello","timestamp":1234567890.123,"confidence":null}
```

```json
{"type":"complete","text":"Hello, Haku.","timestamp":1234567891.123,"confidence":null}
```

Timestamps are Unix seconds, not milliseconds. Confidence may be null. Transcription text is sanitised to ASCII by the current implementation. A client must handle non-transcription messages rather than assuming every message has `text`. Enabled Whisper paths can suppress completed output when no valid buffered recognition is available.

## Client example

```javascript
const socket = new WebSocket('ws://localhost:8765');
socket.onopen = () => {
  socket.send(JSON.stringify({ type: 'control', action: 'get_state' }));
};
socket.onmessage = ({ data }) => {
  const message = JSON.parse(data);
  if (message.type === 'complete') console.log('Utterance:', message.text);
  else if (message.type === 'partial') console.log('Partial:', message.text);
  else console.log(message.type, message);
};
```

For held buttons, send both press and release. Test actual state/output after a command, not only its acknowledgement.

## Access and troubleshooting

Use the actual service hostname/IP in client URLs; `0.0.0.0` is a bind address. Confirm TCP `8765` is reachable, the service has finished model initialisation, and the correct audio source is selected. HTTP-served control pages can use plain `ws://`; HTTPS-served pages need `wss://` forwarding to avoid mixed-content blocking. The service itself has no built-in TLS or application authentication.
