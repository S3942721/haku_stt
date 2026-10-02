# STT Button Controls

Buttons let an operator override recognition and end-of-utterance behaviour through the JSON WebSocket service. Use the [operator quick start](WEB_CONTROLS_QUICKSTART.md) for the HTML interface and the [protocol guide](WEB_CONTROLS.md) for message formats.

## Operating modes

The server starts in `standard` mode: audio is processed unless paused, stopped, or muted. In `push_to_talk` mode, audio is processed while the push-to-talk button is held. Releasing it completes the current utterance if text has accumulated.

The `ButtonController` handles WebSocket commands; there is no separate `keyboard_controller.py` in this service. Keyboard shortcuts are implemented by the HTML page and require page focus.

## Buttons

| Button | Press / hold effect | Release effect | HTML shortcut |
| --- | --- | --- | --- |
| `manual_eou` | Immediately finalise accumulated text, if present | No additional operation | Space |
| `keep_listening` | Suppress automatic EOU and keep accumulating | Allow EOU; finalise if one was pending | Ctrl |
| `not_listen` | Mute audio processing | Unmute processing | Shift |
| `push_to_talk` | Enable audio in push-to-talk mode | Complete accumulated text | Alt |
| `mode_switch` | Use the special mode action described below | — | M, subject to the UI limitation below |

Held controls use paired messages:

```json
{"type":"button","button":"not_listen","action":"press"}
```

```json
{"type":"button","button":"not_listen","action":"release"}
```

Mute prevents recognition processing; it does not physically disable the microphone or stop network audio transport. Always release held controls after use. An immediate `button_ack` means the message was queued, not that its action was valid or executed.

## Switching to push-to-talk

The server's special operating-mode action is:

```json
{"type":"button","button":"mode_switch","action":"mode_switch"}
```

Each such command switches between `standard` and `push_to_talk` and emits a `status` update with `status: mode_changed` and `details.new_mode`. The ASR mode callback clears its internal mute/listening overrides and recognition state. It does not reset every `ButtonController` held-button state; release held buttons before switching modes.

**Current HTML limitation:** `web_controls.html` sends `action: toggle` for its mode button and M shortcut. That only toggles the `mode_switch` button state and does not select a different operating mode. Send the special command above from a WebSocket client until the page is updated. A `button_ack` alone will not reveal this difference; look for the `mode_changed` status.

## Troubleshooting

- If shortcuts do nothing, focus the control page, check the connection, and try the mouse/touch button.
- If no text completes, confirm recognised text exists and inspect the EOU/Whisper settings; manual EOU cannot create a transcription from silence.
- If audio stays muted, send `not_listen` release and confirm the server is running rather than paused/stopped.
- If push-to-talk has no effect, confirm `mode_changed` selected `push_to_talk`; toggling the HTML mode button is insufficient in the current implementation.
- If a queued acknowledgement never becomes a state change, check the requested button/action and service logs.
