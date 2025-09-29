# Button Control System for WebSocket STT

This document describes the button control functionality for the WebSocket-based Speech-to-Text system.

## Overview

The button control system allows real-time control of STT processing through WebSocket commands. Button input is handled by a web interface (`web_controls.html`) that connects via WebSocket.

**Note**: Button controls are now provided via a web interface. See [WEB_CONTROLS.md](WEB_CONTROLS.md) for complete details.

## Architecture

### Main STT System (`ws_stt.py`)
- Processes speech-to-text and WebSocket commands
- ButtonController handles WebSocket button commands only
- No direct keyboard or UI input handling

### Web Controls (`web_controls.html`) 
- HTML/JavaScript web interface for button controls
- Connects to STT system via WebSocket
- Provides visual buttons and optional keyboard shortcuts
- Displays real-time STT output

## Usage

### 1. Start STT System
```bash
python3 ws_stt.py --websocket-host 0.0.0.0
```

### 2. Open Web Controls
1. Open `web_controls.html` in any web browser
2. Enter WebSocket URI (e.g., `ws://localhost:8765`)
3. Click "Connect"
4. Use the button controls

## Operating Modes

### Standard Mode (Default)
- Always listens and processes audio unless explicitly muted
- Automatic EOU (End of Utterance) detection is active
- All button overrides function normally

### Push-to-Talk Mode
- Only listens when the Push-to-Talk button is held down
- Microphone is effectively muted unless PTT button is active
- When PTT button is released, triggers automatic EOU if speech was processed
- Other button functions still work normally

## Button Functions

### 1. Manual EOU (Space key / WebSocket command)
- **Purpose**: Manually trigger End of Utterance detection
- **Behavior**: If there's any text in the processing buffer, it's immediately processed as a complete utterance
- **Use case**: When automatic EOU detection misses the end of speech

### 2. Keep Listening Override (Ctrl key / WebSocket command)
- **Purpose**: Temporarily override automatic EOU detection
- **Behavior**: 
  - While held: Automatic EOU is suppressed, speech continues to accumulate
  - When released: If an EOU was detected while held, triggers immediately
- **Use case**: When you know you're going to continue speaking despite a natural pause

### 3. Not Listen / Mute (Shift key / WebSocket command)
- **Purpose**: Temporarily mute audio input
- **Behavior**: While held, audio is ignored (gain effectively set to -inf)
- **Use case**: Temporarily stop listening without changing modes

### 4. Push-to-Talk (Alt key / WebSocket command)
- **Purpose**: Control listening in Push-to-Talk mode
- **Behavior**: 
  - Only active in Push-to-Talk mode
  - While held: Audio processing is enabled
  - When released: Triggers EOU if speech was processed
- **Use case**: Traditional push-to-talk operation

### 5. Mode Switch (M key / WebSocket command)
- **Purpose**: Toggle between Standard and Push-to-Talk modes
- **Behavior**: Switches modes and resets all button states
- **Use case**: Runtime switching between always-on and push-to-talk operation

## Keyboard Controls (via keyboard_controller.py)

Keyboard input is now handled by a separate `keyboard_controller.py` script. See [KEYBOARD_CONTROLLER.md](KEYBOARD_CONTROLLER.md) for full details.

### Key Mappings in keyboard_controller.py
- **Space**: Manual EOU
- **Ctrl**: Keep Listening Override
- **Shift**: Not Listen (Mute)
- **Alt**: Push-to-Talk
- **M**: Mode Switch

### Terminal Fallback Mode
If the `keyboard` library is not available, keyboard_controller.py uses terminal input:
- **Space**: Manual EOU
- **c**: Keep Listening Override (toggle)
- **s**: Not Listen (toggle)
- **p**: Push-to-Talk (toggle)
- **m**: Mode Switch
- **q**: Quit

## WebSocket Commands

Button commands can be sent via WebSocket using the following JSON format:

### Basic Button Command Structure
```json
{
    "type": "button",
    "button": "button_name",
    "action": "action_type"
}
```

### Available Buttons
- `manual_eou`
- `keep_listening`
- `not_listen`
- `push_to_talk`
- `mode_switch`

### Available Actions
- `press`: Activate the button
- `release`: Deactivate the button
- `toggle`: Toggle button state
- `mode_switch`: Special action for mode switching

### Examples

#### Manual EOU Trigger
```json
{
    "type": "button",
    "button": "manual_eou",
    "action": "press"
}
```

#### Start Push-to-Talk
```json
{
    "type": "button",
    "button": "push_to_talk",
    "action": "press"
}
```

#### Switch Modes
```json
{
    "type": "button",
    "button": "mode_switch",
    "action": "toggle"
}
```

#### Mute Audio
```json
{
    "type": "button",
    "button": "not_listen",
    "action": "press"
}
```

## Integration Points

### ASR Processing Integration
- Audio processing checks button states before transcribing chunks
- EOU detection considers button overrides before triggering
- Mode switching affects fundamental audio processing behavior

### WebSocket Integration
- Button commands are processed through the existing WebSocket command queue
- Button acknowledgments are sent back to clients
- Mode changes are broadcast as status updates

### State Management
- Button states are tracked independently
- Mode changes reset relevant button states
- System maintains consistency between keyboard and WebSocket inputs

## Technical Implementation

### ButtonController Class
- Manages all button states and mode switching
- Handles both keyboard input (via threading) and WebSocket commands
- Provides callback system for integration with ASR processing

### Audio Processing Integration
- `should_process_audio_with_buttons()`: Determines if audio should be processed
- `check_eou_with_buttons()`: EOU detection considering button states
- Mode-aware processing in `transcribe_chunk()`

### WebSocket Command Processing
- Extended existing command handler to support button commands
- Added button command queue processing in main command loop
- Integrated with existing status and acknowledgment system

## Usage Examples

### Starting the System
```bash
# With keyboard controls
python ws_stt.py --websocket-host localhost

# With custom configuration
python ws_stt.py --config my_config.json --websocket-host 0.0.0.0
```

### Testing Button Controls
```bash
# Run the test script
python test_button_controls.py
```

### WebSocket Client Integration
```python
import asyncio
import websockets
import json

async def send_button_command(button, action):
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        command = {
            "type": "button",
            "button": button,
            "action": action
        }
        await websocket.send(json.dumps(command))
        response = await websocket.recv()
        return json.loads(response)

# Example usage
asyncio.run(send_button_command("manual_eou", "press"))
```

## Dependencies

### Required
- `threading` (built-in)
- `asyncio` (built-in)
- Existing WebSocket infrastructure

### Optional
- `keyboard` library for enhanced keyboard input
- `termios` and `select` for fallback terminal input (Unix-like systems)

### Installation
```bash
# For enhanced keyboard support
pip install keyboard

# Note: keyboard library may require root access on some systems
# Fallback terminal mode works without additional dependencies
```

## Troubleshooting

### Keyboard Input Not Working
1. Try installing the `keyboard` library: `pip install keyboard`
2. Run with sudo if needed: `sudo python ws_stt.py`
3. Use WebSocket commands as alternative
4. Check terminal input fallback mode

### WebSocket Commands Not Responding
1. Verify WebSocket server is enabled (`--websocket-host` parameter)
2. Check WebSocket connection to correct host/port
3. Verify JSON command format
4. Check system logs for error messages

### Push-to-Talk Mode Issues
1. Ensure you're in Push-to-Talk mode (`M` key or mode switch command)
2. Hold the appropriate button (Alt key or `push_to_talk` command)
3. Check button state via status commands
4. Verify audio input is working

### Audio Muting Issues
1. Check if "Not Listen" button is active
2. In Push-to-Talk mode, ensure PTT button is held
3. Verify button states via WebSocket status
4. Check system audio settings

## Future Enhancements

### Planned Features
- Custom key mapping configuration
- Button state persistence
- Multiple simultaneous button support
- GUI integration for button controls
- Voice activation threshold controls
- Advanced push-to-talk configurations

### API Extensions
- REST API for button control
- Button state query endpoints
- Configuration management API
- Integration with external control systems