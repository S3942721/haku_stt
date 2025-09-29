#!/usr/bin/env python3
"""
Minimal test for ButtonController functionality
This test isolates the ButtonController class to verify its basic operation
"""

import sys
import time
import threading
from collections import deque

# Mock the LoggerMixin to avoid dependencies
class LoggerMixin:
    def __init_logger__(self, name, **kwargs):
        class MockLogger:
            def debug(self, msg): print(f"[DEBUG] {msg}")
            def info(self, msg): print(f"[INFO] {msg}")
            def warning(self, msg): print(f"[WARNING] {msg}")
            def error(self, msg): print(f"[ERROR] {msg}")
        self.logger = MockLogger()

# Mock availability flags
KEYBOARD_AVAILABLE = False
TERMINAL_INPUT_AVAILABLE = True

# Simplified ButtonController for testing
class ButtonController(LoggerMixin):
    """Controller for handling button inputs via keyboard and WebSocket commands"""
    
    def __init__(self, log_config=None):
        """Initialize button controller with keyboard and WebSocket support"""
        if log_config is None:
            log_config = {"log_level": "debug"}
        self.__init_logger__("ButtonController", **log_config)
        
        # Button states
        self.button_states = {
            'manual_eou': False,
            'keep_listening': False,
            'not_listen': False,
            'push_to_talk': False,
            'mode_switch': False
        }
        
        # Operating modes
        self.current_mode = 'standard'  # 'standard' or 'push_to_talk'
        
        # Key mappings
        self.key_mappings = {
            'space': 'manual_eou',
            'ctrl': 'keep_listening',
            'shift': 'not_listen',
            'alt': 'push_to_talk',
            'm': 'mode_switch'
        }
        
        # Threading control
        self.running = False
        self.keyboard_thread = None
        self.button_callbacks = {}
        
        # Keyboard hook setup
        self.keyboard_available = KEYBOARD_AVAILABLE
        self.terminal_available = TERMINAL_INPUT_AVAILABLE
        
        self.logger.info(f"ButtonController initialized - Keyboard: {self.keyboard_available}, Terminal: {self.terminal_available}")
    
    def set_callback(self, button_name, callback_func):
        """Set callback function for button events"""
        self.button_callbacks[button_name] = callback_func
        self.logger.debug(f"Callback set for button: {button_name}")
    
    def get_button_state(self, button_name):
        """Get current state of a button"""
        return self.button_states.get(button_name, False)
    
    def set_button_state(self, button_name, state, trigger_callback=True):
        """Set button state and optionally trigger callback"""
        old_state = self.button_states.get(button_name, False)
        self.button_states[button_name] = state
        
        # Trigger callback if state changed
        if trigger_callback and old_state != state and button_name in self.button_callbacks:
            try:
                self.button_callbacks[button_name](button_name, state)
            except Exception as e:
                self.logger.error(f"Error in callback for {button_name}: {e}")
    
    def get_mode(self):
        """Get current operating mode"""
        return self.current_mode
    
    def set_mode(self, mode):
        """Set operating mode"""
        if mode in ['standard', 'push_to_talk']:
            old_mode = self.current_mode
            self.current_mode = mode
            self.logger.info(f"Mode changed from {old_mode} to {mode}")
            
            # Trigger mode change callback if available
            if 'mode_change' in self.button_callbacks:
                try:
                    self.button_callbacks['mode_change'](mode, old_mode)
                except Exception as e:
                    self.logger.error(f"Error in mode change callback: {e}")
        else:
            self.logger.warning(f"Invalid mode: {mode}")
    
    def process_websocket_command(self, command):
        """Process button commands from WebSocket"""
        if not isinstance(command, dict):
            return False
            
        command_type = command.get("type")
        if command_type != "button":
            return False
            
        button_name = command.get("button")
        action = command.get("action", "press")
        
        if button_name not in self.button_states:
            self.logger.warning(f"Unknown button: {button_name}")
            return False
        
        if action == "press":
            self.set_button_state(button_name, True)
        elif action == "release":
            self.set_button_state(button_name, False)
        elif action == "toggle":
            current_state = self.get_button_state(button_name)
            self.set_button_state(button_name, not current_state)
        elif action == "mode_switch" and button_name == "mode_switch":
            new_mode = 'push_to_talk' if self.current_mode == 'standard' else 'standard'
            self.set_mode(new_mode)
        else:
            self.logger.warning(f"Unknown action for {button_name}: {action}")
            return False
        
        self.logger.debug(f"WebSocket button command: {button_name} {action}")
        return True

def test_button_controller():
    """Test the ButtonController functionality"""
    print("Testing ButtonController...")
    
    # Create controller
    controller = ButtonController()
    
    # Test callback system
    def mock_callback(button_name, state):
        print(f"CALLBACK: {button_name} = {state}")
    
    def mock_mode_callback(new_mode, old_mode):
        print(f"MODE CHANGE: {old_mode} -> {new_mode}")
    
    # Set up callbacks
    controller.set_callback('manual_eou', mock_callback)
    controller.set_callback('keep_listening', mock_callback)
    controller.set_callback('push_to_talk', mock_callback)
    controller.set_callback('mode_change', mock_mode_callback)
    
    # Test basic button operations
    print("\n1. Testing button state changes...")
    controller.set_button_state('manual_eou', True)
    controller.set_button_state('manual_eou', False)
    
    # Test mode switching
    print("\n2. Testing mode switching...")
    controller.set_mode('push_to_talk')
    controller.set_mode('standard')
    
    # Test WebSocket commands
    print("\n3. Testing WebSocket commands...")
    
    commands = [
        {"type": "button", "button": "manual_eou", "action": "press"},
        {"type": "button", "button": "manual_eou", "action": "release"},
        {"type": "button", "button": "keep_listening", "action": "toggle"},
        {"type": "button", "button": "mode_switch", "action": "mode_switch"},
        {"type": "button", "button": "push_to_talk", "action": "press"},
        {"type": "button", "button": "push_to_talk", "action": "release"},
    ]
    
    for cmd in commands:
        result = controller.process_websocket_command(cmd)
        print(f"Command {cmd} -> {result}")
        time.sleep(0.5)
    
    # Test state queries
    print("\n4. Testing state queries...")
    print(f"Current mode: {controller.get_mode()}")
    for button in controller.button_states:
        state = controller.get_button_state(button)
        print(f"Button {button}: {state}")
    
    print("\nButtonController test completed!")

if __name__ == "__main__":
    test_button_controller()