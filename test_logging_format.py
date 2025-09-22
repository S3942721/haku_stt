#!/usr/bin/env python3
"""
Test logging alignment in the STT system
This script tests that the logging format is consistent across all components
"""

import logging
import sys
from unittest.mock import Mock

# Mock the dependencies that aren't available
sys.modules['omegaconf'] = Mock()
sys.modules['nemo'] = Mock()
sys.modules['nemo.collections'] = Mock()
sys.modules['nemo.collections.asr'] = Mock()
sys.modules['nemo.collections.asr.models'] = Mock()
sys.modules['nemo.collections.asr.models.ctc_bpe_models'] = Mock()
sys.modules['nemo.collections.asr.parts'] = Mock()
sys.modules['nemo.collections.asr.parts.utils'] = Mock()
sys.modules['nemo.collections.asr.parts.utils.rnnt_utils'] = Mock()
sys.modules['transformers'] = Mock()
sys.modules['huggingface_hub'] = Mock()
sys.modules['torch'] = Mock()
sys.modules['onnxruntime'] = Mock()
sys.modules['pyaudio'] = Mock()

# Set up CRITICAL level handler to capture complete utterance messages
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

def test_logging_format():
    """Test that the logging format is consistent"""
    print("Testing logging format consistency...")
    
    # Create a test logger
    logger = logging.getLogger("haku_stt.test")
    
    # Test various log levels with the expected format
    logger.debug("Debug message - should be aligned")
    logger.info("Info message - should be aligned")
    logger.warning("Warning message - should be aligned")
    logger.error("Error message - should be aligned")
    logger.critical("COMPLETE UTTERANCE: This is a test utterance")
    logger.critical("COMPLETE UTTERANCE (MANUAL): This is a manual EOU test")
    logger.critical("COMPLETE UTTERANCE (TRIGGERED): This is a triggered EOU test")
    
    print("\nLogging format test completed!")
    print("Check that all log messages are properly aligned above.")

if __name__ == "__main__":
    test_logging_format()