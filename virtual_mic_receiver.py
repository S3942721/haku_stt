import queue
import torch
import sounddevice as sd
import numpy as np
import nemo.collections.asr as nemo_asr
import subprocess
import time
import threading
import os
import argparse

# ------------------------------
# 1. Load ASR model (NeMo)
# ------------------------------
print("Loading NeMo ASR model...")
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
    model_name="stt_en_conformer_transducer_small"
)
asr_model.eval()
asr_model.cfg.preprocessor.dither = 0.0
asr_model.cfg.preprocessor.pad_to = 0

# ------------------------------
# 2. Load VAD model (Silero)
# ------------------------------
print("Loading Silero VAD...")
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
)
(get_speech_timestamps, save_audio, read_audio,
 VADIterator, collect_chunks) = utils
vad_iterator = VADIterator(model)

# ------------------------------
# 3. Mic setup
# ------------------------------
samplerate = 16000  # ASR model expects 16 kHz
blocksize = 8000    # 0.5s chunks
channels = 1
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print("Audio callback status:", status)
    q.put(indata.copy())

def start_virtual_mic_receiver():
    """Start the virtual microphone receiver in the background"""
    try:
        print("Starting virtual microphone receiver...")
        script_path = os.path.join(os.path.dirname(__file__), "virtual_mic_receiver.py")
        
        # Check if the script exists
        if not os.path.exists(script_path):
            print(f"Virtual mic receiver script not found at: {script_path}")
            return None
            
        process = subprocess.Popen([
            "python", script_path, "--name", "RobotMicrophone"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait longer and check if process is still running
        time.sleep(5)  # Give it more time to start
        
        if process.poll() is None:  # Process is still running
            print("Virtual microphone receiver started successfully")
            return process
        else:
            # Process terminated, check output
            stdout, stderr = process.communicate()
            print(f"Virtual mic receiver failed to start:")
            print(f"stdout: {stdout.decode()}")
            print(f"stderr: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"Failed to start virtual microphone receiver: {e}")
        return None

def find_robot_audio_device(preferred_device_name=None):
    """Find the best robot audio device or a preferred non-robot device by name"""
    devices = sd.query_devices()

    # If a preferred non-robot mic name was provided, try to use it
    if preferred_device_name:
        preferred = preferred_device_name.strip().lower()
        print(f"Trying preferred microphone name: '{preferred_device_name}'")
        # First try exact match
        for i, device in enumerate(devices):
            if device.get('max_input_channels', 0) > 0:
                if device['name'].lower() == preferred:
                    print(f"Found preferred device (exact) #{i}: {device['name']}")
                    return i, device['name']
        # Then try substring match
        for i, device in enumerate(devices):
            if device.get('max_input_channels', 0) > 0:
                if preferred in device['name'].lower():
                    print(f"Found preferred device (match) #{i}: {device['name']}")
                    return i, device['name']
        print("Preferred device not found, falling back to robot device search...")

    # Priority list of device names to look for (robot audio sources)
    robot_device_patterns = [
        "robotmicrophone.monitor",
        "robotmicrophone", 
        "pepperrobotmic.monitor",
        "pepperrobotmic",
        "robot",
        "pepper"
    ]
    
    print("Searching for robot audio devices...")
    print("Available devices:")
    for i, device in enumerate(devices):
        if device.get('max_input_channels', 0) > 0:
            print(f"  #{i}: {device['name']}")
    
    # First, try exact matches for robot devices
    for pattern in robot_device_patterns:
        for i, device in enumerate(devices):
            if device.get('max_input_channels', 0) > 0:
                device_name_lower = device['name'].lower()
                if pattern in device_name_lower:
                    print(f"Found robot device #{i}: {device['name']}")
                    return i, device['name']
    
    # If no robot device found, check if virtual mic receiver is available
    print("No robot device found. Checking for virtual microphone receiver...")
    
    # Try to start virtual mic receiver if not running
    receiver_process = start_virtual_mic_receiver()
    if receiver_process:
        # Wait longer and check again for robot devices
        print("Waiting for virtual microphone to be available...")
        for attempt in range(3):  # Try 3 times with delays
            time.sleep(3)  # Wait 3 seconds between attempts
            devices = sd.query_devices()
            print(f"Attempt {attempt + 1}: Checking for robot devices...")
            
            for pattern in robot_device_patterns:
                for i, device in enumerate(devices):
                    if device.get('max_input_channels', 0) > 0:
                        device_name_lower = device['name'].lower()
                        if pattern in device_name_lower:
                            print(f"Found robot device after starting receiver #{i}: {device['name']}")
                            return i, device['name']
        
        print("Virtual microphone receiver started but robot device not found")
        print("Available devices after starting receiver:")
        for i, device in enumerate(devices):
            if device.get('max_input_channels', 0) > 0:
                print(f"  #{i}: {device['name']}")
    
    return None, None

def test_device_compatibility(device_index):
    """Test if a device supports our required sample rate"""
    try:
        test_stream = sd.InputStream(
            device=device_index,
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            blocksize=512
        )
        test_stream.close()
        return True
    except Exception:
        return False

# ------------------------------
# 4. Main streaming loop
# ------------------------------
def main(preferred_device_name=None):
    buffer = []
    print("Listening... (Press Ctrl+C to stop)")

    # Try to find preferred or robot audio device first
    device_index, device_name = find_robot_audio_device(preferred_device_name)
    
    if device_index is not None:
        print(f"Using input device #{device_index}: {device_name}")
        
        # Test if the device supports our sample rate
        if not test_device_compatibility(device_index):
            print(f"Selected device doesn't support 16kHz, searching for alternatives...")
            device_index = None
    
    # Fallback to any compatible device if preferred/robot device not found
    if device_index is None:
        print("Searching for any compatible input device...")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device.get('max_input_channels', 0) > 0:
                if test_device_compatibility(i):
                    device_index = i
                    print(f"Found compatible input device #{device_index}: {device['name']}")
                    break
        
        if device_index is None:
            print("No compatible input device found that supports 16kHz sample rate")
            print("\nAvailable devices:")
            for i, device in enumerate(devices):
                if device.get('max_input_channels', 0) > 0:
                    print(f"  #{i}: {device['name']} (max_input_channels: {device['max_input_channels']})")
            print("\nTo use robot audio:")
            print("1. Make sure the robot is streaming audio")
            print("2. Run: python virtual_mic_receiver.py")
            print("3. Then run this script again")
            return

    # minimum duration before sending to ASR (seconds)
    min_duration_s = 0.3

    # use device_index in InputStream
    try:
        with sd.InputStream(
            device=device_index,
            samplerate=samplerate,
            channels=channels,
            dtype="float32",
            blocksize=blocksize,
            callback=callback,
        ):
            print(f"Audio stream started. Listening for speech...")
            while True:
                audio_block = q.get()  # get next audio block from mic

                # break the incoming block into frames that Silero VAD expects
                block = audio_block.squeeze()
                # ensure it's a numpy array
                if not isinstance(block, np.ndarray):
                    block = np.array(block)

                # Silero expects 512 samples for 16kHz, 256 for 8kHz
                vad_frame_size = 512 if samplerate == 16000 else 256

                # iterate over frames in this block
                for start in range(0, len(block), vad_frame_size):
                    frame_np = block[start:start + vad_frame_size]
                    # pad last frame if shorter than expected
                    if frame_np.shape[0] < vad_frame_size:
                        pad = np.zeros(vad_frame_size - frame_np.shape[0], dtype=frame_np.dtype)
                        frame_np = np.concatenate([frame_np, pad])

                    frame_tensor = torch.from_numpy(frame_np).float()

                    # Run through VAD for this frame
                    speech_detected = vad_iterator(frame_tensor, return_seconds=False)

                    if speech_detected:  # still talking
                        # store numpy frames for easier concatenation for ASR
                        buffer.append(frame_np.copy())
                        # debug: show buffered duration occasionally
                        if len(buffer) % 10 == 0:
                            total_samples = sum(b.shape[0] for b in buffer)
                            print(f"[DEBUG] Buffered {total_samples} samples (~{total_samples/samplerate:.2f}s)")
                    else:  # no speech detected -> possible EOU
                        if buffer:
                            # Concatenate buffered speech (numpy)
                            full_audio_np = np.concatenate(buffer).astype(np.float32)
                            buffer = []

                            total_duration = full_audio_np.shape[0] / float(samplerate)
                            print(f"[DEBUG] End of utterance detected. Duration: {total_duration:.2f}s, samples: {full_audio_np.shape[0]}")

                            # Skip very short segments which often produce empty hypothesis
                            if total_duration < min_duration_s:
                                print(f"[DEBUG] Skipping transcribe: segment too short (<{min_duration_s}s)")
                                continue

                            # Ensure 1D numpy float32
                            if full_audio_np.ndim > 1:
                                full_audio_np = np.squeeze(full_audio_np)

                            # Transcribe safely
                            try:
                                text = asr_model.transcribe([full_audio_np])[0]
                                print(f"\nUser: {text}\n")
                            except Exception as e:
                                print(f"[ERROR] ASR transcription failed: {e}")
                                continue
    except sd.PortAudioError as e:
        print(f"Audio device error: {e}")
        print("Robot audio setup instructions:")
        print("1. Make sure the robot is streaming audio to this computer")
        print("2. Run: python virtual_mic_receiver.py")
        print("3. Then run this script again")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Virtual mic receiver with optional non-robot mic selection")
        parser.add_argument("--mic-name", type=str, default=None,
                            help="Preferred non-robot microphone/device name (exact or substring match)")
        args = parser.parse_args()
        main(preferred_device_name=args.mic_name)
    except KeyboardInterrupt:
        print("\nStopping...")


# arecord -D default -d 5 -f cd test.wav