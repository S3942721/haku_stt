import queue
import sounddevice as sd
import numpy as np
import torch

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR

# ------------------------------
# 1. Load ASR model (RNNT Conformer)
# ------------------------------
print("Loading NeMo ASR model...")
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
    model_name="stt_en_conformer_transducer_large"
)
# asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
#     model_name="stt_en_conformer_transducer_medium"
# )
asr_model.eval()

# Configure decoding strategy for EOS (end-of-utterance) detection
asr_model.change_decoding_strategy(decoding_cfg={
    "strategy": "greedy_batch",
    "compute_timestamps": True,
    "preserve_alignments": True,  # keep alignments for EOU detection
    "rnnt_end_of_utterance": True  # enable built-in EOU detector
})

# Push to GPU
if torch.cuda.is_available():
    asr_model = asr_model.to("cuda")
    print(f"ASR model moved to GPU: {torch.cuda.get_device_name()}")

# ------------------------------
# 2. Microphone setup
# ------------------------------
samplerate = 16000  # ASR model expects 16kHz
channels = 1
blocksize = 8000    # 0.5s chunks
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print("Audio callback status:", status)
    q.put(indata.copy())

# ------------------------------
# 3. Streaming helper
# ------------------------------
# FrameBatchASR handles audio framing + buffering for streaming inference
frame_asr = FrameBatchASR(asr_model, frame_len=0.25, total_buffer=4.0, batch_size=1)

# ------------------------------
# 4. Main streaming loop
# ------------------------------
def main():
    print("Listening... (Press Ctrl+C to stop)")

    devices = sd.query_devices()
    print("Available devices:")
    for i, d in enumerate(devices):
        print(f"  {i}: {d['name']}")

    with sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    ):
        buffer = []
        while True:
            audio_block = q.get().squeeze()
            if not isinstance(audio_block, np.ndarray):
                audio_block = np.array(audio_block)

            buffer.extend(audio_block.tolist())

            # Feed audio to streaming ASR helper
            while len(buffer) >= int(samplerate * 0.25):  # process in 250ms frames
                frame = np.array(buffer[: int(samplerate * 0.25)], dtype=np.float32)
                buffer = buffer[int(samplerate * 0.25):]

                print(f"Processing frame of size {len(frame)}...")

                # Send frame to ASR (pass a one-shot iterator as frame_reader)
                frame_iter = iter([frame])
                partial_hypotheses = frame_asr.transcribe(frame_buffers=frame_iter, delay=0.0)

                # Print incremental result
                if partial_hypotheses:
                    hyp = partial_hypotheses[0]["text"]
                    is_final = partial_hypotheses[0].get("final", False)
                    if hyp.strip():
                        if is_final:
                            print(f"\n[FINAL] {hyp}\n")
                        else:
                            print(f"[PARTIAL] {hyp}", end="\r")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping...")
