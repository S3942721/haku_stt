import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="stt_en_fastconformer_hybrid_large_streaming_multi")

python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py \
  pretrained_name="stt_en_fastconformer_hybrid_large_streaming_multi" \
  audio_dir="test.wav"