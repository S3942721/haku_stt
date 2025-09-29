import torch
import numpy as np
import nemo.collections.asr as nemo_asr
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

def test_vad_asr(audio_file_path="test.wav"):
    """Test VAD and ASR on a stored audio file"""
    
    print("="*60)
    print("TESTING VAD AND ASR ON STORED AUDIO FILE")
    print("="*60)
    
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
    
    # ------------------------------
    # 3. Load and process audio file
    # ------------------------------
    print(f"\nLoading audio file: {audio_file_path}")
    try:
        # Load audio file
        audio_data, sample_rate = sf.read(audio_file_path)
        print(f"Original sample rate: {sample_rate} Hz")
        print(f"Original audio shape: {audio_data.shape}")
        print(f"Audio duration: {len(audio_data) / sample_rate:.2f} seconds")
        
        # Convert to mono if stereo
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
            print("Converted stereo to mono")
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            print(f"Resampling from {sample_rate}Hz to 16000Hz...")
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Normalize audio
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize to [-1, 1] range
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        print(f"Processed audio shape: {audio_data.shape}")
        print(f"Audio range: [{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
        
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return False
    
    # ------------------------------
    # 4. Test direct ASR transcription
    # ------------------------------
    print(f"\n{'='*30}")
    print("TESTING DIRECT ASR TRANSCRIPTION")
    print(f"{'='*30}")
    
    try:
        # Test ASR on full audio
        print("Running ASR on full audio...")
        full_result = asr_model.transcribe([audio_data])[0]
        
        # Extract text from result (handle both string and Hypothesis object)
        if hasattr(full_result, 'text'):
            full_transcription = full_result.text
        elif isinstance(full_result, str):
            full_transcription = full_result
        else:
            full_transcription = str(full_result)
            
        print(f"Full transcription: '{full_transcription}'")
        
        if not full_transcription or full_transcription.strip() == "":
            print("WARNING: ASR returned empty transcription for full audio!")
        else:
            print("✓ ASR working - got non-empty transcription")
            
    except Exception as e:
        print(f"ERROR: ASR failed on full audio: {e}")
        return False
    
    # ------------------------------
    # 5. Test VAD speech detection
    # ------------------------------
    print(f"\n{'='*30}")
    print("TESTING VAD SPEECH DETECTION")
    print(f"{'='*30}")
    
    try:
        # Use Silero VAD to get speech timestamps
        audio_tensor = torch.from_numpy(audio_data).float()
        speech_timestamps = get_speech_timestamps(audio_tensor, model, sampling_rate=16000)
        
        print(f"VAD detected {len(speech_timestamps)} speech segments:")
        for i, ts in enumerate(speech_timestamps):
            start_sec = ts['start'] / 16000
            end_sec = ts['end'] / 16000
            duration = end_sec - start_sec
            print(f"  Segment {i+1}: {start_sec:.2f}s - {end_sec:.2f}s (duration: {duration:.2f}s)")
            
            # Extract and transcribe each segment
            segment_audio = audio_data[ts['start']:ts['end']]
            if len(segment_audio) > 0:
                try:
                    segment_result = asr_model.transcribe([segment_audio])[0]
                    # Extract text from result
                    if hasattr(segment_result, 'text'):
                        segment_text = segment_result.text
                    elif isinstance(segment_result, str):
                        segment_text = segment_result
                    else:
                        segment_text = str(segment_result)
                    print(f"    Transcription: '{segment_text}'")
                except Exception as e:
                    print(f"    Transcription error: {e}")
        
        if not speech_timestamps:
            print("WARNING: VAD detected no speech in the audio!")
            print("This might explain why live transcription isn't working.")
            
            # Test with different VAD sensitivity
            print("\nTrying with more sensitive VAD settings...")
            speech_timestamps = get_speech_timestamps(
                audio_tensor, model, 
                sampling_rate=16000,
                threshold=0.3,  # Lower threshold (more sensitive)
                min_speech_duration_ms=100,  # Shorter minimum duration
                min_silence_duration_ms=100   # Shorter silence duration
            )
            print(f"With sensitive settings, VAD detected {len(speech_timestamps)} speech segments")
            for i, ts in enumerate(speech_timestamps):
                start_sec = ts['start'] / 16000
                end_sec = ts['end'] / 16000
                duration = end_sec - start_sec
                print(f"  Segment {i+1}: {start_sec:.2f}s - {end_sec:.2f}s (duration: {duration:.2f}s)")
            
        else:
            print("✓ VAD working - detected speech segments")
            
    except Exception as e:
        print(f"ERROR: VAD failed: {e}")
        return False
    
    # ------------------------------
    # 6. Test VAD + ASR pipeline (like live stream)
    # ------------------------------
    print(f"\n{'='*30}")
    print("TESTING VAD + ASR PIPELINE")
    print(f"{'='*30}")
    
    try:
        vad_iterator = VADIterator(model)
        buffer = []
        frame_size = 512  # 512 samples for 16kHz
        
        print("Processing audio in frames like live stream...")
        speech_segments_found = 0
        
        for start in range(0, len(audio_data), frame_size):
            frame = audio_data[start:start + frame_size]
            
            # Pad if necessary
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)), 'constant')
            
            frame_tensor = torch.from_numpy(frame).float()
            speech_detected = vad_iterator(frame_tensor, return_seconds=False)
            
            if speech_detected:
                buffer.append(frame.copy())
            else:
                if buffer:
                    # End of speech segment
                    segment_audio = np.concatenate(buffer)
                    buffer = []
                    
                    duration = len(segment_audio) / 16000
                    print(f"Speech segment {speech_segments_found + 1}: {duration:.2f}s")
                    
                    if duration >= 0.3:  # Same minimum as live stream
                        try:
                            segment_result = asr_model.transcribe([segment_audio])[0]
                            # Extract text from result
                            if hasattr(segment_result, 'text'):
                                segment_text = segment_result.text
                            elif isinstance(segment_result, str):
                                segment_text = segment_result
                            else:
                                segment_text = str(segment_result)
                            print(f"  Transcription: '{segment_text}'")
                            speech_segments_found += 1
                        except Exception as e:
                            print(f"  ASR error: {e}")
                    else:
                        print(f"  Skipped (too short)")
        
        # Handle final buffer
        if buffer:
            segment_audio = np.concatenate(buffer)
            duration = len(segment_audio) / 16000
            print(f"Final speech segment: {duration:.2f}s")
            if duration >= 0.3:
                try:
                    segment_result = asr_model.transcribe([segment_audio])[0]
                    # Extract text from result
                    if hasattr(segment_result, 'text'):
                        segment_text = segment_result.text
                    elif isinstance(segment_result, str):
                        segment_text = segment_result
                    else:
                        segment_text = str(segment_result)
                    print(f"  Transcription: '{segment_text}'")
                    speech_segments_found += 1
                except Exception as e:
                    print(f"  ASR error: {e}")
        
        print(f"\nTotal speech segments processed: {speech_segments_found}")
        
        if speech_segments_found == 0:
            print("❌ PROBLEM: No speech segments were processed by the pipeline!")
            print("This explains why live transcription isn't working.")
        else:
            print("✓ Pipeline working - processed speech segments")
            
    except Exception as e:
        print(f"ERROR: Pipeline test failed: {e}")
        return False
    
    # ------------------------------
    # 7. Audio quality analysis
    # ------------------------------
    print(f"\n{'='*30}")
    print("AUDIO QUALITY ANALYSIS")
    print(f"{'='*30}")
    
    # Check audio levels
    rms = np.sqrt(np.mean(audio_data**2))
    peak = np.max(np.abs(audio_data))
    
    print(f"RMS level: {rms:.4f}")
    print(f"Peak level: {peak:.4f}")
    print(f"Dynamic range: {20 * np.log10(peak/rms) if rms > 0 else 0:.1f} dB")
    
    if rms < 0.01:
        print("WARNING: Audio level very low - might affect VAD sensitivity")
    elif rms > 0.5:
        print("WARNING: Audio level very high - might be clipped")
    else:
        print("✓ Audio levels look reasonable")
    
    # Check for silence
    silence_threshold = 0.001
    silence_samples = np.sum(np.abs(audio_data) < silence_threshold)
    silence_percentage = (silence_samples / len(audio_data)) * 100
    
    print(f"Silence percentage: {silence_percentage:.1f}%")
    if silence_percentage > 80:
        print("WARNING: Audio is mostly silent")
    
    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}")
    
    return True

if __name__ == "__main__":
    import sys
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "test.wav"
    test_vad_asr(audio_file)
