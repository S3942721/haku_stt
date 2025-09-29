import redis
import threading
import time
import torch
import copy
import numpy as np
from datetime import datetime
from logging import Logger
from utils import get_logger
from omegaconf import OmegaConf, open_dict
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE

class Listener:

    def __init__(self, logger: Logger):

        self.logger = logger
        self.init_model()
        self.init_redis()
        self.init_preprocessor()
        self.run()

    def init_model(self):
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name='stt_en_fastconformer_hybrid_large_streaming_multi')
        self.lookahead_size = 80
        self.encoder_step_length = 80
        self.left_context_size = self.asr_model.encoder.att_context_size[0]
        self.asr_model.encoder.set_default_att_context_size([self.left_context_size, int(self.lookahead_size / self.encoder_step_length)])
        self.asr_model.change_decoding_strategy(decoder_type='rnnt')
        self.decoding_cfg = self.asr_model.cfg.decoding
        self.set_decoding_strategy()
        self.asr_model = self.asr_model.to('cuda')
        self.asr_model.eval()
        self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = self.asr_model.encoder.get_initial_cache_state(batch_size=1)
        self.previous_hypotheses = None
        self.pred_out_stream = None
        self.step_num = 0
        self.pre_encode_cache_size = self.asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
        self.num_channels = self.asr_model.cfg.preprocessor.features
        self.cache_pre_encode = torch.zeros((1, self.num_channels, self.pre_encode_cache_size), device=self.asr_model.device)

    def init_redis(self):
        self.redis_client = redis.StrictRedis(host='127.0.0.1', port='6379')
        self.redis_user_audio_queue = 'user_audio'
        self.redis_client.set("is_user_speaking", 0)

    def init_preprocessor(self):
        cfg = copy.deepcopy(self.asr_model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)

        # some changes for streaming scenario
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        
        self.preprocessor = EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
        self.preprocessor.to(self.asr_model.device)
        
    def set_decoding_strategy(self):
        with open_dict(self.decoding_cfg):
            self.decoding_cfg.strategy = "greedy"
            self.decoding_cfg.preserve_alignments = False
            if hasattr(self.asr_model, 'joint'):  # if an RNNT model
                self.decoding_cfg.greedy.max_symbols = 10
                self.decoding_cfg.fused_batch_size = -1
            self.asr_model.change_decoding_strategy(self.decoding_cfg)

    def preprocess_audio(self, audio):
        audio = np.frombuffer(audio, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
        audio = np.clip(audio, -1.0, 1.0)
        device = self.asr_model.device
        audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(device)
        audio_signal_len = torch.Tensor([audio.shape[0]]).to(device)
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=audio_signal, length=audio_signal_len
        )
        return processed_signal, processed_signal_length

    def transcribe(self, audio):
        processed_signal, processed_signal_length = self.preprocess_audio(audio)
        processed_signal = torch.cat([self.cache_pre_encode, processed_signal], dim=-1)
        processed_signal_length += self.cache_pre_encode.shape[1]
        self.cache_pre_encode = processed_signal[:, :, -self.pre_encode_cache_size:]
        with torch.no_grad():
            (
                self.pred_out_stream,
                transcribed_texts,
                self.cache_last_channel,
                self.cache_last_time,
                self.cache_last_channel_len,
                self.previous_hypotheses,
            ) = self.asr_model.conformer_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=self.cache_last_channel,
                cache_last_time=self.cache_last_time,
                cache_last_channel_len=self.cache_last_channel_len,
                keep_all_outputs=False,
                previous_hypotheses=self.previous_hypotheses,
                previous_pred_out=self.pred_out_stream,
                drop_extra_pre_encoded=None,
                return_transcription=True,
            )
        
        self.logger.info(transcribed_texts[0].text)
        self.logger.info(len(transcribed_texts))
        self.step_num += 1

    def get_audio_from_redis(self):
        _, audio_bytes = self.redis_client.brpop(self.redis_user_audio_queue, timeout=0)
        return audio_bytes
    
    def run(self):
        
        self.logger.info(self.__class__.__name__ + " Running")
        audio_bytes = b''
        while True:
            # I have tried not resetting audio bytes as well, still didn't work
            audio_bytes = b''
            for _ in range(8):
                audio_bytes += self.get_audio_from_redis()
            self.transcribe(audio_bytes)
            

if __name__ == '__main__':

    logger = get_logger(__file__)
    try:

        Listener(logger)
    except BaseException as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        raise