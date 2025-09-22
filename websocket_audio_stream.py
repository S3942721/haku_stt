#!/usr/bin/env python3
"""
WebSocket Audio Stream Receiver for ASR Integration
Accepts 16 kHz mono PCM Int16 frames over WebSocket from browser clients
and exposes a PyAudio-compatible read API for the ASR loop.
"""
import asyncio
import threading
import queue
import time
import numpy as np

try:
    import websockets
except Exception:
    websockets = None

class WebSocketAudioStream:
    def __init__(self, listen_port=8787, sample_rate=16000, verbose=False, host='0.0.0.0'):
        self.listen_port = listen_port
        self.sample_rate = sample_rate
        self.verbose = verbose
        self.host = host

        # Runtime state
        self.loop = None
        self.thread = None
        self.server = None
        self.is_running = False
        self.is_connected = False

        # Audio buffering similar to RemoteAudioStream
        self.audio_queue = queue.Queue(maxsize=100)
        self.target_chunk_size = None
        self.audio_accumulator = np.array([], dtype=np.int16)
        self.lock = threading.Lock()
        self.start_time = 0
        self.last_audio_time = 0
        self.packets_received = 0

    async def _handler(self, websocket):
        if self.verbose:
            print(f"WS-AUDIO: Client connected from {getattr(websocket, 'remote_address', 'unknown')}")
        self.is_connected = True
        try:
            async for message in websocket:
                if isinstance(message, (bytes, bytearray)):
                    # Interpret as little-endian PCM16 mono
                    audio = np.frombuffer(message, dtype=np.int16)
                    self._add_to_accumulator(audio)
                    self.packets_received += 1
                    self.last_audio_time = time.time()

                    if self.verbose and self.packets_received % 50 == 0:
                        rng = (int(np.min(audio)) if audio.size else 0, int(np.max(audio)) if audio.size else 0)
                        print(f"WS-AUDIO pkt#{self.packets_received}: samples={len(audio)}, range={rng}")
                else:
                    # Ignore text messages; optionally handle control in future
                    pass
        except websockets.exceptions.ConnectionClosed:
            if self.verbose:
                print("WS-AUDIO: Client disconnected")
        except Exception as e:
            if self.verbose:
                print(f"WS-AUDIO: Handler error: {e}")
        finally:
            self.is_connected = False

    def _add_to_accumulator(self, audio_data: np.ndarray):
        with self.lock:
            self.audio_accumulator = np.concatenate([self.audio_accumulator, audio_data])
            if self.target_chunk_size is not None:
                while len(self.audio_accumulator) >= self.target_chunk_size:
                    chunk = self.audio_accumulator[:self.target_chunk_size].copy()
                    self.audio_accumulator = self.audio_accumulator[self.target_chunk_size:]
                    try:
                        self.audio_queue.put_nowait(chunk)
                    except queue.Full:
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.put_nowait(chunk)
                        except queue.Empty:
                            pass

    def read_audio_pyaudio_compatible(self, chunk_size, timeout=1.0):
        if not self.is_running:
            return None

        if self.target_chunk_size != chunk_size:
            with self.lock:
                self.target_chunk_size = chunk_size
                # Flush existing accumulator into queue as fixed-size chunks
                while len(self.audio_accumulator) >= self.target_chunk_size:
                    chunk = self.audio_accumulator[:self.target_chunk_size].copy()
                    self.audio_accumulator = self.audio_accumulator[self.target_chunk_size:]
                    try:
                        self.audio_queue.put_nowait(chunk)
                    except queue.Full:
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.put_nowait(chunk)
                        except queue.Empty:
                            pass

        try:
            audio_data = self.audio_queue.get(timeout=timeout)
            # Ensure exact size
            if len(audio_data) != chunk_size:
                if len(audio_data) < chunk_size:
                    pad = np.zeros(chunk_size - len(audio_data), dtype=np.int16)
                    audio_data = np.concatenate([audio_data, pad])
                else:
                    audio_data = audio_data[:chunk_size]
            return audio_data
        except queue.Empty:
            if self.verbose:
                print("WS-AUDIO: Timeout waiting for audio chunk")
            return None

    def start(self):
        if self.is_running:
            return True
        if websockets is None:
            print("ERROR: websockets package not available for WebSocketAudioStream")
            return False

        def run_server():
            try:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)

                async def start_async():
                    try:
                        self.server = await websockets.serve(self._handler, self.host, self.listen_port, ping_interval=20, ping_timeout=10)
                        self.is_running = True
                        self.start_time = time.time()
                        if self.verbose:
                            print(f"WS-AUDIO: Listening on ws://{self.host}:{self.listen_port}")
                        await self.server.wait_closed()
                    except Exception as e:
                        print(f"ERROR: WS-AUDIO server startup error: {e}")
                        self.is_running = False

                self.loop.run_until_complete(start_async())
            finally:
                if self.loop and not self.loop.is_closed():
                    self.loop.close()

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        time.sleep(0.5)
        return self.is_running

    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        try:
            if self.server and self.loop:
                future = asyncio.run_coroutine_threadsafe(self.server.close(), self.loop)
                try:
                    future.result(timeout=2.0)
                except Exception:
                    pass
                self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def is_stream_active(self):
        if not self.is_running or not self.is_connected:
            return False
        return (time.time() - self.last_audio_time) < 2.0

    def get_status(self):
        return {
            'running': self.is_running,
            'connected': self.is_connected,
            'active': self.is_stream_active(),
            'packets_received': self.packets_received,
            'queue_size': self.audio_queue.qsize(),
            'sample_rate': self.sample_rate,
            'port': self.listen_port
        }

if __name__ == "__main__":
    s = WebSocketAudioStream(verbose=True)
    if not s.start():
        raise SystemExit(1)
    print("WS-AUDIO: Waiting for audio... Press Ctrl+C to stop")
    try:
        while True:
            buf = s.read_audio_pyaudio_compatible(1600, timeout=1.0)  # arbitrary test size
            if buf is not None:
                rms = np.sqrt(np.mean(buf.astype(np.float32) ** 2))
                print(f"Chunk: {len(buf)} samples, RMS={rms:.0f}, connected={s.is_connected}")
            else:
                print("No audio")
    except KeyboardInterrupt:
        pass
    finally:
        s.stop()
