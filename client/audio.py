import threading

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024


class AudioRecorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self._sample_rate = sample_rate
        self._channels = channels
        self._frames = []
        self._stream = None
        self._stop_event = None
        self._thread = None

    def start(self) -> None:
        self._frames = []
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
        )
        self._stream.start()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            data, _overflowed = self._stream.read(CHUNK_SIZE)
            self._frames.append(data.copy())

    def stop(self) -> bytes:
        self._stop_event.set()
        self._thread.join()
        self._stream.stop()
        self._stream.close()
        if not self._frames:
            return b""
        audio = np.concatenate(self._frames, axis=0)
        return audio.tobytes()


def play(audio_bytes: bytes, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS) -> None:
    audio = np.frombuffer(audio_bytes, dtype="int16").reshape(-1, channels)
    stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
    )
    stream.start()
    stream.write(audio)
    stream.stop()
    stream.close()
