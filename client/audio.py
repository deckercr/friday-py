import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1


class AudioRecorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self._sample_rate = sample_rate
        self._channels = channels
        self._frames = []
        self._stream = None

    def start(self) -> None:
        self._frames = []
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time, status) -> None:
        self._frames.append(indata.copy())

    def stop(self) -> bytes:
        self._stream.stop()
        self._stream.close()
        if not self._frames:
            return b""
        audio = np.concatenate(self._frames, axis=0)
        return audio.tobytes()


def play(audio_bytes: bytes, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS) -> None:
    audio = np.frombuffer(audio_bytes, dtype="int16").reshape(-1, channels)
    sd.play(audio, samplerate=sample_rate)
    sd.wait()
