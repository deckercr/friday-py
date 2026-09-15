"""
Microphone capture and speaker playback, at the sample format this project
uses everywhere: 16kHz mono int16 PCM.
"""
import queue

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024


def listen_chunks(chunk_samples: int = CHUNK_SIZE, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
    """
    Yields raw int16 mono PCM chunks from the default input device,
    chunk_samples samples at a time, forever. Replaces the old
    AudioRecorder start()/stop() pattern used for push-to-talk - the
    caller now drives how long to keep pulling from this generator
    (e.g. `for chunk in listen_chunks(): ...`), since wake-word listening
    needs the mic open continuously rather than only while a key is held.
    """
    chunk_queue: "queue.Queue[bytes]" = queue.Queue()

    def _callback(indata, frames, time_info, status):
        chunk_queue.put(bytes(indata))

    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        blocksize=chunk_samples,
        callback=_callback,
    ):
        while True:
            yield chunk_queue.get()


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
