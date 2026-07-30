import struct
import wave

import pytest

from scripts.pipeline_check import load_wav_as_float32


def _write_wav(path, sample_width: int, framerate: int = 16000, channels: int = 1):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(framerate)
        if sample_width == 1:
            frames = bytes([128, 129, 130, 131])
        elif sample_width == 2:
            frames = struct.pack("<4h", 0, 100, -100, 200)
        elif sample_width == 4:
            frames = struct.pack("<4i", 0, 1000, -1000, 2000)
        else:
            raise ValueError("unsupported sample width in test helper")
        wav_file.writeframes(frames)


def test_load_wav_as_float32_rejects_non_16bit_pcm(tmp_path):
    path = tmp_path / "eight_bit.wav"
    _write_wav(path, sample_width=1)

    with pytest.raises(ValueError):
        load_wav_as_float32(str(path))


def test_load_wav_as_float32_rejects_32bit_pcm(tmp_path):
    path = tmp_path / "thirtytwo_bit.wav"
    _write_wav(path, sample_width=4)

    with pytest.raises(ValueError):
        load_wav_as_float32(str(path))


def test_load_wav_as_float32_accepts_16bit_pcm(tmp_path):
    path = tmp_path / "sixteen_bit.wav"
    _write_wav(path, sample_width=2)

    audio = load_wav_as_float32(str(path))

    assert audio.size > 0
