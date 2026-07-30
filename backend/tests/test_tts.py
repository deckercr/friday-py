from unittest.mock import MagicMock, patch

from app.tts import TextToSpeech


def _fake_chunk(data: bytes):
    chunk = MagicMock()
    chunk.audio_int16_bytes = data
    return chunk


@patch("app.tts.PiperVoice")
def test_synthesize_chunks_returns_audio_bytes(mock_piper_voice_cls):
    mock_voice = MagicMock()
    mock_voice.synthesize.return_value = [_fake_chunk(b"\x01\x02"), _fake_chunk(b"\x03\x04")]
    mock_piper_voice_cls.load.return_value = mock_voice

    tts = TextToSpeech(model_path="fake.onnx")
    chunks = tts.synthesize_chunks("hello")

    assert chunks == [b"\x01\x02", b"\x03\x04"]
    mock_piper_voice_cls.load.assert_called_once_with("fake.onnx")


@patch("app.tts.wave.open")
@patch("app.tts.PiperVoice")
def test_synthesize_wav_calls_voice_synthesize_wav(mock_piper_voice_cls, mock_wave_open):
    mock_voice = MagicMock()
    mock_piper_voice_cls.load.return_value = mock_voice
    mock_wav_file = MagicMock()
    mock_wave_open.return_value.__enter__.return_value = mock_wav_file

    tts = TextToSpeech(model_path="fake.onnx")
    tts.synthesize_wav("hello", "out.wav")

    assert mock_voice.synthesize_wav.call_count == 1
    call_args = mock_voice.synthesize_wav.call_args
    assert call_args.args[0] == "hello"
