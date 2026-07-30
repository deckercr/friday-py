from unittest.mock import MagicMock, patch

import numpy as np

from app.stt import SpeechToText


def _fake_segment(text: str):
    segment = MagicMock()
    segment.text = text
    return segment


@patch("app.stt.WhisperModel")
def test_transcribe_joins_segment_texts(mock_whisper_model_cls):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        [_fake_segment(" hello "), _fake_segment("world ")],
        MagicMock(),
    )
    mock_whisper_model_cls.return_value = mock_model

    stt = SpeechToText(model_size="tiny", device="cpu", compute_type="int8")
    result = stt.transcribe(np.zeros(16000, dtype=np.float32))

    assert result == "hello world"
    mock_whisper_model_cls.assert_called_once_with(
        "tiny", device="cpu", compute_type="int8"
    )
