import os
from unittest.mock import MagicMock, patch

import numpy as np

from wake_word_model import OpenWakeWordModel


@patch("wake_word_model.Model")
def test_score_delegates_to_openwakeword_prediction(mock_model_cls):
    mock_model = MagicMock()
    mock_model.predict.return_value = {"hey_friday": 0.87}
    mock_model_cls.return_value = mock_model

    adapter = OpenWakeWordModel(model_path="models/hey_friday.onnx", model_name="hey_friday")
    chunk = np.zeros(1280, dtype=np.int16).tobytes()

    score = adapter.score(chunk)

    assert score == 0.87
    mock_model_cls.assert_called_once()
    _, kwargs = mock_model_cls.call_args
    assert kwargs["wakeword_models"] == ["models/hey_friday.onnx"]
    assert kwargs["inference_framework"] == "onnx"
    # Explicit local paths - never left to openwakeword's own network/write
    # access fallback (see the wake_word_model.py module docstring).
    assert kwargs["melspec_model_path"].endswith(os.path.join("models", "melspectrogram.onnx"))
    assert kwargs["embedding_model_path"].endswith(os.path.join("models", "embedding_model.onnx"))
    assert os.path.exists(kwargs["melspec_model_path"])
    assert os.path.exists(kwargs["embedding_model_path"])
    mock_model.predict.assert_called_once()
    called_samples = mock_model.predict.call_args[0][0]
    assert isinstance(called_samples, np.ndarray)
    assert called_samples.dtype == np.int16
