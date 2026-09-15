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
    mock_model_cls.assert_called_once_with(
        wakeword_models=["models/hey_friday.onnx"], inference_framework="onnx"
    )
    mock_model.predict.assert_called_once()
    called_samples = mock_model.predict.call_args[0][0]
    assert isinstance(called_samples, np.ndarray)
    assert called_samples.dtype == np.int16
