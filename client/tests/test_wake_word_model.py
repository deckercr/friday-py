"""Tests for WakeWordModel, which wraps openwakeword for scoring."""
from unittest.mock import Mock, patch
import numpy as np

from wake_word_model import WakeWordModel

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # 80ms at 16kHz


def _chunk(amplitude: int, num_samples: int = CHUNK_SAMPLES) -> bytes:
    """A synthetic int16 mono chunk of constant amplitude."""
    return np.full(num_samples, amplitude, dtype=np.int16).tobytes()


@patch("wake_word_model.openwakeword.Model")
def test_wake_word_model_initializes_with_model_names(mock_model_class):
    """WakeWordModel accepts a list of model names (e.g., 'alexa', 'jarvis')."""
    mock_model_class.return_value = Mock()
    model = WakeWordModel(models=["alexa", "jarvis"])
    assert model is not None
    mock_model_class.assert_called_once_with(wakeword_models=["alexa", "jarvis"])


@patch("wake_word_model.openwakeword.Model")
def test_wake_word_model_score_accepts_chunk_bytes(mock_model_class):
    """score() accepts int16 mono PCM bytes and returns a float."""
    mock_instance = Mock()
    mock_instance.predict.return_value = {"alexa": 0.7}
    mock_model_class.return_value = mock_instance

    model = WakeWordModel(models=["alexa"])
    chunk = _chunk(1000)
    score = model.score(chunk)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score == 0.7


@patch("wake_word_model.openwakeword.Model")
def test_wake_word_model_returns_max_score_across_all_models(mock_model_class):
    """If multiple models are provided, score() returns the max across them."""
    mock_instance = Mock()
    mock_instance.predict.return_value = {"alexa": 0.3, "jarvis": 0.6}
    mock_model_class.return_value = mock_instance

    model = WakeWordModel(models=["alexa", "jarvis"])
    silent_chunk = _chunk(0)
    score = model.score(silent_chunk)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score == 0.6  # Should be max of the two


@patch("wake_word_model.openwakeword.Model")
def test_wake_word_model_handles_empty_chunk(mock_model_class):
    """score() should handle empty or zero-length chunks gracefully."""
    mock_instance = Mock()
    mock_model_class.return_value = mock_instance

    model = WakeWordModel(models=["alexa"])
    empty_chunk = b""
    score = model.score(empty_chunk)
    # Should not raise; score should be a valid float
    assert isinstance(score, float)
    assert score == 0.0
    # predict should not have been called for empty chunk
    mock_instance.predict.assert_not_called()


@patch("wake_word_model.openwakeword.Model")
def test_wake_word_model_single_model(mock_model_class):
    """WakeWordModel works with a single model name."""
    mock_instance = Mock()
    mock_instance.predict.return_value = {"alexa": 0.5}
    mock_model_class.return_value = mock_instance

    model = WakeWordModel(models=["alexa"])
    chunk = _chunk(500)
    score = model.score(chunk)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score == 0.5


@patch("wake_word_model.openwakeword.Model")
def test_wake_word_model_with_multiple_sequential_chunks(mock_model_class):
    """score() accumulates state across multiple chunk calls (openwakeword expects this)."""
    mock_instance = Mock()
    mock_instance.predict.side_effect = [{"alexa": 0.2}, {"alexa": 0.8}]
    mock_model_class.return_value = mock_instance

    model = WakeWordModel(models=["alexa"])
    chunk1 = _chunk(100)
    chunk2 = _chunk(200)

    score1 = model.score(chunk1)
    score2 = model.score(chunk2)

    assert isinstance(score1, float)
    assert isinstance(score2, float)
    assert 0.0 <= score1 <= 1.0
    assert 0.0 <= score2 <= 1.0
    assert score1 == 0.2
    assert score2 == 0.8
