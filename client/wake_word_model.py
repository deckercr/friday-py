"""
Wake word model wrapper around openwakeword.

Provides a simple interface for scoring audio chunks for wake-word detection.
Exposes the score(chunk: bytes) -> float interface expected by wake_listener.py.
"""

import numpy as np
import openwakeword


class WakeWordModel:
    """
    Wraps openwakeword.Model to provide a simple score() interface for chunks.

    score(chunk: bytes) -> float returns the maximum detection score across all
    loaded wake-word models (0.0 = no detection, 1.0 = definite detection).
    """

    def __init__(self, models=None):
        """
        Initialize the wake-word model.

        Args:
            models: List of model names (e.g., ["alexa", "jarvis"]).
                    If None or empty, loads all pre-trained models.
        """
        if models is None:
            models = []
        self._model = openwakeword.Model(wakeword_models=models)

    def score(self, chunk: bytes) -> float:
        """
        Score a chunk of raw int16 mono PCM audio for wake-word presence.

        Args:
            chunk: Raw audio bytes (int16, 16 kHz, mono). Expected to be ~1280 samples (80ms).

        Returns:
            float: Maximum detection score across all models (0.0 to 1.0).
                   0.0 = no wake word detected, 1.0 = definite detection.
        """
        if not chunk:
            return 0.0

        # Convert bytes to numpy int16 array
        samples = np.frombuffer(chunk, dtype=np.int16)
        if samples.size == 0:
            return 0.0

        # Get predictions from openwakeword
        predictions = self._model.predict(samples, timing=False)

        # Return max score across all models
        if not predictions:
            return 0.0

        return float(max(predictions.values()))
