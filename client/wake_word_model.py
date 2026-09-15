"""
Thin adapter wrapping a single openWakeWord phrase model so it satisfies
the score(chunk: bytes) -> float interface WakeWordListener expects.
Loads one openWakeWord Model per phrase, each scoped to exactly one
registered wakeword model - this trades a small amount of redundant
shared feature-extraction computation (negligible at this chunk rate and
model size) for a simple, independently testable interface, rather than
sharing one Model instance across multiple registered phrases.
"""
import os

import numpy as np
from openwakeword.model import Model

# openwakeword's wheel doesn't bundle its shared feature-extraction models
# (melspectrogram.onnx / embedding_model.onnx) - by default it fetches them
# over the network into its own package directory on first use. Committed
# copies ship in client/models/ instead (same directory as our trained
# phrase models) and are passed explicitly below, so loading a wake-word
# model never requires network or write access, in any environment.
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
_MELSPEC_MODEL_PATH = os.path.join(_MODELS_DIR, "melspectrogram.onnx")
_EMBEDDING_MODEL_PATH = os.path.join(_MODELS_DIR, "embedding_model.onnx")


class OpenWakeWordModel:
    def __init__(self, model_path: str, model_name: str):
        self._model = Model(
            wakeword_models=[model_path],
            inference_framework="onnx",
            melspec_model_path=_MELSPEC_MODEL_PATH,
            embedding_model_path=_EMBEDDING_MODEL_PATH,
        )
        self._model_name = model_name

    def score(self, chunk: bytes) -> float:
        samples = np.frombuffer(chunk, dtype=np.int16)
        prediction = self._model.predict(samples)
        return float(prediction[self._model_name])

    def reset(self) -> None:
        # Clears openWakeWord's internal rolling feature buffer so a stale
        # buffer from the previous utterance can't immediately re-trigger
        # once listening resumes.
        self._model.reset()
