"""
Thin adapter wrapping a single openWakeWord phrase model so it satisfies
the score(chunk: bytes) -> float interface WakeWordListener expects.
Loads one openWakeWord Model per phrase, each scoped to exactly one
registered wakeword model - this trades a small amount of redundant
shared feature-extraction computation (negligible at this chunk rate and
model size) for a simple, independently testable interface, rather than
sharing one Model instance across multiple registered phrases.
"""
import numpy as np
from openwakeword.model import Model
from openwakeword.utils import download_models


class OpenWakeWordModel:
    def __init__(self, model_path: str, model_name: str):
        # openwakeword's wheel doesn't bundle its shared feature-extraction
        # models (melspectrogram.onnx / embedding_model.onnx); download_models
        # fetches them if missing and no-ops if already present. `[]` means
        # "no phrase-specific models to download here" - only the shared
        # backbone that every phrase model depends on.
        download_models([])
        self._model = Model(wakeword_models=[model_path], inference_framework="onnx")
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
