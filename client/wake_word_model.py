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
import openwakeword
from openwakeword.model import Model
from openwakeword.utils import download_file

_FEATURE_MODELS_DIR = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")


def _ensure_feature_models() -> None:
    # openwakeword's wheel doesn't bundle its shared feature-extraction
    # models (melspectrogram.onnx / embedding_model.onnx); fetch them if
    # missing, no-op if already present. Deliberately NOT using
    # openwakeword.utils.download_models() here: passing it an empty list
    # doesn't mean "just the shared backbone" as its docstring's phrasing
    # suggests - it means "no filter", which downloads every official
    # bundled phrase model too (unwanted: we supply our own custom models,
    # and this would require network/write access on every startup even in
    # environments that shouldn't need it). VAD models are skipped too -
    # our Model(...) call below never sets vad_threshold, so it stays at
    # the library's default of 0 (disabled), and downloading them would be
    # dead weight.
    if not os.path.exists(_FEATURE_MODELS_DIR):
        os.makedirs(_FEATURE_MODELS_DIR)
    for feature_model in openwakeword.FEATURE_MODELS.values():
        tflite_name = feature_model["download_url"].split("/")[-1]
        if not os.path.exists(os.path.join(_FEATURE_MODELS_DIR, tflite_name)):
            download_file(feature_model["download_url"], _FEATURE_MODELS_DIR)
            download_file(feature_model["download_url"].replace(".tflite", ".onnx"), _FEATURE_MODELS_DIR)


class OpenWakeWordModel:
    def __init__(self, model_path: str, model_name: str):
        _ensure_feature_models()
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
