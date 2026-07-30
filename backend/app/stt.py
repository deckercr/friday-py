import numpy as np
from faster_whisper import WhisperModel


class SpeechToText:
    def __init__(
        self,
        model_size: str = "small.en",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(audio)
        return " ".join(segment.text.strip() for segment in segments)
