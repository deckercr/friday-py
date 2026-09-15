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
        # vad_filter strips silence/non-speech before transcribing, instead
        # of feeding the whole buffer (including any trailing silence from
        # how long a push-to-talk press was held) straight to the model.
        # condition_on_previous_text=False stops Whisper's known repetition-
        # loop failure mode, where it re-uses prior output as context and
        # spirals into repeating itself on longer or noisier clips.
        segments, _ = self._model.transcribe(
            audio, vad_filter=True, condition_on_previous_text=False
        )
        return " ".join(segment.text.strip() for segment in segments)
