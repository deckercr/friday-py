"""
Wake-word listening state machine: LISTENING -> RECORDING -> SENDING -> LISTENING.

Pure logic, no audio hardware or network access - process_chunk() is fed
raw int16 mono PCM chunks by the caller (a real microphone stream in
production, synthetic chunks in tests). wake_models are objects exposing
score(chunk: bytes) -> float; a chunk triggers recording if any model's
score meets wake_threshold. See wake_word_model.py for the real
openWakeWord-backed implementation of that interface.
"""
import collections

import numpy as np

LISTENING = "listening"
RECORDING = "recording"
SENDING = "sending"

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # ~80ms at 16kHz - openWakeWord's expected chunk size

WAKE_THRESHOLD = 0.5
SILENCE_RMS_THRESHOLD = 300
SILENCE_HANGOVER_SECONDS = 1.0
MAX_UTTERANCE_SECONDS = 20.0
MIN_UTTERANCE_SECONDS = 0.5
PRE_ROLL_SECONDS = 0.5


class WakeWordListener:
    def __init__(
        self,
        wake_models,
        on_utterance_ready,
        on_state_change=None,
        wake_threshold: float = WAKE_THRESHOLD,
        silence_rms_threshold: float = SILENCE_RMS_THRESHOLD,
        silence_hangover_seconds: float = SILENCE_HANGOVER_SECONDS,
        max_utterance_seconds: float = MAX_UTTERANCE_SECONDS,
        min_utterance_seconds: float = MIN_UTTERANCE_SECONDS,
        pre_roll_seconds: float = PRE_ROLL_SECONDS,
        sample_rate: int = SAMPLE_RATE,
        chunk_samples: int = CHUNK_SAMPLES,
    ):
        self._wake_models = wake_models
        self._on_utterance_ready = on_utterance_ready
        self._on_state_change = on_state_change or (lambda state: None)
        self._wake_threshold = wake_threshold
        self._silence_rms_threshold = silence_rms_threshold
        self._silence_hangover_seconds = silence_hangover_seconds
        self._max_utterance_seconds = max_utterance_seconds
        self._min_utterance_seconds = min_utterance_seconds
        self._chunk_seconds = chunk_samples / sample_rate

        pre_roll_len = max(1, round(pre_roll_seconds / self._chunk_seconds))
        self._pre_roll = collections.deque(maxlen=pre_roll_len)

        self._state = LISTENING
        self._utterance_chunks = []
        self._silence_seconds = 0.0
        self._recording_seconds = 0.0
        self._voiced_seconds = 0.0

    @property
    def state(self) -> str:
        return self._state

    def mark_sending_finished(self) -> None:
        self._reset_wake_models()
        self._set_state(LISTENING)

    def process_chunk(self, chunk: bytes) -> None:
        if self._state == SENDING:
            return
        if self._state == LISTENING:
            self._pre_roll.append(chunk)
            if any(model.score(chunk) >= self._wake_threshold for model in self._wake_models):
                self._start_recording()
            return
        self._append_and_check_silence(chunk)

    def _start_recording(self) -> None:
        self._utterance_chunks = list(self._pre_roll)
        self._pre_roll.clear()
        self._recording_seconds = len(self._utterance_chunks) * self._chunk_seconds
        self._silence_seconds = 0.0
        # Pre-roll (which includes the wake-word trigger chunk itself, since
        # it was appended to the pre-roll buffer before the trigger check
        # ran) is typically near-silent room tone plus the wake word, not
        # follow-up speech - treated as unvoiced for a simple default.
        # Voiced time is counted only from chunks seen from here on, via
        # _append_and_check_silence.
        self._voiced_seconds = 0.0
        self._set_state(RECORDING)
        if self._recording_seconds >= self._max_utterance_seconds:
            self._finish_utterance()

    def _append_and_check_silence(self, chunk: bytes) -> None:
        self._utterance_chunks.append(chunk)
        self._recording_seconds += self._chunk_seconds

        samples = np.frombuffer(chunk, dtype=np.int16)
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if samples.size else 0.0
        if rms < self._silence_rms_threshold:
            self._silence_seconds += self._chunk_seconds
        else:
            self._silence_seconds = 0.0
            self._voiced_seconds += self._chunk_seconds

        if (
            self._silence_seconds >= self._silence_hangover_seconds
            or self._recording_seconds >= self._max_utterance_seconds
        ):
            self._finish_utterance()

    def _finish_utterance(self) -> None:
        audio_bytes = b"".join(self._utterance_chunks)
        self._utterance_chunks = []
        if self._voiced_seconds < self._min_utterance_seconds:
            self._reset_wake_models()
            self._set_state(LISTENING)
            return
        self._set_state(SENDING)
        self._on_utterance_ready(audio_bytes)

    def _reset_wake_models(self) -> None:
        for model in self._wake_models:
            if hasattr(model, "reset"):
                model.reset()

    def _set_state(self, new_state: str) -> None:
        self._state = new_state
        self._on_state_change(new_state)
