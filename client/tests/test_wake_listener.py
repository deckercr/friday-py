from wake_listener import WakeWordListener

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # 80ms


def _chunk(amplitude: int, num_samples: int = CHUNK_SAMPLES) -> bytes:
    """A synthetic int16 mono chunk of constant amplitude (RMS == amplitude)."""
    import numpy as np
    return np.full(num_samples, amplitude, dtype=np.int16).tobytes()


class FakeWakeModel:
    """Scores 1.0 on its Nth call (1-indexed), 0.0 on every other call."""

    def __init__(self, trigger_on_call=None):
        self._trigger_on_call = trigger_on_call
        self.calls = 0

    def score(self, chunk: bytes) -> float:
        self.calls += 1
        return 1.0 if self._trigger_on_call == self.calls else 0.0


def test_ignores_chunks_below_wake_threshold():
    listener = WakeWordListener(
        wake_models=[FakeWakeModel()], on_utterance_ready=lambda audio: None
    )

    listener.process_chunk(_chunk(0))

    assert listener.state == "listening"


def test_wake_word_trigger_starts_recording():
    listener = WakeWordListener(
        wake_models=[FakeWakeModel(trigger_on_call=1)],
        on_utterance_ready=lambda audio: None,
    )

    listener.process_chunk(_chunk(0))

    assert listener.state == "recording"


def test_pre_roll_is_prepended_when_recording_starts():
    delivered = []
    chunk_seconds = CHUNK_SAMPLES / SAMPLE_RATE
    listener = WakeWordListener(
        wake_models=[FakeWakeModel(trigger_on_call=2)],
        on_utterance_ready=delivered.append,
        pre_roll_seconds=chunk_seconds * 2,   # keep the 2 most recent chunks
        max_utterance_seconds=chunk_seconds,  # end immediately once recording starts
        min_utterance_seconds=0.0,
    )
    pre_roll_chunk = _chunk(111)
    trigger_chunk = _chunk(222)

    listener.process_chunk(pre_roll_chunk)  # listening: buffered into pre-roll
    listener.process_chunk(trigger_chunk)   # triggers -> recording -> already at max duration -> sends

    assert delivered[0] == pre_roll_chunk + trigger_chunk


def test_silence_hangover_ends_utterance_and_delivers_audio():
    delivered = []
    chunk_seconds = CHUNK_SAMPLES / SAMPLE_RATE
    listener = WakeWordListener(
        wake_models=[FakeWakeModel(trigger_on_call=1)],
        on_utterance_ready=delivered.append,
        silence_rms_threshold=100,
        silence_hangover_seconds=chunk_seconds * 2,
        min_utterance_seconds=0.0,
    )

    listener.process_chunk(_chunk(0))     # triggers wake word -> recording
    listener.process_chunk(_chunk(5000))  # loud speech chunk
    listener.process_chunk(_chunk(0))     # silence chunk 1
    listener.process_chunk(_chunk(0))     # silence chunk 2 -> hangover reached

    assert listener.state == "sending"
    assert len(delivered) == 1


def test_loud_chunk_resets_silence_timer():
    delivered = []
    chunk_seconds = CHUNK_SAMPLES / SAMPLE_RATE
    listener = WakeWordListener(
        wake_models=[FakeWakeModel(trigger_on_call=1)],
        on_utterance_ready=delivered.append,
        silence_rms_threshold=100,
        silence_hangover_seconds=chunk_seconds * 2,
    )

    listener.process_chunk(_chunk(0))     # trigger -> recording
    listener.process_chunk(_chunk(0))     # silence chunk 1
    listener.process_chunk(_chunk(5000))  # loud -> resets silence timer
    listener.process_chunk(_chunk(0))     # silence chunk 1 (again, not 2 in a row)

    assert listener.state == "recording"
    assert delivered == []


def test_max_duration_forces_end_even_without_silence():
    delivered = []
    chunk_seconds = CHUNK_SAMPLES / SAMPLE_RATE
    listener = WakeWordListener(
        wake_models=[FakeWakeModel(trigger_on_call=1)],
        on_utterance_ready=delivered.append,
        max_utterance_seconds=chunk_seconds * 2,
        min_utterance_seconds=0.0,
    )

    listener.process_chunk(_chunk(0))     # trigger -> recording
    listener.process_chunk(_chunk(5000))  # loud, non-silent, but hits max duration

    assert listener.state == "sending"
    assert len(delivered) == 1


def test_short_utterance_below_min_duration_is_dropped_silently():
    delivered = []
    chunk_seconds = CHUNK_SAMPLES / SAMPLE_RATE
    listener = WakeWordListener(
        wake_models=[FakeWakeModel(trigger_on_call=1)],
        on_utterance_ready=delivered.append,
        max_utterance_seconds=chunk_seconds,  # ends after just 1 chunk
        min_utterance_seconds=1.0,            # far longer than what gets recorded
    )

    listener.process_chunk(_chunk(0))  # trigger -> recording -> immediately hits max duration

    assert listener.state == "listening"
    assert delivered == []


def test_sending_state_ignores_further_scoring_until_marked_finished():
    chunk_seconds = CHUNK_SAMPLES / SAMPLE_RATE
    model = FakeWakeModel(trigger_on_call=1)
    listener = WakeWordListener(
        wake_models=[model],
        on_utterance_ready=lambda audio: None,
        max_utterance_seconds=chunk_seconds,
        min_utterance_seconds=0.0,
    )
    listener.process_chunk(_chunk(0))  # triggers, immediately ends (max duration) -> sending
    assert listener.state == "sending"
    calls_before = model.calls

    listener.process_chunk(_chunk(0))  # must be ignored while sending

    assert listener.state == "sending"
    assert model.calls == calls_before

    listener.mark_sending_finished()

    assert listener.state == "listening"


def test_on_state_change_fires_for_each_transition():
    chunk_seconds = CHUNK_SAMPLES / SAMPLE_RATE
    seen = []
    listener = WakeWordListener(
        wake_models=[FakeWakeModel(trigger_on_call=1)],
        on_utterance_ready=lambda audio: None,
        on_state_change=seen.append,
        max_utterance_seconds=chunk_seconds,
        min_utterance_seconds=0.0,
    )

    listener.process_chunk(_chunk(0))  # listening -> recording -> sending
    listener.mark_sending_finished()   # sending -> listening

    assert seen == ["recording", "sending", "listening"]
