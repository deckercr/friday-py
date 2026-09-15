import time
from unittest.mock import patch

import numpy as np

from audio import QUEUE_MAXSIZE, listen_chunks, play


class _FakeInputStream:
    def __init__(self, chunks_to_deliver, **kwargs):
        self._chunks = chunks_to_deliver
        self._callback = kwargs["callback"]

    def __enter__(self):
        for chunk in self._chunks:
            self._callback(chunk, len(chunk), None, None)
        return self

    def __exit__(self, *exc_info):
        return False


@patch("audio.sd.InputStream")
def test_listen_chunks_yields_callback_data_as_bytes(mock_input_stream_cls):
    chunks = [b"\x01\x02", b"\x03\x04"]
    mock_input_stream_cls.side_effect = lambda **kwargs: _FakeInputStream(chunks, **kwargs)

    generator = listen_chunks(chunk_samples=1, sample_rate=16000, channels=1)
    received = [next(generator), next(generator)]

    assert received == chunks


@patch("audio.sd.InputStream")
def test_listen_chunks_drops_oldest_when_queue_is_full(mock_input_stream_cls):
    # One more chunk than the queue can hold; each is distinguishable by
    # its own bytes so we can tell which ones survived.
    chunks = [bytes([i]) for i in range(QUEUE_MAXSIZE + 1)]
    mock_input_stream_cls.side_effect = lambda **kwargs: _FakeInputStream(chunks, **kwargs)

    generator = listen_chunks(chunk_samples=1, sample_rate=16000, channels=1)
    received = [next(generator) for _ in range(QUEUE_MAXSIZE)]

    # The oldest chunk (index 0) was dropped to make room for the last one;
    # everything from index 1 onward survived, in order.
    assert received == chunks[1:]


@patch("audio.sd")
def test_play_uses_blocking_output_stream(mock_sd):
    audio = np.array([[1], [2]], dtype="int16")
    audio_bytes = audio.tobytes()

    play(audio_bytes)

    mock_sd.OutputStream.assert_called_once()
    _, kwargs = mock_sd.OutputStream.call_args
    assert "callback" not in kwargs

    output_stream = mock_sd.OutputStream.return_value
    output_stream.start.assert_called_once()
    output_stream.write.assert_called_once()
    written_audio = output_stream.write.call_args[0][0]
    np.testing.assert_array_equal(written_audio, audio)
    output_stream.stop.assert_called_once()
    output_stream.close.assert_called_once()

    mock_sd.play.assert_not_called()
    mock_sd.wait.assert_not_called()
