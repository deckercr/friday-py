import time
from unittest.mock import patch

import numpy as np

from audio import AudioRecorder, play


def _counting_read_side_effect(frames, channels=1):
    """Return each frame in `frames` in order, one per call; after that,
    return an empty read so the background loop doesn't accumulate any more
    data but also doesn't block or raise (it just keeps polling until the
    test tells the recorder to stop).
    """
    call_count = 0

    def _read(chunk_size):
        nonlocal call_count
        if call_count < len(frames):
            data = frames[call_count]
            call_count += 1
            return data, False
        return np.zeros((0, channels), dtype="int16"), False

    return _read


def _wait_for_calls(mock_read, minimum, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mock_read.call_count >= minimum:
            return
        time.sleep(0.005)
    raise AssertionError(
        f"mock read was only called {mock_read.call_count} times, expected >= {minimum}"
    )


@patch("audio.sd")
def test_start_creates_blocking_input_stream_and_starts_it(mock_sd):
    mock_sd.InputStream.return_value.read.side_effect = _counting_read_side_effect([])
    recorder = AudioRecorder()

    recorder.start()
    try:
        mock_sd.InputStream.assert_called_once()
        _, kwargs = mock_sd.InputStream.call_args
        assert "callback" not in kwargs
        mock_sd.InputStream.return_value.start.assert_called_once()
    finally:
        recorder.stop()


@patch("audio.sd")
def test_stop_concatenates_frames_captured_by_read_loop(mock_sd):
    frames = [
        np.array([[1], [2]], dtype="int16"),
        np.array([[3]], dtype="int16"),
    ]
    mock_read = mock_sd.InputStream.return_value.read
    mock_read.side_effect = _counting_read_side_effect(frames)

    recorder = AudioRecorder()
    recorder.start()

    _wait_for_calls(mock_read, minimum=len(frames))

    result = recorder.stop()

    expected = np.concatenate(frames, axis=0).tobytes()
    assert result == expected
    mock_sd.InputStream.return_value.stop.assert_called_once()
    mock_sd.InputStream.return_value.close.assert_called_once()


@patch("audio.sd")
def test_stop_with_no_frames_returns_empty_bytes(mock_sd):
    mock_sd.InputStream.return_value.read.side_effect = _counting_read_side_effect([])
    recorder = AudioRecorder()
    recorder.start()

    result = recorder.stop()

    assert result == b""


@patch("audio.sd")
def test_stop_joins_thread_and_does_not_hang(mock_sd):
    frames = [np.array([[1], [2]], dtype="int16")]
    mock_read = mock_sd.InputStream.return_value.read
    mock_read.side_effect = _counting_read_side_effect(frames)

    recorder = AudioRecorder()
    recorder.start()
    _wait_for_calls(mock_read, minimum=len(frames))

    recorder.stop()

    assert not recorder._thread.is_alive()


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
