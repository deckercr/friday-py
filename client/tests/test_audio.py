from unittest.mock import patch

import numpy as np

from audio import AudioRecorder, play


@patch("audio.sd")
def test_start_creates_and_starts_input_stream(mock_sd):
    recorder = AudioRecorder()
    recorder.start()

    mock_sd.InputStream.assert_called_once()
    mock_sd.InputStream.return_value.start.assert_called_once()


@patch("audio.sd")
def test_stop_concatenates_captured_frames(mock_sd):
    recorder = AudioRecorder()
    recorder.start()
    recorder._callback(np.array([[1], [2]], dtype="int16"), 2, None, None)
    recorder._callback(np.array([[3]], dtype="int16"), 1, None, None)

    result = recorder.stop()

    expected = np.array([[1], [2], [3]], dtype="int16").tobytes()
    assert result == expected


@patch("audio.sd")
def test_stop_with_no_frames_returns_empty_bytes(mock_sd):
    recorder = AudioRecorder()
    recorder.start()

    result = recorder.stop()

    assert result == b""


@patch("audio.sd")
def test_play_calls_sounddevice_play_and_wait(mock_sd):
    audio_bytes = np.array([[1], [2]], dtype="int16").tobytes()
    play(audio_bytes)

    mock_sd.play.assert_called_once()
    mock_sd.wait.assert_called_once()
