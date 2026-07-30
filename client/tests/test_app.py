from unittest.mock import Mock, MagicMock
from pynput import keyboard

from app import FridayTrayApp, HOTKEY
from ws_client import UtteranceResult


class FakeClient:
    def __init__(self, result):
        self._result = result
        self.sent_audio = None

    def send_utterance(self, audio_bytes):
        self.sent_audio = audio_bytes
        return self._result


def test_handle_utterance_plays_audio_chunks(monkeypatch):
    played = []
    monkeypatch.setattr("app.play", lambda chunk: played.append(chunk))

    tray_app = FridayTrayApp(server_url="ws://test")
    tray_app._client = FakeClient(UtteranceResult("hi", "You said: hi", [b"\x01", b"\x02"]))

    tray_app._handle_utterance(b"\x00\x00")

    assert played == [b"\x01", b"\x02"]
    assert tray_app._client.sent_audio == b"\x00\x00"


def test_handle_utterance_does_not_play_audio_on_error(monkeypatch):
    played = []
    monkeypatch.setattr("app.play", lambda chunk: played.append(chunk))

    tray_app = FridayTrayApp(server_url="ws://test")
    tray_app._client = FakeClient(UtteranceResult("hi", "", [], error="boom"))

    tray_app._handle_utterance(b"\x00\x00")

    assert played == []


class FailingClient:
    def send_utterance(self, audio_bytes):
        raise OSError("connection refused")


def test_handle_utterance_recovers_from_connection_error(monkeypatch):
    played = []
    monkeypatch.setattr("app.play", lambda chunk: played.append(chunk))

    tray_app = FridayTrayApp(server_url="ws://test")
    tray_app._client = FailingClient()

    tray_app._handle_utterance(b"\x00\x00")

    assert played == []
    assert tray_app._client is None


def test_on_press_hotkey_twice_only_starts_recording_once():
    """Test that repeated on_press events (X11 auto-repeat) only call start() once."""
    tray_app = FridayTrayApp(server_url="ws://test")
    tray_app._recorder = Mock()

    # First press should start recording
    tray_app._on_press(HOTKEY)
    assert tray_app._recorder.start.call_count == 1
    assert tray_app._recording is True

    # Second press (from X11 auto-repeat) should NOT call start() again
    tray_app._on_press(HOTKEY)
    assert tray_app._recorder.start.call_count == 1
    assert tray_app._recording is True


def test_on_release_after_on_press_resets_state_and_allows_restart():
    """Test that releasing after pressing correctly stops recording and allows restart."""
    tray_app = FridayTrayApp(server_url="ws://test")
    tray_app._recorder = Mock()
    tray_app._client = FakeClient(UtteranceResult("hi", "You said: hi", []))

    # Press the hotkey
    tray_app._on_press(HOTKEY)
    assert tray_app._recording is True
    assert tray_app._recorder.start.call_count == 1

    # Release the hotkey
    tray_app._on_press(HOTKEY)  # This won't do anything due to guard
    tray_app._on_release(HOTKEY)
    assert tray_app._recording is False
    assert tray_app._recorder.stop.call_count == 1

    # Pressing again should work (not stuck in recording state)
    tray_app._on_press(HOTKEY)
    assert tray_app._recording is True
    assert tray_app._recorder.start.call_count == 2


def test_on_release_without_prior_press_does_not_call_stop():
    """Test that releasing without a prior press doesn't call stop()."""
    tray_app = FridayTrayApp(server_url="ws://test")
    tray_app._recorder = Mock()

    # Release without ever pressing
    tray_app._on_release(HOTKEY)

    # stop() should not be called
    assert tray_app._recorder.stop.call_count == 0
    assert tray_app._recording is False
