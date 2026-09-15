import threading
from unittest.mock import MagicMock

import numpy as np

from app import FridayApp, _create_icon_image, _ICON_COLORS
from ws_client import UtteranceResult


class FakeClient:
    def __init__(self, result):
        self._result = result
        self.sent_audio = None

    def send_utterance(self, audio_bytes):
        self.sent_audio = audio_bytes
        return self._result


class FailingClient:
    def send_utterance(self, audio_bytes):
        raise OSError("connection refused")


class MalformedProtocolClient:
    def send_utterance(self, audio_bytes):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")


class MissingKeyProtocolClient:
    def send_utterance(self, audio_bytes):
        raise KeyError("type")


def test_on_state_change_updates_icon_color():
    app = FridayApp(server_url="ws://test", wake_models=[])

    app._on_state_change("recording")

    expected = np.array(_create_icon_image(_ICON_COLORS["recording"]))
    actual = np.array(app._icon.icon)
    np.testing.assert_array_equal(actual, expected)


def test_send_and_report_plays_audio_and_resumes_listening(monkeypatch):
    played = []
    monkeypatch.setattr("app.play", lambda chunk: played.append(chunk))
    app = FridayApp(server_url="ws://test", wake_models=[])
    app._client = FakeClient(UtteranceResult("hi", "You said: hi", [b"\x01", b"\x02"]))
    resumed = []
    monkeypatch.setattr(app._listener, "mark_sending_finished", lambda: resumed.append(True))

    app._send_and_report(b"\x00\x00")

    assert played == [b"\x01", b"\x02"]
    assert app._client.sent_audio == b"\x00\x00"
    assert resumed == [True]


def test_send_and_report_does_not_play_audio_on_error(monkeypatch):
    played = []
    monkeypatch.setattr("app.play", lambda chunk: played.append(chunk))
    app = FridayApp(server_url="ws://test", wake_models=[])
    app._client = FakeClient(UtteranceResult("hi", "", [], error="boom"))
    monkeypatch.setattr(app._listener, "mark_sending_finished", lambda: None)

    app._send_and_report(b"\x00\x00")

    assert played == []


def test_send_and_report_recovers_from_connection_error(monkeypatch):
    played = []
    monkeypatch.setattr("app.play", lambda chunk: played.append(chunk))
    app = FridayApp(server_url="ws://test", wake_models=[])
    app._client = FailingClient()
    resumed = []
    monkeypatch.setattr(app._listener, "mark_sending_finished", lambda: resumed.append(True))

    app._send_and_report(b"\x00\x00")

    assert played == []
    assert app._client is None
    assert resumed == [True]


def test_send_and_report_recovers_from_malformed_protocol_response(monkeypatch):
    played = []
    monkeypatch.setattr("app.play", lambda chunk: played.append(chunk))
    app = FridayApp(server_url="ws://test", wake_models=[])
    app._client = MalformedProtocolClient()
    monkeypatch.setattr(app._listener, "mark_sending_finished", lambda: None)

    app._send_and_report(b"\x00\x00")

    assert played == []
    assert app._client is None


def test_send_and_report_resumes_listening_when_playback_raises(monkeypatch):
    def _raising_play(chunk):
        raise RuntimeError("PortAudio error")

    monkeypatch.setattr("app.play", _raising_play)
    app = FridayApp(server_url="ws://test", wake_models=[])
    app._client = FakeClient(UtteranceResult("hi", "You said: hi", [b"\x01"]))
    resumed = []
    monkeypatch.setattr(app._listener, "mark_sending_finished", lambda: resumed.append(True))

    try:
        app._send_and_report(b"\x00\x00")
    except RuntimeError:
        pass

    assert resumed == [True]


def test_send_and_report_recovers_from_missing_protocol_key(monkeypatch):
    played = []
    monkeypatch.setattr("app.play", lambda chunk: played.append(chunk))
    app = FridayApp(server_url="ws://test", wake_models=[])
    app._client = MissingKeyProtocolClient()
    monkeypatch.setattr(app._listener, "mark_sending_finished", lambda: None)

    app._send_and_report(b"\x00\x00")

    assert played == []
    assert app._client is None


def test_on_utterance_ready_dispatches_to_background_thread(monkeypatch):
    app = FridayApp(server_url="ws://test", wake_models=[])
    calls = []
    monkeypatch.setattr(app, "_send_and_report", lambda audio: calls.append(audio))

    class ImmediateThread:
        def __init__(self, target, args, daemon):
            self._target = target
            self._args = args

        def start(self):
            self._target(*self._args)

    monkeypatch.setattr(threading, "Thread", ImmediateThread)

    app._on_utterance_ready(b"\x00\x00")

    assert calls == [b"\x00\x00"]
