from app import FridayTrayApp
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
