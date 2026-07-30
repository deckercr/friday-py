import json

from ws_client import FridayClient


class FakeConnection:
    def __init__(self, responses):
        self._responses = list(responses)
        self.sent = []

    def send(self, data):
        self.sent.append(data)

    def recv(self):
        return self._responses.pop(0)


def test_send_utterance_collects_transcript_response_and_audio():
    connection = FakeConnection(
        [
            json.dumps({"type": "transcript", "text": "hello"}),
            json.dumps({"type": "response_text", "text": "You said: hello"}),
            b"\x01\x02",
            b"\x03\x04",
            json.dumps({"type": "response_complete"}),
        ]
    )
    client = FridayClient(connection)

    result = client.send_utterance(b"\x00\x00")

    assert result.transcript == "hello"
    assert result.response_text == "You said: hello"
    assert result.audio_chunks == [b"\x01\x02", b"\x03\x04"]
    assert result.error is None
    assert connection.sent[0] == b"\x00\x00"
    assert json.loads(connection.sent[1]) == {"type": "end_utterance"}


def test_send_utterance_returns_error_message():
    connection = FakeConnection(
        [
            json.dumps({"type": "transcript", "text": "hello"}),
            json.dumps({"type": "error", "message": "boom"}),
        ]
    )
    client = FridayClient(connection)

    result = client.send_utterance(b"\x00\x00")

    assert result.error == "boom"
    assert result.audio_chunks == []
