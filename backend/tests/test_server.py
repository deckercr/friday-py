import json

from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from app.server import MAX_UTTERANCE_BYTES, create_app


class FakeStt:
    def transcribe(self, audio):
        return "hello world"


class FakeTts:
    def synthesize_chunks(self, text):
        return [b"\x00\x01", b"\x02\x03"]


class FakeTtsThatFails:
    def synthesize_chunks(self, text):
        raise RuntimeError("synth failed")


def test_full_utterance_round_trip():
    app = create_app(FakeStt(), FakeTts())
    client = TestClient(app)

    with client.websocket_connect("/ws/session") as ws:
        ws.send_bytes(b"\x00\x00" * 100)
        ws.send_text(json.dumps({"type": "end_utterance"}))

        transcript_msg = json.loads(ws.receive_text())
        assert transcript_msg == {"type": "transcript", "text": "hello world"}

        response_msg = json.loads(ws.receive_text())
        assert response_msg == {"type": "response_text", "text": "You said: hello world"}

        assert ws.receive_bytes() == b"\x00\x01"
        assert ws.receive_bytes() == b"\x02\x03"

        complete_msg = json.loads(ws.receive_text())
        assert complete_msg == {"type": "response_complete"}


def test_tts_failure_sends_error_message():
    app = create_app(FakeStt(), FakeTtsThatFails())
    client = TestClient(app)

    with client.websocket_connect("/ws/session") as ws:
        ws.send_bytes(b"\x00\x00" * 100)
        ws.send_text(json.dumps({"type": "end_utterance"}))

        ws.receive_text()  # transcript
        ws.receive_text()  # response_text

        error_msg = json.loads(ws.receive_text())
        assert error_msg == {"type": "error", "message": "synth failed"}


def test_oversized_utterance_closes_connection():
    app = create_app(FakeStt(), FakeTts())
    client = TestClient(app)

    with client.websocket_connect("/ws/session") as ws:
        ws.send_bytes(b"\x00" * (MAX_UTTERANCE_BYTES + 1))

        try:
            ws.receive_bytes()
            raised = False
        except WebSocketDisconnect as exc:
            raised = True
            assert exc.code == 1009

        assert raised
