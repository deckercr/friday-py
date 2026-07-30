import json
from dataclasses import dataclass

from websockets.sync.client import connect

SERVER_URL = "ws://localhost:8000/ws/session"
RECV_TIMEOUT_SECONDS = 30


@dataclass
class UtteranceResult:
    transcript: str
    response_text: str
    audio_chunks: list
    error: str | None = None


class FridayClient:
    def __init__(self, connection):
        self._connection = connection

    def send_utterance(self, audio_bytes: bytes) -> UtteranceResult:
        self._connection.send(audio_bytes)
        self._connection.send(json.dumps({"type": "end_utterance"}))

        transcript = ""
        response_text = ""
        audio_chunks = []

        while True:
            message = self._connection.recv(timeout=RECV_TIMEOUT_SECONDS)
            if isinstance(message, bytes):
                audio_chunks.append(message)
                continue
            parsed = json.loads(message)
            if parsed["type"] == "transcript":
                transcript = parsed["text"]
            elif parsed["type"] == "response_text":
                response_text = parsed["text"]
            elif parsed["type"] == "error":
                return UtteranceResult(transcript, response_text, audio_chunks, error=parsed["message"])
            elif parsed["type"] == "response_complete":
                return UtteranceResult(transcript, response_text, audio_chunks)


def connect_to_server(url: str = SERVER_URL) -> FridayClient:
    return FridayClient(connect(url))
