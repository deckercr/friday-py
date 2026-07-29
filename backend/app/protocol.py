import json
from dataclasses import asdict, dataclass

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes per sample (16-bit PCM)


@dataclass
class TranscriptMessage:
    text: str
    type: str = "transcript"


@dataclass
class ResponseTextMessage:
    text: str
    type: str = "response_text"


@dataclass
class ErrorMessage:
    message: str
    type: str = "error"


@dataclass
class EndUtteranceMessage:
    type: str = "end_utterance"


@dataclass
class ResponseCompleteMessage:
    type: str = "response_complete"


def encode(message) -> str:
    return json.dumps(asdict(message))


def decode(raw: str) -> dict:
    return json.loads(raw)
