import json

from app.protocol import (
    EndUtteranceMessage,
    ErrorMessage,
    ResponseCompleteMessage,
    ResponseTextMessage,
    TranscriptMessage,
    decode,
    encode,
)


def test_encode_transcript_message():
    result = encode(TranscriptMessage(text="hello"))
    assert json.loads(result) == {"text": "hello", "type": "transcript"}


def test_encode_response_text_message():
    result = encode(ResponseTextMessage(text="hi there"))
    assert json.loads(result) == {"text": "hi there", "type": "response_text"}


def test_encode_error_message():
    result = encode(ErrorMessage(message="boom"))
    assert json.loads(result) == {"message": "boom", "type": "error"}


def test_encode_end_utterance_message():
    result = encode(EndUtteranceMessage())
    assert json.loads(result) == {"type": "end_utterance"}


def test_encode_response_complete_message():
    result = encode(ResponseCompleteMessage())
    assert json.loads(result) == {"type": "response_complete"}


def test_decode_round_trip():
    raw = encode(TranscriptMessage(text="round trip"))
    assert decode(raw) == {"text": "round trip", "type": "transcript"}
