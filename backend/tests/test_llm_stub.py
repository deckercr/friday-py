from app.llm_stub import generate_response


def test_echoes_transcript():
    assert generate_response("hello") == "You said: hello"


def test_empty_transcript_returns_fallback():
    assert generate_response("") == "I didn't catch that."
    assert generate_response("   ") == "I didn't catch that."
