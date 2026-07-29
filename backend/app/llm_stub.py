# TODO: replace with a real Qwen3-Coder call in a later plan.
def generate_response(transcript: str) -> str:
    if not transcript.strip():
        return "I didn't catch that."
    return f"You said: {transcript}"
