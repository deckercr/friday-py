"""Manual checkpoint: feed a WAV file through STT -> stub LLM -> TTS, write a response WAV.

Usage: uv run python scripts/pipeline_check.py --input test.wav --output response.wav
"""
import argparse
import wave

import numpy as np

from app.llm_stub import generate_response
from app.server import pcm16_bytes_to_float32
from app.stt import SpeechToText
from app.tts import TextToSpeech


def load_wav_as_float32(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wav_file:
        assert wav_file.getframerate() == 16000, "expected 16kHz input WAV"
        assert wav_file.getnchannels() == 1, "expected mono input WAV"
        raw = wav_file.readframes(wav_file.getnframes())
    return pcm16_bytes_to_float32(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voice-model", default="models/en_US-lessac-medium.onnx")
    args = parser.parse_args()

    stt = SpeechToText()
    tts = TextToSpeech(model_path=args.voice_model)

    audio = load_wav_as_float32(args.input)
    transcript = stt.transcribe(audio)
    print(f"Transcript: {transcript}")

    response_text = generate_response(transcript)
    print(f"Response: {response_text}")

    tts.synthesize_wav(response_text, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
