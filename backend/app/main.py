import os
from pathlib import Path

import uvicorn
from fastapi.staticfiles import StaticFiles

from app.server import create_app
from app.stt import SpeechToText
from app.tts import TextToSpeech

BASE_DIR = Path(__file__).resolve().parent.parent


def build_app():
    stt = SpeechToText()
    tts = TextToSpeech(model_path=str(BASE_DIR / "models" / "en_GB-southern_english_female-low.onnx"))
    app = create_app(stt, tts)
    app.mount("/", StaticFiles(directory=str(BASE_DIR.parent / "frontend"), html=True), name="frontend")
    return app


if __name__ == "__main__":
    uvicorn.run(build_app(), host=os.environ.get("FRIDAY_HOST", "127.0.0.1"), port=8000)
