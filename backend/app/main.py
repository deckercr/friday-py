import uvicorn
from fastapi.staticfiles import StaticFiles

from app.server import create_app
from app.stt import SpeechToText
from app.tts import TextToSpeech


def build_app():
    stt = SpeechToText()
    tts = TextToSpeech(model_path="models/en_GB-southern_english_female-low.onnx")
    app = create_app(stt, tts)
    app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
    return app


if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=8000)
