import asyncio

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.llm_stub import generate_response
from app.protocol import (
    ErrorMessage,
    ResponseCompleteMessage,
    ResponseTextMessage,
    TranscriptMessage,
    decode,
    encode,
)

MAX_UTTERANCE_BYTES = 16_000 * 1 * 2 * 30  # 30 seconds of PCM16 mono


def pcm16_bytes_to_float32(raw: bytes) -> np.ndarray:
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0


def create_app(stt, tts) -> FastAPI:
    app = FastAPI()

    @app.websocket("/ws/session")
    async def session(websocket: WebSocket):
        await websocket.accept()
        buffer = bytearray()
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    chunk = message["bytes"]
                    if len(buffer) + len(chunk) > MAX_UTTERANCE_BYTES:
                        await websocket.close(code=1009, reason="utterance too large")
                        return
                    buffer.extend(chunk)
                    continue
                if message.get("text") is not None:
                    parsed = decode(message["text"])
                    if parsed.get("type") == "end_utterance":
                        await _handle_utterance(websocket, bytes(buffer), stt, tts)
                        buffer = bytearray()
        except WebSocketDisconnect:
            return

    return app


async def _handle_utterance(websocket: WebSocket, raw_audio: bytes, stt, tts) -> None:
    try:
        audio = pcm16_bytes_to_float32(raw_audio)
        transcript = await asyncio.to_thread(stt.transcribe, audio)
        await websocket.send_text(encode(TranscriptMessage(text=transcript)))

        response_text = generate_response(transcript)
        await websocket.send_text(encode(ResponseTextMessage(text=response_text)))

        chunks = await asyncio.to_thread(tts.synthesize_chunks, response_text)
        for chunk in chunks:
            await websocket.send_bytes(chunk)

        await websocket.send_text(encode(ResponseCompleteMessage()))
    except Exception as exc:
        await websocket.send_text(encode(ErrorMessage(message=str(exc))))
