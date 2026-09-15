import os
import threading

import pystray
from PIL import Image, ImageDraw
from websockets.exceptions import WebSocketException

from audio import listen_chunks, play
from wake_listener import CHUNK_SAMPLES, SAMPLE_RATE, WakeWordListener
from wake_word_model import OpenWakeWordModel
from ws_client import SERVER_URL, connect_to_server

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

_ICON_COLORS = {
    "listening": "gray",
    "recording": "red",
    "sending": "orange",
}


def _create_icon_image(color: str) -> Image.Image:
    image = Image.new("RGB", (64, 64), color)
    draw = ImageDraw.Draw(image)
    draw.ellipse((16, 16, 48, 48), fill="white")
    return image


class FridayApp:
    def __init__(self, server_url: str, wake_models):
        self._server_url = server_url
        self._client = None
        self._icon = pystray.Icon("friday", icon=_create_icon_image(_ICON_COLORS["listening"]))
        self._listener = WakeWordListener(
            wake_models=wake_models,
            on_utterance_ready=self._on_utterance_ready,
            on_state_change=self._on_state_change,
        )

    def _on_state_change(self, state: str) -> None:
        self._icon.icon = _create_icon_image(_ICON_COLORS[state])

    def _on_utterance_ready(self, audio_bytes: bytes) -> None:
        # Off the mic-listening thread: send_utterance() blocks on the full
        # network round trip (transcribe + synthesize) plus playback.
        threading.Thread(target=self._send_and_report, args=(audio_bytes,), daemon=True).start()

    def _send_and_report(self, audio_bytes: bytes) -> None:
        try:
            try:
                if self._client is None:
                    self._client = connect_to_server(self._server_url)
                result = self._client.send_utterance(audio_bytes)
            except (OSError, WebSocketException, ValueError, KeyError) as exc:
                print(f"Disconnected ({exc}); will reconnect on next utterance")
                self._client = None
                return

            if result.error:
                print(f"Error: {result.error}")
                return

            print(f"Transcript: {result.transcript}")
            print(f"Response: {result.response_text}")
            for chunk in result.audio_chunks:
                play(chunk)
        finally:
            self._listener.mark_sending_finished()

    def _listen_forever(self) -> None:
        for chunk in listen_chunks(chunk_samples=CHUNK_SAMPLES, sample_rate=SAMPLE_RATE):
            try:
                self._listener.process_chunk(chunk)
            except Exception as exc:  # noqa: BLE001 - keep the mic thread alive at all costs
                print(f"Warning: error processing audio chunk ({exc}); still listening")

    def run(self) -> None:
        threading.Thread(target=self._listen_forever, daemon=True).start()
        self._icon.run()


if __name__ == "__main__":
    wake_models = [
        OpenWakeWordModel(os.path.join(MODELS_DIR, "hey_friday.onnx"), "hey_friday"),
        OpenWakeWordModel(os.path.join(MODELS_DIR, "friday.onnx"), "friday"),
    ]
    # Point at a remote backend (e.g. the GPU host) with FRIDAY_SERVER_URL,
    # mirroring the backend's own FRIDAY_HOST convention.
    FridayApp(
        server_url=os.environ.get("FRIDAY_SERVER_URL", SERVER_URL),
        wake_models=wake_models,
    ).run()
