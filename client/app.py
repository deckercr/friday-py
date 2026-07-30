import pystray
from PIL import Image, ImageDraw
from pynput import keyboard
from websockets.exceptions import WebSocketException

from audio import AudioRecorder, play
from ws_client import connect_to_server

HOTKEY = keyboard.Key.f9


def _create_icon_image(color: str) -> Image.Image:
    image = Image.new("RGB", (64, 64), color)
    draw = ImageDraw.Draw(image)
    draw.ellipse((16, 16, 48, 48), fill="white")
    return image


class FridayTrayApp:
    def __init__(self, server_url: str):
        self._server_url = server_url
        self._recorder = AudioRecorder()
        self._client = None
        self._icon = pystray.Icon("friday", icon=_create_icon_image("gray"))
        self._recording = False

    def _on_press(self, key) -> None:
        if key == HOTKEY and not self._recording:
            self._recording = True
            self._icon.icon = _create_icon_image("red")
            self._recorder.start()

    def _on_release(self, key) -> None:
        if key == HOTKEY and self._recording:
            self._recording = False
            audio_bytes = self._recorder.stop()
            self._icon.icon = _create_icon_image("gray")
            self._handle_utterance(audio_bytes)

    def _handle_utterance(self, audio_bytes: bytes) -> None:
        try:
            if self._client is None:
                self._client = connect_to_server(self._server_url)
            result = self._client.send_utterance(audio_bytes)
        except (OSError, WebSocketException) as exc:
            print(f"Disconnected ({exc}); will reconnect on next talk")
            self._client = None
            return

        if result.error:
            print(f"Error: {result.error}")
            return
        print(f"Transcript: {result.transcript}")
        print(f"Response: {result.response_text}")
        for chunk in result.audio_chunks:
            play(chunk)

    def run(self) -> None:
        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        listener.start()
        self._icon.run()


if __name__ == "__main__":
    FridayTrayApp(server_url="ws://localhost:8000/ws/session").run()
