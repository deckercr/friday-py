import wave

from piper import PiperVoice


class TextToSpeech:
    def __init__(self, model_path: str):
        self._voice = PiperVoice.load(model_path)

    def synthesize_chunks(self, text: str) -> list[bytes]:
        return [chunk.audio_int16_bytes for chunk in self._voice.synthesize(text)]

    def synthesize_wav(self, text: str, output_path: str) -> None:
        with wave.open(output_path, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)
