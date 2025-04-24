from elevenlabs.client import ElevenLabs
import requests

class TTSHandler:
    def __init__(self, config):
        self.config = config  # Store config in instance variable
        self.tts_engine = config["TTS_ENGINE"]
        self.eleven = None
        if self.tts_engine == "elevenlabs":
            self.eleven = ElevenLabs(api_key=config["ELEVENLABS_KEY"])

    def generate_audio(self, text):
        if self.tts_engine == "elevenlabs":
            return self._generate_elevenlabs(text)
        else:
            return self._generate_alltalk(text)

    def _generate_elevenlabs(self, text):
        audio = b"".join(self.eleven.generate(
            text=text,
            voice=self.config["VOICE"],
            model=self.config["ELEVENLABS_MODEL"],
            output_format="mp3_44100_128"
        ))
        return audio

    def _generate_alltalk(self, text):
        try:
            response = requests.post(
                f"{self.config['ALLTALK_HOST']}/api/tts",
                json={
                    "text": text,
                    "voice": self.config["ALLTALK_VOICE"],
                    "model": self.config["ALLTALK_MODEL"]
                },
                timeout=30
            )
            response.raise_for_status()

            if response.status_code == 200:
                return response.content
            else:
                raise Exception(f"AllTalk API error: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"AllTalk API request failed: {str(e)}")
            if self.eleven:
                print("Falling back to ElevenLabs...")
                return self._generate_elevenlabs(text)
            raise