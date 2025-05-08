import asyncio
from typing import Optional
import os

# For API TTS
try:
    from elevenlabs.client import ElevenLabs
except ImportError:
    ElevenLabs = None

try:
    from cartesia import AsyncCartesia
except ImportError:
    AsyncCartesia = None

# For Piper/local TTS
import subprocess
import tempfile

from client_config import PREFERRED_TTS_ENGINE, VOICE, ELEVENLABS_MODEL, PIPER_MODEL, PIPER_VOICE, CARTESIA_MODEL, CARTESIA_VOICE_ID
from client_secret import ELEVENLABS_KEY, CARTESIA_API_KEY

class TTSHandler:
    def __init__(self):
        # Set up clients as needed
        self.preferred = PREFERRED_TTS_ENGINE
        self.voice = VOICE
        self.elevenlabs_model = ELEVENLABS_MODEL
        self.piper_model = PIPER_MODEL
        self.piper_voice = PIPER_VOICE
        self.cartesia_model = CARTESIA_MODEL
        self.cartesia_voice_id = CARTESIA_VOICE_ID

        self.elevenlabs_key = ELEVENLABS_KEY
        self.cartesia_api_key = CARTESIA_API_KEY

        self.eleven = ElevenLabs(api_key=self.elevenlabs_key) if ElevenLabs and self.elevenlabs_key else None
        self.cartesia = AsyncCartesia(api_key=self.cartesia_api_key) if AsyncCartesia and self.cartesia_api_key else None

    async def generate_audio(self, text: str) -> Optional[bytes]:
        # Try preferred first, then fallbacks
        engines = ["cartesia", "elevenlabs", "piper"]
        if self.preferred in engines:
            order = [self.preferred] + [e for e in engines if e != self.preferred]
        else:
            order = engines

        for engine in order:
            try:
                if engine == "cartesia" and self.cartesia:
                    audio = await self._generate_cartesia(text)
                    if audio:
                        return audio
                elif engine == "elevenlabs" and self.eleven:
                    audio = await self._generate_elevenlabs(text)
                    if audio:
                        return audio
                elif engine == "piper":
                    audio = await self._generate_piper(text)
                    if audio:
                        return audio
            except Exception as e:
                print(f"[TTSHandler] {engine} failed: {e}")
                continue
        # All failed
        return None

    async def _generate_cartesia(self, text: str) -> Optional[bytes]:
        # Cartesia async API
        audio_chunks = []
        async for output in self.cartesia.tts.bytes(
            model_id=self.cartesia_model,
            transcript=text,
            voice={"id": self.cartesia_voice_id},
            language="en",
            output_format={
                "container": "raw",
                "encoding": "pcm_f32le",
                "sample_rate": 44100,
            },
        ):
            audio_chunks.append(output)
        if audio_chunks:
            return b"".join(audio_chunks)
        return None

    async def _generate_elevenlabs(self, text: str) -> Optional[bytes]:
        # ElevenLabs is sync, so run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_elevenlabs_sync, text)

    def _generate_elevenlabs_sync(self, text: str) -> Optional[bytes]:
        if not self.eleven:
            return None
        audio = b"".join(self.eleven.generate(
            text=text,
            voice=self.voice,
            model=self.elevenlabs_model,
            output_format="mp3_44100_128"
        ))
        return audio if audio else None

    async def _generate_piper(self, text: str) -> Optional[bytes]:
        # Run Piper locally, output wav, read and return
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_filename = temp_wav.name
        try:
            cmd = ["piper"]
            if self.piper_model:
                cmd.extend(["--model", self.piper_model])
            if self.piper_voice:
                cmd.extend(["--voice", self.piper_voice])
            cmd.extend(["--output_file", temp_filename])

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate(input=text.encode())
            if os.path.exists(temp_filename):
                with open(temp_filename, "rb") as f:
                    data = f.read()
                return data
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
        return None