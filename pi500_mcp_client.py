#!/usr/bin/env python3

import asyncio
import os
import sys
import json
import time
import select
import traceback
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from piper import PiperVoice
from elevenlabs.client import ElevenLabs


# ========== SMARTASS IMPORTS ==========
import config
import secret  # All keys: ELEVENLABS_KEY, ANTHROPIC_API_KEY, etc.
from audio_manager_vosk import AudioManager
from vosk_transcriber import VoskTranscriber
from display_manager import DisplayManager

# MCP protocol client imports (replace with your actual MCP client library)
import websockets

# Wake word detection
import pyaudio
import snowboydetect
import numpy as np
from evdev import InputDevice, list_devices, ecodes

# ========== CONFIGURATION ==========
WAKEWORD_DIR = Path("/home/user/LAURA/wakewords")
WAKEWORD_RESOURCE = "resources/common.res"
WAKE_WORDS = config.WAKE_WORDS
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK = 2048
CHAT_LOG_DIR = config.CHAT_LOG_DIR
PERSONA = config.ACTIVE_PERSONA
VOICE = config.VOICE
TTS_ENGINE = config.TTS_ENGINE

DEVICE_ID = getattr(config, "DEVICE_ID", "pi500-client")
TTS_LOCATION = getattr(config, "TTS_LOCATION", "server")  # "client" or "server"
PIPER_MODEL = getattr(config, "PIPER_MODEL", "")  # Path to your Piper model
PIPER_VOICE = getattr(config, "PIPER_VOICE", "")  # Optional path to voice directory
ELEVENLABS_KEY = getattr(secret, "ELEVENLABS_KEY", "")  # Get ElevenLabs key
CLIENT_TTS_ENGINE = getattr(config, "CLIENT_TTS_ENGINE", "piper")  # "piper" or "elevenlabs"

# Update the DEVICE_CAPABILITIES
DEVICE_CAPABILITIES = {
    "input": ["text", "audio"],
    "output": ["text", "audio"],
    "persona": PERSONA,
    "voice": VOICE,
    "tts_engine": TTS_ENGINE,
    "tts_location": TTS_LOCATION,  # Tell server where TTS should happen
}

# ========== MCP SERVER CONFIG ==========
MCP_SERVER_URI = "ws://localhost:8765"  # Update to your MCP server address/port

OUTPUT_MODE = getattr(config, "OUTPUT_MODE", ["text", "audio"])

def find_pi_keyboard():
    for path in list_devices():
        dev = InputDevice(path)
        if "Pi 500" in dev.name and "Keyboard" in dev.name:
            return dev
    return None

class WakeWordDetector:
    def __init__(self):
        self.model_paths = [str(WAKEWORD_DIR / name) for name in WAKE_WORDS]
        self.sensitivities = ",".join(str(WAKE_WORDS[name]) for name in WAKE_WORDS)
        self.model_names = list(WAKE_WORDS.keys())
        if not Path(WAKEWORD_RESOURCE).exists():
            raise FileNotFoundError(f"Snowboy resource file missing: {WAKEWORD_RESOURCE}")
        for path in self.model_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Wakeword model file missing: {path}")
        self.detector = snowboydetect.SnowboyDetect(
            resource_filename=str(WAKEWORD_RESOURCE).encode(),
            model_str=",".join(self.model_paths).encode()
        )
        self.detector.SetSensitivity(self.sensitivities.encode())
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            rate=AUDIO_SAMPLE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=AUDIO_CHUNK
        )
    def detect(self):
        data = self.stream.read(AUDIO_CHUNK, exception_on_overflow=False)
        result = self.detector.RunDetection(data)
        if result > 0:
            return self.model_names[result - 1]
        return None
    def cleanup(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()

class PiMCPClient:
    def __init__(self):
        self.audio_manager = AudioManager()
        self.display_manager = DisplayManager(
            svg_path=config.SVG_PATH,
            boot_img_path=config.BOOT_IMG_PATH,
            window_size=config.WINDOW_SIZE
        )
        self.stt = VoskTranscriber(config.VOSK_MODEL_PATH)
        self.wakeword = WakeWordDetector()
        self.keyboard = find_pi_keyboard()
        self.chat_log_dir = CHAT_LOG_DIR
        os.makedirs(self.chat_log_dir, exist_ok=True)
        self.notification_queue = asyncio.Queue()
        self.persona = PERSONA
        self.voice = VOICE
        self.session_id = None
        self.init_piper()
        self.eleven = None
        if TTS_LOCATION == "client" and CLIENT_TTS_ENGINE == "elevenlabs" and ELEVENLABS_KEY:
            self.eleven = ElevenLabs(api_key=ELEVENLABS_KEY)

        
    async def save_log(self, role, content):
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self.chat_log_dir, f"chat_log_{today}.json")
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    logs = json.load(f)
            except Exception:
                pass
        logs.append(entry)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)

    def listen_keyboard(self) -> bool:
        if not self.keyboard:
            return False
        r, _, _ = select.select([self.keyboard.fd], [], [], 0)
        if not r:
            return False
        for _ in range(5):
            event = self.keyboard.read_one()
            if event and event.type == ecodes.EV_KEY and event.code == 125 and event.value == 1:
                return True
        return False

    def listen_wakeword(self) -> Optional[str]:
        return self.wakeword.detect()

    async def play_audio(self, audio_bytes):
        fname = "assistant_response.mp3"
        with open(fname, "wb") as f:
            f.write(audio_bytes)
        await self.audio_manager.queue_audio(audio_file=fname)
        await self.audio_manager.wait_for_audio_completion()

    async def handle_notification(self, msg):
        await self.display_manager.update_display("speaking", mood="cheerful")
        await self.audio_manager.queue_audio(audio_file=None, generated_text=msg)
        await self.audio_manager.wait_for_audio_completion()
        await self.display_manager.update_display("idle")

    async def register_device(self, ws):
        register_msg = {
            "tool": "register_device",
            "args": {
                "device_id": DEVICE_ID,
                "capabilities": DEVICE_CAPABILITIES
            }
        }
        await ws.send(json.dumps(register_msg))
        resp_raw = await ws.recv()
        try:
            resp = json.loads(resp_raw)
            self.session_id = resp["session_id"]
            print(f"Registered, session_id: {self.session_id}")
        except Exception:
            print("Failed to register device, response:", resp_raw)
            raise RuntimeError("Device registration failed")

    async def send_process_input(self, ws, transcript):
        request = {
            "tool": "process_input",
            "args": {
                "session_id": self.session_id,
                "input_type": "text",
                "payload": {"text": transcript},
                "output_mode": OUTPUT_MODE,
                "timestamp": datetime.utcnow().isoformat(),
            }
        }
        await ws.send(json.dumps(request))

    async def run(self):
        print(f"Connecting to MCP server at {MCP_SERVER_URI}...")
        async with websockets.connect(MCP_SERVER_URI) as ws:
            print("Connected.")
            await self.register_device(ws)
            await self.display_manager.update_display("idle")
            while True:
                await self.display_manager.update_display("idle")
                try:
                    notif = self.notification_queue.get_nowait()
                    await self.handle_notification(notif)
                    continue
                except asyncio.QueueEmpty:
                    pass
                wake_model = None
                if self.listen_keyboard():
                    print("Keyboard wake")
                    wake_model = "keyboard"
                else:
                    wm = self.listen_wakeword()
                    if wm:
                        print(f"Wakeword detected: {wm}")
                        wake_model = wm
                if not wake_model:
                    await asyncio.sleep(0.05)
                    continue
                await self.display_manager.update_display("listening")
                transcript = await self.record_and_transcribe()
                if not transcript:
                    await self.display_manager.update_display("idle")
                    continue
                await self.save_log("user", transcript)
                await self.send_process_input(ws, transcript)
                await self.display_manager.update_display("thinking")
                response_raw = await ws.recv()
                try:
                    response = json.loads(response_raw)
                except Exception:
                    print("Invalid server response")
                    continue
                if response.get("persona"):
                    self.persona = response["persona"]
                if response.get("voice"):
                    self.voice = response["voice"]
                mood = response.get("mood", "casual")
                await self.display_manager.update_display("speaking", mood=mood)
                
                # Handle audio response based on TTS location preference
                if TTS_LOCATION == "client" and TTS_ENGINE == "piper" and "text" in response:
                    # Use Piper for client-side TTS
                    await self.generate_piper_speech(response["text"])
                elif "audio" in response and response["audio"]:
                    # Use server-generated audio
                    await self.play_audio(bytes(response["audio"]))
                elif "text" in response and response["text"]:
                    # Fallback to client's default TTS
                    await self.audio_manager.queue_audio(audio_file=None, generated_text=response["text"])
                    await self.audio_manager.wait_for_audio_completion()
                await self.display_manager.update_display("idle")
                await self.save_log("assistant", response.get("text", ""))

    def init_piper(self):
        """Initialize Piper TTS if needed"""
        if TTS_LOCATION == "client" and TTS_ENGINE == "piper":
            try:
                subprocess.run(["piper", "--help"], capture_output=True, check=False)
                print("Piper TTS initialized for client-side speech generation")
                return True
            except FileNotFoundError:
                print("Warning: Piper not found but client-side TTS requested.")
                return False
        return False

    async def generate_piper_speech(self, text):
        """Generate speech using Piper locally"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_filename = temp_wav.name
        
        try:
            # Build the Piper command
            cmd = ["piper"]
            if PIPER_MODEL:
                cmd.extend(["--model", PIPER_MODEL])
            if PIPER_VOICE:
                cmd.extend(["--voice", PIPER_VOICE])
            cmd.extend(["--output_file", temp_filename])
            
            # Run Piper with text as input
            process = subprocess.run(
                cmd,
                input=text.encode(),
                capture_output=True,
                check=True
            )
            
            # Play the generated audio
            await self.audio_manager.queue_audio(audio_file=temp_filename)
            await self.audio_manager.wait_for_audio_completion()
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)        

async def main():
    client = PiMCPClient()
    try:
        await client.run()
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
    finally:
        client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
