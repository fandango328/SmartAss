#!/usr/bin/env python3

import requests
import aiohttp
import asyncio
import os
import sys
import json
import time
import select
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import socket
from contextlib import asynccontextmanager

# ========== SMARTASS IMPORTS ==========
import client_config as config
import secret  # All keys: ELEVENLABS_KEY, ANTHROPIC_API_KEY, etc.
from audio_manager_vosk import AudioManager
from vosk_transcriber import VoskTranscriber
from display_manager import DisplayManager
from core_functions import capture_speech

import pyaudio
sys.path.append('/home/user/LAURA/snowboy')
import snowboydetect
from evdev import InputDevice, list_devices, ecodes

from client_system_manager import ClientSystemManager
from client_tts_handler import TTSHandler

# ========== CONFIGURATION ==========
WAKEWORD_DIR = config.WAKEWORD_DIR
WAKEWORD_RESOURCE = config.WAKEWORD_RESOURCE
WAKE_WORDS = config.WAKE_WORDS
AUDIO_SAMPLE_RATE = config.AUDIO_SAMPLE_RATE
AUDIO_CHUNK = config.AUDIO_CHUNK
CHAT_LOG_DIR = config.CHAT_LOG_DIR
PERSONA = config.ACTIVE_PERSONA
VOICE = config.VOICE
TTS_ENGINE = config.TTS_ENGINE
DEVICE_ID = config.DEVICE_ID
TTS_LOCATION = config.TTS_LOCATION
CLIENT_TTS_ENGINE = config.CLIENT_TTS_ENGINE
MCP_SERVER_URI = config.MCP_SERVER_URI
OUTPUT_MODE = config.OUTPUT_MODE

AUTH_URL = "http://192.168.0.50:5000/oauth/token"
def get_access_token():
    print("[DEBUG] Requesting access token...")
    try:
        resp = requests.post(
            AUTH_URL,
            data={"grant_type": "client_credentials", "scope": "mcp_api"},
            auth=("test-client", "secret")
        )
        resp.raise_for_status()
        token = resp.json()["access_token"]
        print("[DEBUG] Received access token.")
        return token
    except Exception as e:
        print(f"[FATAL] Could not get access token: {e}")
        sys.exit(1)

DEVICE_CAPABILITIES = {
    "input": ["text", "audio"],
    "output": ["text", "audio"],
    "persona": PERSONA,
    "voice": VOICE,
    "tts_engine": TTS_ENGINE,
    "tts_location": TTS_LOCATION,
}

def find_pi_keyboard():
    for path in list_devices():
        dev = InputDevice(path)
        if "Pi 500" in dev.name and "Keyboard" in dev.name:
            return dev
    return None

class WakeWordDetector:
    def __init__(self):
        print("[DEBUG] Initializing WakeWordDetector...")
        self.model_paths = [str(WAKEWORD_DIR / name) for name in WAKE_WORDS]
        self.sensitivities = ",".join(str(WAKE_WORDS[name]) for name in WAKE_WORDS)
        self.model_names = list(WAKE_WORDS.keys())
        if not Path(WAKEWORD_RESOURCE).exists():
            print(f"[ERROR] Snowboy resource file missing: {WAKEWORD_RESOURCE}")
            raise FileNotFoundError(f"Snowboy resource file missing: {WAKEWORD_RESOURCE}")
        for path in self.model_paths:
            if not Path(path).exists():
                print(f"[ERROR] Wakeword model file missing: {path}")
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
        print("[DEBUG] WakeWordDetector initialized.")

    def detect(self):
        try:
            data = self.stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            result = self.detector.RunDetection(data)
            if result > 0:
                print(f"[DEBUG] Wakeword detected: {self.model_names[result-1]}")
                return self.model_names[result - 1]
            return None
        except Exception as e:
            print(f"[ERROR] Wakeword detection failure: {e}")
            return None

    def cleanup(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()

class PiMCPClient:
    def __init__(self):
        print("[DEBUG] Initializing PiMCPClient...")
        try:
            self.audio_manager = AudioManager()
        except Exception as e:
            print(f"[ERROR] AudioManager failed to initialize: {e}")
            raise

        try:
            self.display_manager = DisplayManager(
                svg_path=config.SVG_PATH,
                boot_img_path=config.BOOT_IMG_PATH,
                window_size=config.WINDOW_SIZE
            )
        except Exception as e:
            print(f"[ERROR] DisplayManager failed to initialize: {e}")
            raise

        try:
            self.stt = VoskTranscriber(config.VOSK_MODEL_PATH)
        except Exception as e:
            print(f"[ERROR] VoskTranscriber failed to initialize: {e}")
            raise

        try:
            self.wakeword = WakeWordDetector()
        except Exception as e:
            print(f"[ERROR] WakeWordDetector failed to initialize: {e}")
            raise

        self.keyboard = find_pi_keyboard()
        if self.keyboard:
            print(f"[DEBUG] Found Pi Keyboard: {self.keyboard}")
        else:
            print("[DEBUG] Pi Keyboard NOT found.")

        self.notification_queue = asyncio.Queue()
        self.persona = PERSONA
        self.voice = VOICE
        self.session_id = None
        self.tts_handler = TTSHandler()
        self.system_manager = ClientSystemManager()
        self.access_token = get_access_token()
        self.server_url = "http://192.168.0.50:8765/"
        print("[DEBUG] PiMCPClient initialized.")

    def listen_keyboard(self) -> bool:
        if not self.keyboard:
            return False
        r, _, _ = select.select([self.keyboard.fd], [], [], 0)
        if not r:
            return False
        for _ in range(5):
            event = self.keyboard.read_one()
            if event and event.type == ecodes.EV_KEY and event.code == 125 and event.value == 1:
                print("[DEBUG] Keyboard wake detected.")
                return True
        return False

    def listen_wakeword(self) -> Optional[str]:
        return self.wakeword.detect()

    async def play_audio(self, audio_bytes):
        fname = "assistant_response.mp3"
        try:
            with open(fname, "wb") as f:
                f.write(audio_bytes)
            await self.audio_manager.queue_audio(audio_file=fname)
            await self.audio_manager.wait_for_audio_completion()
        except Exception as e:
            print(f"[ERROR] Failed to play audio: {e}")

    async def handle_notification(self, msg):
        try:
            await self.display_manager.update_display("speaking", mood="cheerful")
            await self.audio_manager.queue_audio(audio_file=None, generated_text=msg)
            await self.audio_manager.wait_for_audio_completion()
            await self.display_manager.update_display("idle")
        except Exception as e:
            print(f"[ERROR] Failed to handle notification: {e}")

    async def register_device(self):
        register_msg = {
            "name": "register_device",
            "arguments": {
                "device_id": DEVICE_ID,
                "capabilities": DEVICE_CAPABILITIES
            }
        }
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        print("[DEBUG] Register device payload:")
        print(json.dumps(register_msg, indent=2))
        print(f"[DEBUG] Register device POST to {self.server_url}")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.server_url, headers=headers, json=register_msg) as resp:
                    resp_raw = await resp.text()
                    print(f"[DEBUG] Server response status: {resp.status}")
                    print(f"[DEBUG] Server response body: {resp_raw}")
                    if resp.status != 200:
                        print(f"[ERROR] Registration failed: {resp.status} {resp_raw}")
                        return False
                    try:
                        resp_json = json.loads(resp_raw)
                    except Exception as e:
                        print(f"[ERROR] Could not parse registration response: {e}")
                        return False
                    self.session_id = resp_json.get("session_id", None)
                    if self.session_id:
                        print(f"[INFO] Registered, session_id: {self.session_id}")
                        return True
                    else:
                        print(f"[ERROR] Device registration failed: {resp_json}")
                        return False
            except Exception as e:
                print(f"[ERROR] Registration exception: {e}")
                traceback.print_exc()
                return False

    async def send_process_input(self, transcript):
        if not self.session_id:
            print("[ERROR] No session_id, not sending input.")
            return
        request = {
            "name": "run_LAURA",
            "arguments": {
                "session_id": self.session_id,
                "input_type": "text",
                "payload": {"text": transcript},
                "output_mode": [OUTPUT_MODE] if isinstance(OUTPUT_MODE, str) else OUTPUT_MODE,
                "timestamp": datetime.utcnow().isoformat(),
            }
        }
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        print("[DEBUG] Sending process input payload:")
        print(json.dumps(request, indent=2))
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.server_url, headers=headers, json=request) as resp:
                    resp_raw = await resp.text()
                    print(f"[DEBUG] Process input response status: {resp.status}")
                    print(f"[DEBUG] Process input response body: {resp_raw}")
                    if resp.status != 200:
                        print(f"[ERROR] Input POST failed: {resp.status} {resp_raw}")
                        return None
                    try:
                        return json.loads(resp_raw)
                    except Exception as parse_err:
                        print(f"[ERROR] Failed to parse server response: {parse_err}\nRaw: {resp_raw}")
                        return None
            except Exception as e:
                print(f"[ERROR] Failed to send user input to server: {e}")
                traceback.print_exc()
                return None

    async def run(self):
        try:
            reg_ok = await self.register_device()
            if not reg_ok:
                print("[FATAL ERROR] Could not register device. Exiting.")
                return

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
                try:
                    if self.listen_keyboard():
                        print("[INFO] Keyboard wake")
                        wake_model = "keyboard"
                    else:
                        wm = self.listen_wakeword()
                        if wm:
                            print(f"[INFO] Wakeword detected: {wm}")
                            wake_model = wm
                except Exception as e:
                    print(f"[ERROR] Wakeword/keyboard detection failed: {e}")
                    continue

                if not wake_model:
                    await asyncio.sleep(0.05)
                    continue

                await self.display_manager.update_display("listening")
                try:
                    transcript = await capture_speech(self.audio_manager, self.display_manager)
                except Exception as e:
                    print(f"[ERROR] Audio capture failed: {e}")
                    traceback.print_exc()
                    await self.display_manager.update_display("idle")
                    continue

                if not transcript:
                    await self.display_manager.update_display("idle")
                    continue

                # NEW: Check for and handle local commands
                try:
                    is_cmd, cmd_type, arg = self.system_manager.detect_command(transcript)
                    if is_cmd:
                        await self.system_manager.handle_command(cmd_type, arg)
                        await self.display_manager.update_display("idle")
                        continue  # Skip sending to server
                except Exception as e:
                    print(f"[ERROR] Local command detection/handling failed: {e}")
                    traceback.print_exc()
                    await self.display_manager.update_display("idle")
                    continue

                try:
                    response = await self.send_process_input(transcript)
                except Exception as e:
                    print(f"[ERROR] Failed to send input: {e}")
                    traceback.print_exc()
                    response = None

                await self.display_manager.update_display("thinking")
                if not response:
                    await self.display_manager.update_display("idle")
                    continue

                try:
                    if response.get("persona"):
                        self.persona = response["persona"]
                    if response.get("voice"):
                        self.voice = response["voice"]

                    mood = response.get("mood", "casual")
                    await self.display_manager.update_display("speaking", mood=mood)
                    # Handle audio response based on TTS location preference
                    if TTS_LOCATION == "client" and "text" in response and response["text"]:
                        audio_bytes = await self.tts_handler.generate_audio(response["text"])
                        if audio_bytes:
                            await self.play_audio(audio_bytes)
                        else:
                            print("[ERROR] TTSHandler failed to generate audio.")
                    elif "audio" in response and response["audio"]:
                        await self.play_audio(bytes(response["audio"]))
                    elif "text" in response and response["text"]:
                        await self.audio_manager.queue_audio(audio_file=None, generated_text=response["text"])
                        await self.audio_manager.wait_for_audio_completion()
                except Exception as e:
                    print(f"[ERROR] Failed to play/generate assistant response: {e}")
                    traceback.print_exc()

                await self.display_manager.update_display("idle")
        except Exception as e:
            print(f"[FATAL ERROR] Lost connection to MCP server or other failure: {e}")
            traceback.print_exc()

    def cleanup(self):
        try:
            self.wakeword.cleanup()
        except Exception as e:
            print(f"[ERROR] Wakeword cleanup failed: {e}")
        # Add any other cleanup as necessary

async def main():
    client = PiMCPClient()
    try:
        await client.run()
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"[FATAL ERROR] Unhandled exception: {e}")
        traceback.print_exc()
    finally:
        client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
