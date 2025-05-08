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
import numpy as np

# ========== SMARTASS IMPORTS ==========
import client_config as config
import secret  # All keys: ELEVENLABS_KEY, ANTHROPIC_API_KEY, etc.
from audio_manager_vosk import AudioManager
from vosk_transcriber import VoskTranscriber
from display_manager import DisplayManager

# Core/shared logic (make sure these exist in core_functions/system_manager)
from core_functions import capture_speech  # robust, centralized speech capture
try:
    from core_functions import save_log  # If centralized, otherwise fallback to in-class
except ImportError:
    save_log = None

# TTS
from elevenlabs.client import ElevenLabs

# Wake word detection
import pyaudio
import snowboydetect
from evdev import InputDevice, list_devices, ecodes

# MCP protocol client
import websockets

# NEW: Import the client system manager
from client_system_manager import ClientSystemManager

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
PIPER_MODEL = config.PIPER_MODEL
PIPER_VOICE = config.PIPER_VOICE
ELEVENLABS_KEY = getattr(secret, "ELEVENLABS_KEY", "")
CLIENT_TTS_ENGINE = config.CLIENT_TTS_ENGINE
MCP_SERVER_URI = config.MCP_SERVER_URI
OUTPUT_MODE = config.OUTPUT_MODE

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
    def detect(self):
        try:
            data = self.stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            result = self.detector.RunDetection(data)
            if result > 0:
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
        self.chat_log_dir = CHAT_LOG_DIR
        try:
            os.makedirs(self.chat_log_dir, exist_ok=True)
        except Exception as e:
            print(f"[ERROR] Failed to create chat log dir: {e}")
        self.notification_queue = asyncio.Queue()
        self.persona = PERSONA
        self.voice = VOICE
        self.session_id = None
        self.init_piper()
        self.eleven = None
        if TTS_LOCATION == "client" and CLIENT_TTS_ENGINE == "elevenlabs" and ELEVENLABS_KEY:
            try:
                self.eleven = ElevenLabs(api_key=ELEVENLABS_KEY)
            except Exception as e:
                print(f"[ERROR] ElevenLabs client failed to initialize: {e}")
        # NEW: Initialize the client system manager
        self.system_manager = ClientSystemManager()

    # Use centralized save_log if available, else fallback
    async def save_log(self, role, content):
        if save_log:
            await save_log({"role": role, "content": content})
            return
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
            except Exception as e:
                print(f"[ERROR] Could not read existing chat log: {e}")
        logs.append(entry)
        try:
            with open(log_file, "w") as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Could not write chat log: {e}")

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

    async def register_device(self, ws):
        register_msg = {
            "tool": "register_device",
            "args": {
                "device_id": DEVICE_ID,
                "capabilities": DEVICE_CAPABILITIES
            }
        }
        try:
            await ws.send(json.dumps(register_msg))
            resp_raw = await ws.recv()
        except Exception as e:
            print(f"[ERROR] Failed to send/receive registration: {e}")
            return False
        try:
            resp = json.loads(resp_raw)
            self.session_id = resp["session_id"]
            print(f"Registered, session_id: {self.session_id}")
            return True
        except Exception as e:
            print(f"[ERROR] Device registration failed: {e}")
            print(f"Server response: {resp_raw}")
            return False

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
        try:
            await ws.send(json.dumps(request))
        except Exception as e:
            print(f"[ERROR] Failed to send user input to server: {e}")

    async def run(self):
        print(f"Connecting to MCP server at {MCP_SERVER_URI}...")
        try:
            async with websockets.connect(MCP_SERVER_URI) as ws:
                print("Connected.")
                reg_ok = await self.register_device(ws)
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
                            print("Keyboard wake")
                            wake_model = "keyboard"
                        else:
                            wm = self.listen_wakeword()
                            if wm:
                                print(f"Wakeword detected: {wm}")
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

                    await self.save_log("user", transcript)
                    try:
                        await self.send_process_input(ws, transcript)
                    except Exception as e:
                        print(f"[ERROR] Failed to send input: {e}")
                        continue
                    await self.display_manager.update_display("thinking")
                    try:
                        response_raw = await ws.recv()
                    except Exception as e:
                        print(f"[ERROR] Failed to receive server response: {e}")
                        await self.display_manager.update_display("idle")
                        continue
                    try:
                        response = json.loads(response_raw)
                    except Exception as e:
                        print(f"[ERROR] Invalid server response: {e}")
                        print(f"Raw response: {response_raw}")
                        continue
                    if response.get("persona"):
                        self.persona = response["persona"]
                    if response.get("voice"):
                        self.voice = response["voice"]
                    mood = response.get("mood", "casual")
                    await self.display_manager.update_display("speaking", mood=mood)
                    try:
                        # Handle audio response based on TTS location preference
                        if TTS_LOCATION == "client" and TTS_ENGINE == "piper" and "text" in response:
                            await self.generate_piper_speech(response["text"])
                        elif "audio" in response and response["audio"]:
                            await self.play_audio(bytes(response["audio"]))
                        elif "text" in response and response["text"]:
                            await self.audio_manager.queue_audio(audio_file=None, generated_text=response["text"])
                            await self.audio_manager.wait_for_audio_completion()
                    except Exception as e:
                        print(f"[ERROR] Failed to play/generate assistant response: {e}")
                    await self.display_manager.update_display("idle")
                    try:
                        await self.save_log("assistant", response.get("text", ""))
                    except Exception as e:
                        print(f"[ERROR] Failed to log assistant response: {e}")
        except Exception as e:
            print(f"[FATAL ERROR] Lost connection to MCP server or other failure: {e}")
            traceback.print_exc()

    def init_piper(self):
        """Initialize Piper TTS if needed"""
        if TTS_LOCATION == "client" and TTS_ENGINE == "piper":
            try:
                subprocess.run(["piper", "--help"], capture_output=True, check=False)
                print("Piper TTS initialized for client-side speech generation")
                return True
            except FileNotFoundError:
                print("[ERROR] Piper not found but client-side TTS requested.")
                return False
        return False

    async def generate_piper_speech(self, text):
        """Generate speech using Piper locally"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_filename = temp_wav.name
        try:
            cmd = ["piper"]
            if PIPER_MODEL:
                cmd.extend(["--model", PIPER_MODEL])
            if PIPER_VOICE:
                cmd.extend(["--voice", PIPER_VOICE])
            cmd.extend(["--output_file", temp_filename])
            try:
                subprocess.run(
                    cmd,
                    input=text.encode(),
                    capture_output=True,
                    check=True
                )
            except Exception as e:
                print(f"[ERROR] Piper TTS failed: {e}")
                return
            await self.audio_manager.queue_audio(audio_file=temp_filename)
            await self.audio_manager.wait_for_audio_completion()
        finally:
            if os.path.exists(temp_filename):
                try:
                    os.unlink(temp_filename)
                except Exception as e:
                    print(f"[ERROR] Failed to clean up temp Piper file: {e}")

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
