#!/usr/bin/env python3

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

# ========== SMARTASS IMPORTS ==========
import config
import secret  # All keys: ELEVENLABS_KEY, ANTHROPIC_API_KEY, etc.
from audio_manager_vosk import AudioManager
from vosk_transcriber import VoskTranscriber
from display_manager import DisplayManager

# MCP protocol client imports (replace with your actual MCP client library)
# from mcp.client import MCPClient  # <-- Replace with your actual client
# For demo, use websockets for a simple example
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

# ========== MCP SERVER CONFIG ==========
MCP_SERVER_URI = "ws://localhost:8765"  # Update to your MCP server address/port

# ========== CLIENT LOGIC ==========

def find_pi_keyboard():
    for path in list_devices():
        dev = InputDevice(path)
        if "Pi 500" in dev.name and "Keyboard" in dev.name:
            return dev
    return None

class WakeWordDetector:
    def __init__(self):
        # Build model paths and sensitivities
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
        # Audio and display managers
        self.audio_manager = AudioManager()
        self.display_manager = DisplayManager(
            svg_path=config.SVG_PATH,
            boot_img_path=config.BOOT_IMG_PATH,
            window_size=config.WINDOW_SIZE
        )
        # Vosk Transcriber
        self.stt = VoskTranscriber(config.VOSK_MODEL_PATH)
        # Wakeword and keyboard
        self.wakeword = WakeWordDetector()
        self.keyboard = find_pi_keyboard()
        # Session logging
        self.chat_log_dir = CHAT_LOG_DIR
        os.makedirs(self.chat_log_dir, exist_ok=True)
        # Notification queue
        self.notification_queue = asyncio.Queue()
        # Persona and voice are controlled by server instructions only
        self.persona = PERSONA
        self.voice = VOICE

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

    async def record_and_transcribe(self) -> Optional[str]:
        # Use audio_manager and vosk for robust Pi-optimized recording
        self.stt.reset()
        stream, _ = await self.audio_manager.start_listening()
        voice_detected = False
        start_time = time.time()
        max_length = 15  # seconds
        silence_frames = 0
        min_voice_time = 1.5
        energies = []
        while time.time() - start_time < max_length:
            frame = self.audio_manager.read_audio_frame()
            if not frame:
                await asyncio.sleep(0.01)
                continue
            float_data = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            energy = np.sqrt(np.mean(float_data ** 2)) if len(float_data) else 0
            energies.append(energy)
            if len(energies) > 10:
                energies = energies[-10:]
            is_end, is_speech, partial = self.stt.process_frame(frame)
            if energy > config.VAD_SETTINGS["energy_threshold"]:
                voice_detected = True
                silence_frames = 0
            elif voice_detected:
                silence_frames += 1
            if silence_frames > int(0.7 * AUDIO_SAMPLE_RATE / self.audio_manager.frame_length):
                break
            await asyncio.sleep(0.01)
        await self.audio_manager.stop_listening()
        if not voice_detected:
            return None
        return self.stt.get_final_text().strip()

    async def play_audio(self, audio_bytes):
        # Save and play audio using the audio manager (mp3/wav)
        fname = "assistant_response.mp3"
        with open(fname, "wb") as f:
            f.write(audio_bytes)
        await self.audio_manager.queue_audio(audio_file=fname)
        await self.audio_manager.wait_for_audio_completion()

    async def handle_notification(self, msg):
        # Plays notification and updates display
        await self.display_manager.update_display("speaking", mood="cheerful")
        await self.audio_manager.queue_audio(audio_file=None, generated_text=msg)
        await self.audio_manager.wait_for_audio_completion()
        await self.display_manager.update_display("idle")

    async def run(self):
        # Connect to MCP server
        print(f"Connecting to MCP server at {MCP_SERVER_URI}...")
        async with websockets.connect(MCP_SERVER_URI) as ws:
            print("Connected.")
            await self.display_manager.update_display("idle")
            while True:
                await self.display_manager.update_display("idle")
                # Idle: check for notification
                try:
                    notif = self.notification_queue.get_nowait()
                    await self.handle_notification(notif)
                    continue
                except asyncio.QueueEmpty:
                    pass
                # Listen for wake
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
                # Play wake sound if available
                await self.display_manager.update_display("listening")
                # Record and transcribe
                transcript = await self.record_and_transcribe()
                if not transcript:
                    await self.display_manager.update_display("idle")
                    continue
                await self.save_log("user", transcript)
                # Send to MCP server
                request = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "persona": self.persona,
                    "voice": self.voice,
                    "input_type": "text",
                    "payload": {"text": transcript},
                }
                await ws.send(json.dumps(request))
                await self.display_manager.update_display("thinking")
                # Wait for server response
                response_raw = await ws.recv()
                try:
                    response = json.loads(response_raw)
                except Exception:
                    print("Invalid server response")
                    continue
                # Handle persona/voice/mood update if present
                if response.get("persona"):
                    self.persona = response["persona"]
                if response.get("voice"):
                    self.voice = response["voice"]
                mood = response.get("mood", "casual")
                await self.display_manager.update_display("speaking", mood=mood)
                if "audio" in response and response["audio"]:
                    await self.play_audio(bytes(response["audio"]))
                elif "text" in response and response["text"]:
                    # Optionally run TTS locally if no audio provided
                    await self.audio_manager.queue_audio(audio_file=None, generated_text=response["text"])
                    await self.audio_manager.wait_for_audio_completion()
                await self.display_manager.update_display("idle")
                await self.save_log("assistant", response.get("text", ""))

    def cleanup(self):
        self.wakeword.cleanup()
        self.audio_manager.cleanup()
        self.display_manager.cleanup()
        print("Cleaned up.")

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
