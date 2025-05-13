#!/usr/bin/env python3

import asyncio
import os
import json
import time
import select
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import the config module with runtime TTS provider/voice handling
import client_config as config
from client_config import (
    get_active_tts_provider, set_active_tts_provider, get_voice_for_persona,
    load_client_config, save_client_config,
)
from audio_manager_vosk import AudioManager
from vosk_transcriber import VoskTranscriber
from display_manager import DisplayManager
from client_system_manager import ClientSystemManager
from client_tts_handler import TTSHandler

import pyaudio
from evdev import InputDevice, list_devices, ecodes

from mcp import ClientSession
from mcp.client.sse import sse_client

# Utility: Find the Pi 500 keyboard
def find_pi_keyboard():
    for path in list_devices():
        dev = InputDevice(path)
        if "Pi 500" in dev.name and "Keyboard" in dev.name:
            return dev
    return None

# Wakeword detector class (same as before)
class WakeWordDetector:
    def __init__(self):
        from client_config import WAKEWORD_DIR, WAKEWORD_RESOURCE, WAKE_WORDS, AUDIO_SAMPLE_RATE, AUDIO_CHUNK
        self.model_paths = [str(WAKEWORD_DIR / name) for name in WAKE_WORDS]
        self.sensitivities = ",".join(str(WAKE_WORDS[name]) for name in WAKE_WORDS)
        self.model_names = list(WAKE_WORDS.keys())
        import snowboydetect
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            rate=AUDIO_SAMPLE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=AUDIO_CHUNK
        )
        self.detector = snowboydetect.SnowboyDetect(
            resource_filename=str(WAKEWORD_RESOURCE).encode(),
            model_str=",".join(self.model_paths).encode()
        )
        self.detector.SetSensitivity(self.sensitivities.encode())

    def detect(self):
        from client_config import AUDIO_CHUNK
        try:
            data = self.stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            result = self.detector.RunDetection(data)
            if result > 0:
                return self.model_names[result - 1]
            return None
        except Exception:
            return None

    def cleanup(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()

# System command detection
def detect_system_command(transcript):
    t = transcript.lower()
    if "enable remote tts" in t or "api tts" in t:
        return True, "switch_tts_mode", "api"
    elif "enable local tts" in t or "local tts" in t:
        return True, "switch_tts_mode", "local"
    elif "text only mode" in t or "text only" in t:
        return True, "switch_tts_mode", "text"
    elif "switch tts provider to cartesia" in t:
        return True, "switch_api_tts_provider", "cartesia"
    elif "switch tts provider to elevenlabs" in t:
        return True, "switch_api_tts_provider", "elevenlabs"
    elif "enable remote transcription" in t or "remote transcription" in t:
        return True, "switch_stt_mode", "remote"
    elif "enable local transcription" in t or "local transcription" in t:
        return True, "switch_stt_mode", "local"
    return False, None, None

class PiMCPClient:
    def __init__(self):
        from client_config import (
            SVG_PATH, BOOT_IMG_PATH, WINDOW_SIZE, VOSK_MODEL_PATH, DEVICE_ID, MCP_SERVER_URI
        )
        self.audio_manager = AudioManager()
        self.display_manager = DisplayManager(
            svg_path=SVG_PATH,
            boot_img_path=BOOT_IMG_PATH,
            window_size=WINDOW_SIZE
        )
        self.stt = VoskTranscriber(VOSK_MODEL_PATH)
        self.wakeword = WakeWordDetector()
        self.keyboard = find_pi_keyboard()
        self.tts_handler = TTSHandler()
        self.system_manager = ClientSystemManager()
        self.server_url = MCP_SERVER_URI
        self.session_id = None
        self.stt_mode = config._client_config.get("stt_mode", "local")

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

    async def capture_speech(self):
        try:
            await self.display_manager.update_display("listening")
            transcript = await self.audio_manager.capture_speech_vosk(self.stt)
            return transcript
        except Exception as e:
            print(f"[ERROR] Speech capture failed: {e}")
            return None

    async def run(self):
        from client_config import DEVICE_ID
        try:
            async with sse_client(
                self.server_url,
                headers={}
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    register_result = await session.call_tool(
                        "register_device",
                        arguments={
                            "device_id": DEVICE_ID,
                            "capabilities": {
                                "input": ["text", "audio"],
                                "output": ["text", "audio"],
                                "tts_mode": config._client_config.get("tts_mode", "api"),
                                "api_tts_provider": get_active_tts_provider(),
                            }
                        }
                    )
                    session_info = None
                    if getattr(register_result, "content", None) and len(register_result.content) > 0:
                        first_content = register_result.content[0]
                        if hasattr(first_content, "text"):
                            try:
                                session_info = json.loads(first_content.text)
                            except Exception as e:
                                print(f"[FATAL ERROR] Could not parse session info JSON: {e}")
                                return
                    if not session_info or "session_id" not in session_info:
                        print("[FATAL ERROR] Registration failed (no session_id in server response).")
                        return
                    self.session_id = session_info["session_id"]

                    await self.display_manager.update_display("idle")
                    while True:
                        await self.display_manager.update_display("idle")

                        wake_model = None
                        try:
                            if self.listen_keyboard():
                                wake_model = "keyboard"
                            else:
                                wm = self.listen_wakeword()
                                if wm:
                                    wake_model = wm
                        except Exception as e:
                            print(f"[ERROR] Wakeword/keyboard detection failed: {e}")
                            continue

                        if not wake_model:
                            await asyncio.sleep(0.05)
                            continue

                        await self.display_manager.update_display("listening")
                        try:
                            transcript = await self.capture_speech()
                        except Exception as e:
                            print(f"[ERROR] Audio capture failed: {e}")
                            await self.display_manager.update_display("idle")
                            continue

                        if not transcript:
                            await self.display_manager.update_display("idle")
                            continue

                        # System command detection and runtime switching
                        is_cmd, cmd_type, arg = detect_system_command(transcript)
                        if is_cmd:
                            if cmd_type == "switch_tts_mode":
                                config._client_config["tts_mode"] = arg
                                save_client_config(config._client_config)
                                print(f"[INFO] TTS mode switched to: {arg}")
                            elif cmd_type == "switch_api_tts_provider":
                                set_active_tts_provider(arg)
                                print(f"[INFO] API TTS provider switched to: {arg}")
                            elif cmd_type == "switch_stt_mode":
                                self.stt_mode = arg
                                config._client_config["stt_mode"] = arg
                                save_client_config(config._client_config)
                                print(f"[INFO] STT mode switched to: {arg}")
                            else:
                                await self.system_manager.handle_command(cmd_type, arg)
                            await self.display_manager.update_display("idle")
                            continue

                        try:
                            await self.display_manager.update_display("thinking")
                            response = await session.call_tool(
                                "run_LAURA",
                                arguments={
                                    "session_id": self.session_id,
                                    "input_type": "text",
                                    "payload": {"text": transcript},
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                            )
                        except Exception as e:
                            print(f"[ERROR] Failed to send input: {e}")
                            traceback.print_exc()
                            response = None

                        if not response:
                            await self.display_manager.update_display("idle")
                            continue

                        # Get active persona from the server response
                        active_persona = response.get("active_persona", "laura")
                        provider = get_active_tts_provider()
                        voice = get_voice_for_persona(provider, active_persona)

                        # TTS/Output
                        try:
                            mood = response.get("mood", "casual")
                            await self.display_manager.update_display("speaking", mood=mood)
                            tts_mode = config._client_config.get("tts_mode", "api")
                            if tts_mode == "local" and "text" in response and response["text"]:
                                audio_bytes = await self.tts_handler.generate_audio_local(response["text"])
                                if audio_bytes:
                                    await self.play_audio(audio_bytes)
                                else:
                                    print("[ERROR] Local TTS failed to generate audio.")
                            elif tts_mode == "api" and "text" in response and response["text"]:
                                if provider == "elevenlabs":
                                    audio_bytes = await self.tts_handler.generate_audio_elevenlabs(response["text"], voice)
                                elif provider == "cartesia":
                                    audio_bytes = await self.tts_handler.generate_audio_cartesia(response["text"], voice)
                                else:
                                    print(f"[ERROR] Unknown API TTS provider: {provider}")
                                    audio_bytes = None
                                if audio_bytes:
                                    await self.play_audio(audio_bytes)
                                else:
                                    print("[ERROR] API TTS failed to generate audio.")
                            elif tts_mode == "text" and "text" in response and response["text"]:
                                print("Assistant:", response["text"])
                            else:
                                print("[WARN] No valid response to play or display.")
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
