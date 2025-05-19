#!/usr/bin/env python3
import numpy as np
import time
import re
import sys
sys.path.append('/home/user/LAURA/snowboy')
import asyncio
import os
import json
import time
import select
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional
from evdev import ecodes

# Import the config module with runtime TTS provider/voice handling
import client_config as config
from client_config import (
    get_active_tts_provider, set_active_tts_provider, get_voice_for_persona,
    load_client_config, save_client_config, client_settings
)
from client_config import client_settings # Import the loaded settings dictionary
from audio_manager_vosk import AudioManager
from vosk_transcriber import VoskTranscriber
from display_manager import DisplayManager
from client_system_manager import ClientSystemManager # Assuming this is your intended system manager
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
        # (Using the __init__ from the previous response that correctly loads client_settings)
        from client_config import SVG_PATH, BOOT_IMG_PATH, WINDOW_SIZE # Assuming these are static
        
        self.audio_manager = AudioManager()
        self.display_manager = DisplayManager(
            svg_path=SVG_PATH,
            boot_img_path=BOOT_IMG_PATH,
            window_size=WINDOW_SIZE
        )

        vosk_model_p = client_settings.get("vosk_model_path")
        if not vosk_model_p:
            from client_config import VOSK_MODEL_PATH as VOSK_MODEL_PATH_DEFAULT
            vosk_model_p = str(VOSK_MODEL_PATH_DEFAULT)
        # VoskTranscriber no longer takes audio_manager in __init__
        self.stt = VoskTranscriber(model_path=vosk_model_p, sample_rate=client_settings.get("audio_sample_rate", 16000))


        self.wakeword = WakeWordDetector() # Assumes it uses client_config internally
        self.keyboard = find_pi_keyboard()
        self.tts_handler = TTSHandler() # Assumes it uses client_config internally

        self.system_manager = ClientSystemManager() # Define or import this correctly

        self.server_url = client_settings.get("mcp_server_uri")
        self.device_id = client_settings.get("device_id")
        self.session_id = None
        self.stt_mode = client_settings.get("stt_mode", "local")
        self.is_follow_up_interaction = False
        
        # Load VAD settings directly here for easy access in capture_speech
        # These are already prioritized (VAD_settings.json > client_settings.json > fallbacks)
        self.vad_params = client_settings.get("vad_settings", {}).copy()
        # Ensure all fallback keys are present in self.vad_params
        _fallback_vad = client_settings.get("_default_config", {}).get("vad_settings", {})
        for key, default_val in _fallback_vad.items():
            if key not in self.vad_params:
                self.vad_params[key] = default_val
        print(f"[PiMCPClient DEBUG] Effective VAD params: {self.vad_params}")


    async def _check_manual_stop(self): # Helper for keyboard check
        if not self.keyboard:
            return False
        # Simplified check for the meta key press
        # Note: select might be problematic in a tight async loop if not handled carefully.
        # For simplicity, this example directly reads if available.
        # Consider if evdev can be used in a truly non-blocking way or if this check needs to be less frequent.
        try:
            for event in self.keyboard.read(): # read() is a generator, might block if no events
                                               # Use read_one() in a loop with select for non-blocking
                if event.type == ecodes.EV_KEY and event.code == ecodes.KEY_LEFTMETA and event.value == 1: # 125 is KEY_LEFTMETA
                    print("[CAPTURE SPEECH] Manual stop via keyboard.")
                    return True
        except BlockingIOError: # No events available
            pass
        except Exception as e:
            print(f"[CAPTURE SPEECH] Keyboard check error: {e}")
        return False

    async def capture_speech_with_vad(self): # Renamed to avoid conflict if old one is kept temporarily
        """
        Captures and transcribes speech using VAD logic adapted from core_functions.py.
        Uses self.stt (VoskTranscriber) for transcription.
        """
        await self.display_manager.update_display("listening")
        self.stt.reset() # Reset transcriber state for a new utterance

        # Determine timeouts based on follow-up state
        # VAD settings are in seconds in JSON, convert to ms if logic needs, or use seconds
        initial_timeout_s_key = "follow_up_initial_listen_timeout_s" if self.is_follow_up_interaction else "initial_listen_timeout_s"
        initial_timeout_s = float(self.vad_params.get(initial_timeout_s_key, 7.0))

        max_rec_time_s_key = "follow_up_max_recording_time_s" if self.is_follow_up_interaction else "max_recording_time"
        max_recording_time_s = float(self.vad_params.get(max_rec_time_s_key, 30.0))

        # Get other VAD parameters (these are directly from VAD_settings.json via client_settings)
        energy_threshold = float(self.vad_params.get("energy_threshold", 0.01))
        continued_ratio = float(self.vad_params.get("continued_threshold_ratio", 0.65))
        silence_duration_s = float(self.vad_params.get("silence_duration", 2.0))
        min_speech_duration_s = float(self.vad_params.get("min_speech_duration", 0.3))
        speech_buffer_time_s = float(self.vad_params.get("speech_buffer_time", 0.2))
        frame_history_length = int(self.vad_params.get("frame_history_length", 10))

        # VAD state variables
        voice_detected_in_vad = False
        vad_is_speaking_state = False # True when VAD thinks speech is active
        speech_start_time_mono = None # Monotonic time when speech segment started
        # last_speech_activity_time_mono = time.monotonic() # Not used in this version of VAD from core_functions
        silence_frames_count = 0
        max_silence_frames = int(silence_duration_s * self.audio_manager.sample_rate / self.audio_manager.frame_length)
        
        frame_history_for_vad = [] # For smoothing energy
        
        # For Vosk partials display (optional)
        # last_partial_time = time.monotonic()
        # partial_display_interval_s = 5 if self.is_follow_up_interaction else 2

        print(f"[CAPTURE SPEECH] Listening {'(follow-up)' if self.is_follow_up_interaction else ''}. Initial Timeout: {initial_timeout_s:.1f}s. Max Rec: {max_recording_time_s:.1f}s")
        print(f"[CAPTURE SPEECH] VAD Params: EnergyThresh={energy_threshold:.4f}, SilenceDur={silence_duration_s:.1f}s, MinSpeech={min_speech_duration_s:.1f}s")

        overall_listen_start_time_mono = time.monotonic()
        
        transcript = None
        
        try:
            await self.audio_manager.start_listening()
            if not self.audio_manager.audio_stream:
                print("[CAPTURE SPEECH ERROR] Audio stream not available.")
                return None

            while True:
                current_time_mono = time.monotonic()

                # 1. Check for initial timeout (if no voice detected yet by VAD)
                if not voice_detected_in_vad and (current_time_mono - overall_listen_start_time_mono) > initial_timeout_s:
                    print(f"[CAPTURE SPEECH] VAD initial timeout ({initial_timeout_s:.1f}s). No voice detected.")
                    # Update display based on context (from core_functions)
                    # await self.display_manager.update_display('sleep' if not self.is_follow_up_interaction else 'idle')
                    break # Exit listening loop

                # 2. Read audio frame
                pcm_bytes = self.audio_manager.read_audio_frame()
                if not pcm_bytes:
                    await asyncio.sleep(0.01) # Wait briefly if no data
                    continue
                
                # 3. Manual stop check (if voice has been detected)
                if voice_detected_in_vad:
                    if await self._check_manual_stop():
                        if speech_start_time_mono and (current_time_mono - speech_start_time_mono) > min_speech_duration_s:
                            print("[CAPTURE SPEECH] Manual stop triggered.")
                            await asyncio.sleep(speech_buffer_time_s) # Record buffer
                            break
                        else:
                            print("[CAPTURE SPEECH] Manual stop but recording too short, continuing.")
                            # manual_stop = False # Reset if needed by more complex logic

                # 4. Process with Vosk (feed every frame)
                # is_vosk_final_chunk, is_speech_in_vosk_frame, vosk_partial_text = self.stt.process_frame(pcm_bytes)
                self.stt.process_frame(pcm_bytes) # process_frame updates self.stt.partial_text and self.stt.complete_text

                # 5. VAD Energy Calculation (from core_functions)
                float_data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                energy = np.sqrt(np.mean(float_data**2)) if len(float_data) > 0 else 0.0
                
                frame_history_for_vad.append(energy)
                if len(frame_history_for_vad) > frame_history_length:
                    frame_history_for_vad.pop(0)
                avg_energy = sum(frame_history_for_vad) / len(frame_history_for_vad) if frame_history_for_vad else 0.0

                # 6. VAD State Machine (from core_functions)
                if not vad_is_speaking_state: # If VAD currently thinks it's silence
                    if avg_energy > energy_threshold: # Speech just started according to VAD
                        print(f"[CAPTURE SPEECH] VAD: Voice started. Energy: {avg_energy:.4f} (Thresh: {energy_threshold:.4f})")
                        voice_detected_in_vad = True # Mark that voice was detected at least once
                        vad_is_speaking_state = True
                        speech_start_time_mono = current_time_mono # Record VAD speech start time
                        silence_frames_count = 0
                else: # If VAD currently thinks it's speech
                    current_speech_duration_s = current_time_mono - speech_start_time_mono

                    if avg_energy > (energy_threshold * continued_ratio): # Speech continues
                        silence_frames_count = 0
                    else: # Potential silence during speech
                        silence_frames_count += 1
                    
                    # Check for end conditions
                    if silence_frames_count >= max_silence_frames and current_speech_duration_s > min_speech_duration_s:
                        print(f"[CAPTURE SPEECH] VAD: End of speech by silence. Duration: {current_speech_duration_s:.2f}s")
                        await asyncio.sleep(speech_buffer_time_s) # Record buffer
                        break # Exit listening loop

                    if current_speech_duration_s > max_recording_time_s:
                        print(f"[CAPTURE SPEECH] VAD: Max recording time ({max_recording_time_s:.1f}s) reached.")
                        break # Exit listening loop
                
                # Optional: Display partial results (less frequently for follow-ups)
                # if vosk_partial_text and (current_time_mono - last_partial_time) > partial_display_interval_s:
                #     last_partial_time = current_time_mono
                #     print(f"Partial: {vosk_partial_text}")

                await asyncio.sleep(0.01) # Small yield

            # --- End of listening loop ---

            if not voice_detected_in_vad: # If VAD never detected speech above threshold
                # Update display state based on context (from core_functions)
                await self.display_manager.update_display('sleep' if not self.is_follow_up_interaction else 'idle')
                return None

            # Get final transcription from Vosk
            transcript = self.stt.get_final_text()
            print(f"[CAPTURE SPEECH] Raw transcript from Vosk: '{transcript}'")

            # Apply post-processing (from core_functions)
            if transcript:
                transcript = re.sub(r'^(that were|that was)\s+', '', transcript, flags=re.IGNORECASE)
                
                words = transcript.split()
                num_words = len(words)
                
                min_words_key = "vosk_reject_min_words" # Assuming same for follow-up from vad_params
                min_words_to_accept = int(self.vad_params.get(min_words_key, 2))
                min_chars_single_word = int(self.vad_params.get("vosk_reject_min_chars_if_single_word", 4))

                if num_words == 0: # Handle empty transcript after regex
                    print("[CAPTURE SPEECH] Transcript empty after cleaning.")
                    transcript = None
                elif num_words < min_words_to_accept:
                    print(f"[CAPTURE SPEECH] Discarding (words): '{transcript}' ({num_words} < {min_words_to_accept})")
                    transcript = None
                elif num_words == 1 and len(words[0]) < min_chars_single_word:
                    print(f"[CAPTURE SPEECH] Discarding (chars): '{transcript}' ({len(words[0])} < {min_chars_single_word})")
                    transcript = None
            else: # No transcript from Vosk at all
                print("[CAPTURE SPEECH] No transcript returned by Vosk.")
                transcript = None
            
            if transcript:
                print(f"[CAPTURE SPEECH] Final processed transcript: '{transcript}'")
                return transcript.strip()
            else:
                # If transcript became None after processing, ensure display is updated
                await self.display_manager.update_display('idle' if self.is_follow_up_interaction else 'sleep')
                return None

        except Exception as e:
            print(f"[ERROR] Speech capture failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            await self.audio_manager.stop_listening()
            # Ensure display returns to a sensible state if loop broke unexpectedly
            # Or let the main run loop handle this. For now, capture_speech focuses on capture.

    async def capture_speech(self):
        return await self.capture_speech_with_vad()

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

    async def run(self):
        # DEVICE_ID should be sourced from self.device_id, which is set in __init__ from client_settings
        # from client_config import DEVICE_ID # This direct import might not be needed if self.device_id is used

        try:
            print(f"[INFO] Attempting to connect to MCP server at {self.server_url}")
            async with sse_client(
                self.server_url,
                headers={} # Add any necessary headers, e.g., for auth
            ) as (read, write):
                print("[INFO] SSE client connected. Initializing ClientSession...")
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    print(f"[INFO] ClientSession initialized. Registering device: {self.device_id}")
                    
                    # Use self.device_id which is loaded from client_settings in __init__
                    register_result = await session.call_tool(
                        "register_device",
                        arguments={
                            "device_id": self.device_id,
                            "capabilities": {
                                "input": ["text", "audio"],
                                "output": ["text", "audio"],
                                # Source tts_mode and provider from client_settings via helpers or direct access
                                "tts_mode": client_settings.get("tts_mode", "api"),
                                "api_tts_provider": get_active_tts_provider(),
                            }
                        }
                    )
                    
                    session_info = None
                    if hasattr(register_result, "content") and \
                       isinstance(register_result.content, list) and \
                       len(register_result.content) > 0:
                        first_content = register_result.content[0]
                        if hasattr(first_content, "text") and isinstance(first_content.text, str):
                            try:
                                session_info = json.loads(first_content.text)
                                print(f"[INFO] Device registration successful. Session Info: {session_info}")
                            except json.JSONDecodeError as e:
                                print(f"[FATAL ERROR] Could not parse session info JSON from server: '{first_content.text}'. Error: {e}")
                                return
                            except Exception as e: # Catch any other parsing errors
                                print(f"[FATAL ERROR] Unexpected error parsing session info JSON: {e}")
                                traceback.print_exc()
                                return
                        else:
                            print("[FATAL ERROR] Registration response content item has no 'text' field or not a string.")
                            return
                    else:
                        print(f"[FATAL ERROR] Registration failed or no valid content in server response. Response: {register_result}")
                        return

                    if not isinstance(session_info, dict) or "session_id" not in session_info:
                        print(f"[FATAL ERROR] Registration failed (no session_id in parsed server response). Parsed Info: {session_info}")
                        return
                    self.session_id = session_info["session_id"]
                    print(f"[INFO] Session ID obtained: {self.session_id}")

                    await self.display_manager.update_display("idle")
                    
                    while True:
                        # Set display to idle at the start of each loop iteration if not already listening for wakeword
                        # await self.display_manager.update_display("idle") # This might be too frequent, handled by context

                        wake_model = None
                        # Reset follow-up state if we are waiting for a new wake event
                        # self.is_follow_up_interaction = False # Moved this to after wake detection

                        try:
                            if self.listen_keyboard(): # Assuming listen_keyboard is synchronous
                                wake_model = "keyboard"
                                print("[INFO] Keyboard wake detected.")
                            else:
                                wm = self.listen_wakeword() # Assuming listen_wakeword is synchronous
                                if wm:
                                    wake_model = wm
                                    print(f"[INFO] Wakeword '{wm}' detected.")
                        except Exception as e:
                            print(f"[ERROR] Wakeword/keyboard detection failed: {e}")
                            traceback.print_exc()
                            await asyncio.sleep(0.1) # Avoid tight loop on error
                            continue

                        if not wake_model:
                            await asyncio.sleep(0.05) # Standard sleep when idle and no wake event
                            continue
                        
                        # A wake event occurred, so the next interaction is NOT a follow-up from a previous turn
                        self.is_follow_up_interaction = False
                        print(f"[INFO] Wake event processed. is_follow_up_interaction set to: {self.is_follow_up_interaction}")

                        # capture_speech_with_vad will set display to "listening"
                        try:
                            transcript = await self.capture_speech_with_vad() # This now uses the VAD logic
                        except Exception as e:
                            print(f"[ERROR] Audio capture failed unexpectedly in main loop: {e}")
                            traceback.print_exc()
                            await self.display_manager.update_display("idle")
                            continue

                        if not transcript:
                            print("[INFO] No transcript captured or speech too short/invalid.")
                            await self.display_manager.update_display("idle")
                            self.is_follow_up_interaction = False # Reset if capture failed
                            continue
                        
                        print(f"[INFO] Captured transcript: '{transcript}'")

                        # System command detection and runtime switching
                        # Ensure detect_system_command is defined or imported correctly
                        is_cmd, cmd_type, arg = detect_system_command(transcript)
                        if is_cmd:
                            print(f"[INFO] Detected system command: {cmd_type} with arg: {arg}")
                            if cmd_type == "switch_tts_mode":
                                client_settings["tts_mode"] = arg # Update the live settings dict
                                save_client_config(client_settings) # Persist the change
                                print(f"[INFO] TTS mode switched to: {arg}")
                            elif cmd_type == "switch_api_tts_provider":
                                set_active_tts_provider(arg) # This helper saves the config
                                print(f"[INFO] API TTS provider switched to: {arg}")
                            elif cmd_type == "switch_stt_mode":
                                self.stt_mode = arg # Update runtime state
                                client_settings["stt_mode"] = arg # Update live settings
                                save_client_config(client_settings) # Persist
                                print(f"[INFO] STT mode switched to: {arg}")
                            # else: # Removed system_manager.handle_command as it was not defined/used in original snippet
                                # await self.system_manager.handle_command(cmd_type, arg)
                            await self.display_manager.update_display("idle")
                            self.is_follow_up_interaction = False # System command resets follow-up
                            continue

                        # --- Call LAURA Tool ---
                        response_from_laura_tool = None 
                        try:
                            await self.display_manager.update_display("thinking")
                            print(f"[INFO] Sending to LAURA tool: '{transcript}'")
                            
                            # The server expects output_mode to be a list.
                            # If your client primarily handles text output from LAURA and does its own TTS,
                            # ["text"] is a likely valid value.
                            expected_output_mode_list = ["text"] 

                            response_from_laura_tool = await session.call_tool(
                                "run_LAURA",
                                arguments={
                                    "session_id": self.session_id,
                                    "input_type": "text", 
                                    "payload": {"text": transcript},
                                    "timestamp": datetime.utcnow().isoformat() + "Z",
                                    # *** FIX: Send as a list ***
                                    "output_mode": expected_output_mode_list
                                }
                            )
                        except Exception as e:
                            print(f"[ERROR] Failed to send input to LAURA tool: {e}")
                            traceback.print_exc()
                            await self.display_manager.update_display("idle")
                            self.is_follow_up_interaction = False 
                            continue 
                        
                        laura_response_data = None 
                        first_content_item = response_from_laura_tool.content[0]

                        if hasattr(first_content_item, 'text') and isinstance(first_content_item.text, str):
                            try:
                                laura_response_data = json.loads(first_content_item.text)
                                print(f"[INFO] LAURA response data (parsed JSON): {laura_response_data}")
                            except json.JSONDecodeError as e:
                                print(f"[ERROR] Could not parse LAURA response JSON from server: '{first_content_item.text}'. Error: {e}")
                            except Exception as e:
                                print(f"[ERROR] Unexpected error parsing LAURA response JSON: {e}")
                                traceback.print_exc()
                        else:
                            print("[ERROR] LAURA response content item has no 'text' field or not a string.")
                        
                        if not isinstance(laura_response_data, dict):
                            print(f"[ERROR] Parsed LAURA response is not a dictionary. Type: {type(laura_response_data)}, Value: {laura_response_data}")
                            await self.display_manager.update_display("idle")
                            self.is_follow_up_interaction = False # Failed parsing
                            continue

                        # --- Successfully Parsed LAURA Response ---
                        active_persona = laura_response_data.get("active_persona", "laura")
                        provider = get_active_tts_provider()
                        voice = get_voice_for_persona(provider, active_persona)

                        try:
                            mood = laura_response_data.get("mood", "casual")
                            await self.display_manager.update_display("speaking", mood=mood)
                            
                            tts_mode = client_settings.get("tts_mode", "api")
                            response_text_to_speak = laura_response_data.get("text")

                            if not response_text_to_speak:
                                print("[WARN] LAURA response has no 'text' field for TTS/display.")
                                self.is_follow_up_interaction = False # No text means conversation might be over
                            else:
                                if tts_mode == "local":
                                    audio_bytes = await self.tts_handler.generate_audio_local(response_text_to_speak)
                                    if audio_bytes: await self.play_audio(audio_bytes)
                                    else: print("[ERROR] Local TTS failed to generate audio.")
                                elif tts_mode == "api":
                                    # ... (elevenlabs, cartesia logic as before)
                                    if provider == "elevenlabs":
                                        audio_bytes = await self.tts_handler.generate_audio_elevenlabs(response_text_to_speak, voice)
                                    elif provider == "cartesia":
                                        audio_bytes = await self.tts_handler.generate_audio_cartesia(response_text_to_speak, voice)
                                    else:
                                        print(f"[ERROR] Unknown API TTS provider: {provider}")
                                        audio_bytes = None
                                    if audio_bytes: await self.play_audio(audio_bytes)
                                    else: print("[ERROR] API TTS failed to generate audio.")
                                elif tts_mode == "text":
                                    print(f"Assistant ({active_persona}): {response_text_to_speak}")
                                
                                # If TTS was attempted (even if failed) or text displayed, consider it a turn taken.
                                self.is_follow_up_interaction = True # Next interaction is a follow-up
                                print(f"[INFO] Interaction complete. is_follow_up_interaction set to: {self.is_follow_up_interaction}")

                        except Exception as e:
                            print(f"[ERROR] Failed to process or play assistant response: {e}")
                            traceback.print_exc()
                            self.is_follow_up_interaction = False # Error during TTS resets follow-up

                        await self.display_manager.update_display("idle")
        
        except ConnectionRefusedError:
            print(f"[FATAL ERROR] Connection refused to MCP server at {self.server_url}. Is it running?")
            traceback.print_exc()
        except (ConnectionError, Exception) as e:
            print(f"[FATAL ERROR] WebSocket connection closed unexpectedly: {e}")
            traceback.print_exc()
        except Exception as e: # Catch-all for other errors in the main try block
            print(f"[FATAL ERROR] Lost connection to MCP server or other critical failure in run loop: {e}")
            traceback.print_exc()
        finally:
            print("[INFO] Exiting run loop or run loop attempt.")
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
