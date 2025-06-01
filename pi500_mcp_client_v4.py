#!/usr/bin/env python3

import asyncio
import json
import os
import tempfile
import traceback
from pathlib import Path
import time # For monotonic clock in VAD
from datetime import datetime # For timestamp in send_to_laura_tool
import re
import select # For non-blocking keyboard check

# MCP Imports
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

# Local Imports
from audio_manager_vosk import AudioManager
from client_tts_handler import TTSHandler
from display_manager import DisplayManager
from vosk_transcriber import VoskTranscriber

# Evdev for Pi 500 Keyboard
try:
    from evdev import InputDevice, list_devices, ecodes
except ImportError:
    print("[WARN] evdev module not found. Pi 500 keyboard specific features will be disabled.")
    InputDevice = None
    list_devices = lambda: []
    ecodes = None

# PyAudio for Snowboy
try:
    import pyaudio
except ImportError:
    print("[WARN] PyAudio module not found. Wakeword detection will be disabled.")
    pyaudio = None

# Snowboy
try:
    import snowboydetect # Assuming snowboydetect.py and _snowboydetect.so are in PYTHONPATH
except ImportError:
    print("[WARN] snowboydetect module not found. Wakeword detection will be disabled.")
    snowboydetect = None


from client_config import (
    SERVER_URL, DEVICE_ID, KEEP_TEMP_AUDIO_FILES, # DEFAULT_PERSONA REMOVED
    VAD_SETTINGS, VOSK_MODEL_PATH, # VOSK_MODEL_PATH is often part of VAD_SETTINGS
    AUDIO_SAMPLE_RATE, SNOWBOY_AUDIO_CHUNK_SIZE,
    WAKEWORD_MODEL_DIR, WAKEWORD_RESOURCE_FILE, WAKE_WORDS_AND_SENSITIVITIES,
    # DISPLAY_SVG_PATH, DISPLAY_BOOT_IMG_PATH, DISPLAY_WINDOW_SIZE REMOVED
    # QUERY_FILES_DIR, QUERY_OFFLOAD_DIR REMOVED
    client_settings, save_client_settings, get_active_tts_provider, set_active_tts_provider, load_client_settings
)
# from client_secret import YOUR_API_KEY # If needed directly by client

# For VAD
import numpy as np

# Utility: Find the Pi 500 keyboard (from v2)
def find_pi_keyboard():
    if not InputDevice: return None
    for path_str in list_devices():
        try:
            dev = InputDevice(path_str)
            if "Pi 500" in dev.name and "Keyboard" in dev.name:
                print(f"[INFO] Found Pi 500 Keyboard: {dev.name} at {dev.path}")
                return dev
        except Exception as e:
            print(f"[WARN] Could not inspect device {path_str}: {e}")
    print("[WARN] Pi 500 Keyboard not found.")
    return None

# Wakeword detector class (from v2, adapted)
class WakeWordDetector:
    def __init__(self):
        self.model_paths = [str(Path(WAKEWORD_MODEL_DIR) / name) for name in WAKE_WORDS_AND_SENSITIVITIES.keys()]
        self.model_names = list(WAKE_WORDS_AND_SENSITIVITIES.keys())
        self.sensitivities_list = [str(s) for s in WAKE_WORDS_AND_SENSITIVITIES.values()]

        self.detector = None
        self.pa = None
        self.stream = None

        if not snowboydetect or not pyaudio:
            print("[ERROR] Snowboy or PyAudio not available. Wakeword detection disabled.")
            return

        if not Path(WAKEWORD_RESOURCE_FILE).exists():
            print(f"[ERROR] Snowboy resource file not found: {WAKEWORD_RESOURCE_FILE}")
            return
        for model_path in self.model_paths:
            if not Path(model_path).exists():
                print(f"[ERROR] Snowboy model file not found: {model_path}")
                return

        try:
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(
                rate=AUDIO_SAMPLE_RATE,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=SNOWBOY_AUDIO_CHUNK_SIZE,
            )
            self.detector = snowboydetect.SnowboyDetect(
                resource_filename=str(WAKEWORD_RESOURCE_FILE).encode(),
                model_str=",".join(self.model_paths).encode()
            )
            self.detector.SetSensitivity(",".join(self.sensitivities_list).encode())
            # self.detector.SetAudioGain(1.0) # Optional: Adjust audio gain
            print("[INFO] WakeWordDetector initialized successfully with Snowboy.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize WakeWordDetector: {e}")
            traceback.print_exc()
            self.detector = None
            self.stream = None
            if self.pa: self.pa.terminate()
            self.pa = None

    def detect(self) -> str | None: # Synchronous
        if not self.detector or not self.stream:
            return None
        try:
            data = self.stream.read(SNOWBOY_AUDIO_CHUNK_SIZE, exception_on_overflow=False)
            result = self.detector.RunDetection(data)
            if result > 0:
                return self.model_names[result - 1] # Map index back to model name
            return None
        except IOError as e:
            if hasattr(pyaudio, 'paInputOverflowed') and e.errno == pyaudio.paInputOverflowed:
                # print("[WARN] WakeWordDetector: Input overflowed.") # Can be noisy
                return None
            else:
                print(f"[ERROR] WakeWordDetector IOError: {e}")
                return None
        except Exception as e:
            print(f"[ERROR] WakeWordDetector exception: {e}")
            # traceback.print_exc() # Can be noisy
            return None

    def cleanup(self):
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"[ERROR] Error closing wakeword stream: {e}")
        if self.pa:
            try:
                self.pa.terminate()
            except Exception as e:
                print(f"[ERROR] Error terminating PyAudio for wakeword: {e}")
        self.detector = None # Important to free Snowboy resources if it has a __del__ or similar
        print("[INFO] WakeWordDetector cleaned up.")

# System command detection (from v2, adapted for client_settings)
def detect_system_command(transcript: str) -> tuple[bool, str | None, str | None]:
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
    # Add more system commands as needed (e.g., STT mode)
    # elif "enable remote transcription" in t: return True, "switch_stt_mode", "remote"
    # elif "enable local transcription" in t: return True, "switch_stt_mode", "local"
    return False, None, None


class PiMCPClient:
    def __init__(self, server_url: str, device_id: str):
        self.server_url = server_url
        self.device_id = device_id
        self.session_id: str | None = None
        self.active_persona: str = "client_default" # Default client persona, will be updated by server response
        self.is_follow_up_interaction: bool = False
        self.debug_keep_audio_files = KEEP_TEMP_AUDIO_FILES
        self.mcp_session: ClientSession | None = None

        # Initialize core components
        self.audio_manager = AudioManager(sample_rate=AUDIO_SAMPLE_RATE) # Ensure AudioManager uses the rate
        self.tts_handler = TTSHandler() # Relies on client_settings for mode/provider
        
        # DisplayManager setup: fetch paths from client_settings
        self.display_manager = DisplayManager(
            svg_path=client_settings.get("DISPLAY_SVG_PATH"),
            boot_img_path=client_settings.get("DISPLAY_BOOT_IMG_PATH"),
            window_size=client_settings.get("DISPLAY_WINDOW_SIZE")
        )
        self.wakeword_detector = WakeWordDetector()
        self.keyboard_device = find_pi_keyboard()

        if self.keyboard_device and ecodes:
            print(f"[INFO] Keyboard diagnostic: Found {self.keyboard_device.name}, expects Left Meta Key code: {ecodes.KEY_LEFTMETA}")
        else:
            print("[WARN] Keyboard diagnostic: Pi 500 Keyboard not fully available. Meta key features might be limited.")


        # Use VOSK_MODEL_PATH from config, or path within VAD_SETTINGS
        effective_vosk_model_path = VAD_SETTINGS.get("vosk_model_path", VOSK_MODEL_PATH)
        if isinstance(effective_vosk_model_path, Path):
            effective_vosk_model_path_str = str(effective_vosk_model_path)
        else:
            effective_vosk_model_path_str = effective_vosk_model_path # Already a string or something else
        if not Path(effective_vosk_model_path).exists():
            print(f"[WARN] Vosk model path '{effective_vosk_model_path}' does not exist. Please check client_config.py.")
        self.transcriber = VoskTranscriber(model_path=effective_vosk_model_path_str, sample_rate=AUDIO_SAMPLE_RATE)

        print(f"[PiMCPClient DEBUG] Effective VAD params: {VAD_SETTINGS}")

        # Conversation hook phrases (combined from v2 and v3)
        self.continuation_phrases = [
            "let me know", "tell me more", "what else", "anything else",
            "can i help you with anything else", "do you need more information",
            "share your thoughts", "i'm listening", "go ahead",
            "feel free to elaborate", "i'd like to hear more", "please continue",
            "what do you think", "how does that sound", "what's next",
            ".", # Makes hook detection very frequent
        ]


    async def initialize_session(self): # Removed read, write params
        """Initializes the client session with the MCP server."""
        try:
            if not self.mcp_session:
                print("[ERROR] MCP session object not available for device registration.")
                return False
                
            await asyncio.sleep(0.1) # 100 milliseconds
            print("[INFO] Small delay added before first tool call.")

            registration_payload = {
                "device_id": self.device_id,
                "capabilities": {
                    "input": ["text", "audio"],
                    "output": ["text", "audio"],
                    "tts_mode": client_settings.get("tts_mode", "api"),
                    "api_tts_provider": get_active_tts_provider(),
                }
            }
            print(f"[INFO] Calling 'register_device' tool with payload: {registration_payload}")
            response_obj = await self.mcp_session.call_tool("register_device", arguments=registration_payload)

            response_data = None
            if hasattr(response_obj, 'content') and isinstance(response_obj.content, list) and response_obj.content:
                if hasattr(response_obj.content[0], 'text') and isinstance(response_obj.content[0].text, str):
                    try:
                        response_data = json.loads(response_obj.content[0].text)
                    except json.JSONDecodeError:
                        print(f"[ERROR] Failed to parse JSON from register_device response: {response_obj.content[0].text}")
                        return False
                elif isinstance(response_obj.content[0], dict):
                         response_data = response_obj.content[0]
            elif isinstance(response_obj, dict):
                response_data = response_obj

            if response_data and response_data.get("session_id"):
                self.session_id = response_data["session_id"]
                self.capabilities = response_data.get("capabilities", registration_payload["capabilities"])
                print(f"[INFO] Device registration successful. Session ID: {self.session_id}, Capabilities: {self.capabilities}")
                return True
            else:
                print(f"[ERROR] Device registration failed. Response: {response_data} (Original: {response_obj})")
                return False
        except Exception as e:
            print(f"[ERROR] Error during session initialization/device registration: {e}")
            traceback.print_exc()
            return False

    async def play_audio(self, audio_bytes: bytes, source_engine: str):
        if not audio_bytes:
            print("[PiMCPClient.play_audio] No audio bytes to play.")
            return
        # --- FUTURE DEVELOPMENT CONSIDERATION for play_audio ---
        # If multiple asynchronous tasks might try to call play_audio concurrently (e.g., background notifications
        # while user is interacting), a queueing mechanism or a lock within AudioManager might be needed
        # to prevent audio streams from overlapping or interrupting each other unexpectedly.
        # For the current single-threaded interaction model, this is generally not an issue.
        # AudioManager's internal queue helps, but concurrent calls *to* play_audio need care.
        # --- END CONSIDERATION ---
        temp_dir = Path(tempfile.gettempdir())
        fname_base = f"assistant_response_{int(time.time())}" # Unique name
        ext = ".wav" if source_engine.lower() in ["piper", "cartesia"] else ".mp3"
        fname = temp_dir / (fname_base + ext)

        try:
            with open(fname, "wb") as f: f.write(audio_bytes)
            await self.audio_manager.queue_audio(audio_file=str(fname))
            await self.audio_manager.wait_for_audio_completion()
        except Exception as e:
            print(f"[ERROR] PiMCPClient.play_audio: Failed to play audio from {fname}: {e}")
            traceback.print_exc()
        finally:
            if os.path.exists(fname) and not self.debug_keep_audio_files:
                try: os.remove(fname)
                except Exception as e_del: print(f"[WARN] Failed to delete temp audio file {fname}: {e_del}")

    async def _check_manual_vad_stop(self): # From v2, adapted
        if not self.keyboard_device or not ecodes: return False
        try:
            r, _, _ = select.select([self.keyboard_device.fd], [], [], 0)
            if r:
                for event in self.keyboard_device.read(): # Read all available events
                    if event.type == ecodes.EV_KEY and event.code == ecodes.KEY_LEFTMETA and event.value == 1: # Pressed
                        print("[VAD] Manual stop via keyboard (Meta key press).")
                        return True
        except BlockingIOError: pass
        except Exception as e: print(f"[VAD] Keyboard check error: {e}")
        return False

    async def capture_speech_with_vad(self) -> str | None:
        await self.display_manager.update_display("listening", mood="curious" if not self.is_follow_up_interaction else "attentive")
        await self.audio_manager.initialize_input() # Ensure input is ready
        audio_stream = await self.audio_manager.start_listening()
        if not audio_stream:
            print("[ERROR] Failed to start audio stream for VAD.")
            return None

        vad_config = VAD_SETTINGS
        initial_timeout = vad_config['follow_up_initial_listen_timeout_s'] if self.is_follow_up_interaction else vad_config['initial_listen_timeout_s']
        max_recording_time = vad_config['follow_up_max_recording_time_s'] if self.is_follow_up_interaction else vad_config['max_recording_time']
        energy_thresh = vad_config['energy_threshold']
        continued_ratio = vad_config['continued_threshold_ratio']
        silence_dur_s = vad_config['silence_duration']
        min_speech_s = vad_config['min_speech_duration']
        frame_history_len = vad_config['frame_history_length']
        speech_buffer_time_s = vad_config.get("speech_buffer_time_s", 0.2) # From v2

        print(f"[VAD] Listening {'(follow-up)' if self.is_follow_up_interaction else ''}. Timeout: {initial_timeout:.1f}s. Max Rec: {max_recording_time:.1f}s")
        print(f"[VAD] Params: Energy={energy_thresh:.4f}, SilenceDur={silence_dur_s:.1f}s, MinSpeech={min_speech_s:.1f}s")

        self.transcriber.reset()
        overall_listen_start_time = time.monotonic()
        speech_start_time = 0
        voice_started = False
        silence_frames_count = 0
        frames_per_second = self.audio_manager.sample_rate / self.audio_manager.frame_length
        silence_frames_needed = int(silence_dur_s * frames_per_second)
        frame_history = []
        
        try:
            while True:
                current_time = time.monotonic()
                if await self._check_manual_vad_stop():
                    if voice_started and (current_time - speech_start_time) > min_speech_s:
                         await asyncio.sleep(speech_buffer_time_s) # Record a small buffer
                    else: # No significant speech or VAD not started, so just stop
                         self.transcriber.reset() # Clear any partials
                    break 

                if not voice_started and (current_time - overall_listen_start_time > initial_timeout):
                    print(f"[VAD] Initial timeout ({initial_timeout:.1f}s). No voice.")
                    return None
                if voice_started and (current_time - speech_start_time > max_recording_time):
                    print(f"[VAD] Max recording time ({max_recording_time:.1f}s) reached.")
                    break

                pcm_bytes = self.audio_manager.read_audio_frame()
                if not pcm_bytes:
                    await asyncio.sleep(0.01)
                    continue

                frame_data_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
                frame_data_float32 = frame_data_int16.astype(np.float32) / 32768.0
                current_energy = np.sqrt(np.mean(frame_data_float32**2)) if len(frame_data_float32) > 0 else 0.0
                
                frame_history.append(current_energy)
                if len(frame_history) > frame_history_len: frame_history.pop(0)
                avg_energy = sum(frame_history) / len(frame_history) if frame_history else 0.0

                is_final_vosk, _, partial_text_vosk = self.transcriber.process_frame(pcm_bytes)

                if not voice_started:
                    if avg_energy > energy_thresh:
                        voice_started = True
                        speech_start_time = current_time
                        silence_frames_count = 0
                        print(f"[VAD] Voice started. Energy: {avg_energy:.4f}")
                else: # Voice has started
                    if avg_energy > (energy_thresh * continued_ratio):
                        silence_frames_count = 0
                    else:
                        silence_frames_count += 1

                    speech_duration_s = current_time - speech_start_time
                    if silence_frames_count >= silence_frames_needed and speech_duration_s >= min_speech_s:
                        print(f"[VAD] End of speech by silence. Duration: {speech_duration_s:.2f}s")
                        await asyncio.sleep(speech_buffer_time_s) # Record buffer
                        break
                    if is_final_vosk and speech_duration_s >= min_speech_s:
                        print(f"[VAD] End of speech by Vosk final. Duration: {speech_duration_s:.2f}s")
                        break
                
                if partial_text_vosk:
                    if not hasattr(self, "last_partial_print_time") or (current_time - getattr(self, "last_partial_print_time", 0) > 1.0):
                        print(f"[VAD] Partial: {partial_text_vosk}") # Re-enabled console print for partials
                        self.display_manager.update_display("listening", mood="curious", text=partial_text_vosk) # Show partials on display
                        self.last_partial_print_time = current_time
            
            final_transcript = self.transcriber.get_final_text()
            print(f"[VAD] Raw final transcript from Vosk: '{final_transcript}'") # Print raw before stripping/rejection

            if final_transcript:
                final_transcript = final_transcript.strip()
                if not final_transcript:
                    print("[VAD] Transcript empty after stripping.")
                    return None

                num_words = len(final_transcript.split())
                min_chars_single = vad_config.get('vosk_reject_min_chars_if_single_word', 3)
                min_words_overall = vad_config.get('vosk_reject_min_words', 1)

                if num_words == 0: return None # Should be caught by empty strip check
                if num_words < min_words_overall:
                    print(f"[VAD] Rejecting (too few words: {num_words} < {min_words_overall}): '{final_transcript}'")
                    return None
                if num_words == 1 and len(final_transcript) < min_chars_single:
                    print(f"[VAD] Rejecting (single short word: len {len(final_transcript)} < {min_chars_single}): '{final_transcript}'")
                    return None
                
                print(f"[VAD] Accepted final transcript: '{final_transcript}'")
                # Consider briefly showing final_transcript on DisplayManager here if desired
                # await self.display_manager.update_display("recognized_speech", mood="neutral", text=final_transcript)
                # await asyncio.sleep(1.0) 
                return final_transcript
            
            print("[VAD] No final transcript obtained.")
            return None
        except Exception as e:
            print(f"[ERROR] Error during VAD/transcription: {e}")
            traceback.print_exc()
            return None
        finally:
            await self.audio_manager.stop_listening()
            if hasattr(self, "last_partial_print_time"): del self.last_partial_print_time
            # Display state will be handled by the caller (e.g., to "thinking" or "idle")
            # await self.display_manager.update_display("idle", mood="neutral")

    async def send_to_laura_tool(self, transcript: str) -> dict | None:
        if not self.session_id or not self.mcp_session:
            print("[ERROR] Session not initialized. Cannot send message to LAURA tool.")
            return {"text": "Error: Client session not ready.", "mood": "error"}
        try:
            tool_call_args = {
                "session_id": self.session_id,
                "input_type": "text",
                "payload": {"text": transcript, "active_persona": self.active_persona},
                "output_mode": ["text", "audio"],
                "broadcast": False,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            print(f"[INFO] Calling 'run_LAURA' tool with: {tool_call_args}")
            response_payload = await self.mcp_session.call_tool("run_LAURA", arguments=tool_call_args)

            response_payload = None
            if hasattr(response_obj, 'content') and isinstance(response_obj.content, list) and response_obj.content:
                if hasattr(response_obj.content[0], 'text') and isinstance(response_obj.content[0].text, str):
                    try:
                        response_payload = json.loads(response_obj.content[0].text)
                    except json.JSONDecodeError:
                        print(f"[ERROR] Failed to parse JSON from run_LAURA response: {response_obj.content[0].text}")
                elif isinstance(response_obj.content[0], dict):
                    response_payload = response_obj.content[0]
            elif isinstance(response_obj, dict):
                response_payload = response_obj

            print(f"[INFO] 'run_LAURA' tool response payload: {response_payload}")

            if response_payload:
                if "text" in response_payload:
                    return response_payload
                elif "error" in response_payload:
                    error_msg = response_payload.get('error', 'Unknown error')
                    return {"text": f"Sorry, an error occurred: {error_msg}", "mood": "error", **response_payload}
                else:
                    return {"text": "Sorry, I received an unexpected response.", "mood": "confused"}
            return {"text": "Sorry, no response received from LAURA tool.", "mood": "error"}
        except Exception as e:
            print(f"[ERROR] Failed to call 'run_LAURA' or process response: {e}")
            traceback.print_exc()
            return {"text": "Sorry, a communication problem occurred.", "mood": "error"}

    def _clean_text_for_tts(self, text_from_server: str, mood_from_server: str | None) -> str:
        if not text_from_server:
            return ""
        cleaned_text = text_from_server
        mood_match = re.match(r'^\[(.*?)\]([\s\S]*)', cleaned_text, re.IGNORECASE | re.DOTALL)
        if mood_match:
            cleaned_text = mood_match.group(2).strip()
        elif mood_from_server:
            if cleaned_text.lower().startswith(f"[{mood_from_server.lower()}]"):
                 cleaned_text = cleaned_text[len(mood_from_server)+2:].lstrip()
            elif cleaned_text.startswith(f"[{mood_from_server.capitalize()}]"):
                 cleaned_text = cleaned_text[len(mood_from_server)+2:].lstrip()

        formatted_message = cleaned_text.replace('\n\n', '. ')
        formatted_message = formatted_message.replace('\n', ' ')
        formatted_message = re.sub(r'\s+', ' ', formatted_message)
        formatted_message = re.sub(r'\(\s*\)', '', formatted_message)
        return formatted_message.strip()

    def has_conversation_hook(self, response_text: str | None, response_data: dict | None = None) -> bool:
        if not response_text: return False
        if response_data and isinstance(response_data.get("control_signals"), list):
            if "CONTINUE_CONVERSATION" in [str(s).upper() for s in response_data["control_signals"]]:
                print("[DEBUG] Hook detected: Server explicit CONTINUE_CONVERSATION signal")
                return True
        if "[continue]" in response_text.lower():
            print("[DEBUG] Hook detected: [continue] tag in text")
            return True
        if "?" in response_text:
            print("[DEBUG] Hook detected: Question mark")
            return True
        response_lower = response_text.lower()
        for phrase in self.continuation_phrases:
            if phrase in response_lower:
                # Special care for "." to avoid triggering on every single period if that's too much.
                # Current logic: if "." is in phrases and present in response_lower, it's a hook.
                print(f"[DEBUG] Hook detected: Phrase '{phrase}')") # Changed to use f-string
                return True
        return False

    async def upload_document(self, file_path: str) -> dict | None:
        if not self.mcp_session or not self.session_id:
            print("[ERROR] Cannot upload document: MCP session not active.")
            return None
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            import base64
            file_b64 = base64.b64encode(file_data).decode('utf-8')

            print(f"[INFO] Uploading document: {Path(file_path).name}")
            upload_result_obj = await self.mcp_session.call_tool(
                "upload_document",
                arguments={
                    "session_id": self.session_id,
                    "filename": Path(file_path).name,
                    "content_b64": file_b64,
                    "content_type": "application/octet-stream"
                }
            )
            # Parse upload_result_obj, assuming it might be a wrapper
            upload_status_data = None
            if hasattr(upload_result_obj, 'content') and isinstance(upload_result_obj.content, list) and upload_result_obj.content:
                if hasattr(upload_result_obj.content[0], 'text') and isinstance(upload_result_obj.content[0].text, str):
                    try: upload_status_data = json.loads(upload_result_obj.content[0].text)
                    except: pass # Ignore parse error, treat as non-dict
                elif isinstance(upload_result_obj.content[0], dict):
                    upload_status_data = upload_result_obj.content[0]
            elif isinstance(upload_result_obj, dict):
                 upload_status_data = upload_result_obj

            if upload_status_data and upload_status_data.get("status") == "success":
                print(f"[INFO] Document upload successful via tool: {Path(file_path).name}")
                return {"status": "success", "filename": Path(file_path).name, "details": upload_status_data}
            else:
                print(f"[WARN] Document upload via tool may have failed or returned unexpected status: {upload_status_data} (Original: {upload_result_obj})")
                return {"status": "failed_or_unknown", "filename": Path(file_path).name, "details": upload_status_data}

        except Exception as e:
            print(f"[ERROR] Failed to upload {file_path}: {e}")
            traceback.print_exc()
            return {"status": "error", "message": str(e), "filename": Path(file_path).name}

    def _listen_keyboard_sync(self) -> bool: # Synchronous check for main loop
        if not self.keyboard_device or not ecodes or not select: return False
        try:
            r, _, _ = select.select([self.keyboard_device.fd], [], [], 0.01)
            if r:
                for event in self.keyboard_device.read():
                    if event.type == ecodes.EV_KEY and event.code == ecodes.KEY_LEFTMETA and event.value == 1:
                        return True
        except BlockingIOError: pass
        except Exception as e: print(f"[ERROR] Keyboard listen error: {e}")
        return False

    async def run(self):
        print("[INFO] PiMCPClient run loop started.")
        
        # DisplayManager initialization and thread start happens ONCE at the beginning
        # It's an independent client-side visual component.
        await self.display_manager.update_display("booting", mood="casual") # Initial booting display
        self.display_manager.start_display_thread() # Start the display's own Pygame loop in a thread

        # --- This is the main connection loop for the MCP SERVER ---
        # This loop will attempt to establish and maintain a connection to the server.
        while True: 
            try:
                print(f"[INFO] Attempting to connect to MCP server at {self.server_url}...")
                
                # 'async with sse_client' establishes a single connection.
                # If this connection drops or fails, the 'async with' block exits,
                # and the outer 'while True' loop will try to reconnect.
                async with sse_client(self.server_url, headers={}) as (read, write):
                    print("[INFO] SSE client connected. Creating ClientSession...")
                    
                    # This ClientSession manages the interaction over the established SSE connection.
                    async with ClientSession(read, write) as session:
                        self.mcp_session = session # Store the active MCP session object
                        print("[INFO] ClientSession active.")

                        # Initialize client session with the server (e.g., register device)
                        if not await self.initialize_session():
                            print("[ERROR] Failed to initialize session with server. Reconnection will be attempted.")
                            # Break out of the inner session and SSE connection to trigger outer 'while True' retry
                            break # Exits 'async with ClientSession' and then 'async with sse_client'

                        # After session is fully established, change display from 'booting' to 'idle'
                        await self.display_manager.update_display("idle", mood="neutral")

                        # --- Main Interaction Loop (Operates within a single, active SSE connection) ---
                        # This loop handles wake words, user speech, server queries, TTS responses, etc.
                        while True:
                            self.is_follow_up_interaction = False
                            
                            # ... (Your existing logic for wake word detection) ...
                            wake_event_source = None
                            if self._listen_keyboard_sync():
                                wake_event_source = "keyboard"
                                print("[INFO] Keyboard wake detected.")
                            else:
                                wakeword_model = self.wakeword_detector.detect()
                                if wakeword_model:
                                    wake_event_source = f"wakeword ({wakeword_model})"
                                    print(f"[INFO] Wakeword '{wakeword_model}' detected.")

                            if not wake_event_source:
                                await asyncio.sleep(0.05)
                                continue

                            # ... (The rest of your existing interaction logic,
                            #     including capture_speech_with_vad, query_files,
                            #     system commands, send_to_laura_tool, TTS,
                            #     and conversation hooks) ...
                            current_transcript = await self.capture_speech_with_vad()
                            if not current_transcript: 
                                await self.display_manager.update_display("idle", mood="neutral")
                                continue

                            while current_transcript: 
                                print(f"[INFO] Processing transcript: '{current_transcript}' (Follow-up: {self.is_follow_up_interaction})")
                                await self.display_manager.update_display("thinking", mood="thoughtful")

                                query_files_path = Path(client_settings.get("QUERY_FILES_DIR"))
                                if query_files_path.exists():
                                    for file_to_upload in query_files_path.iterdir():
                                        if file_to_upload.is_file():
                                            upload_status_data = await self.upload_document(str(file_to_upload))
                                            
                                            offload_path = Path(client_settings.get("QUERY_OFFLOAD_DIR"))
                                            offload_path.mkdir(parents=True, exist_ok=True)
                                            try:
                                                file_to_upload.rename(offload_path / file_to_upload.name)
                                                if upload_status_data and upload_status_data.get("status") == "success":
                                                    print(f"[INFO] Document {file_to_upload.name} successfully uploaded and moved to offload.")
                                                else:
                                                    details = upload_status_data.get('details') if upload_status_data else 'N/A'
                                                    print(f"[WARN] Document {file_to_upload.name} moved to offload. Upload status: {upload_status_data.get('status', 'unknown') if upload_status_data else 'None'}. Details: {details}")
                                            except Exception as move_err:
                                                details = upload_status_data.get('details') if upload_status_data else 'N/A'
                                                print(f"[ERROR] Could not move {file_to_upload.name} to offload (Upload status: {upload_status_data.get('status', 'unknown') if upload_status_data else 'None'}, Details: {details}): {move_err}")
                                            await asyncio.sleep(0.1)

                                is_cmd, cmd_type, cmd_arg = detect_system_command(current_transcript)
                                if is_cmd:
                                    print(f"[INFO] System command: {cmd_type}('{cmd_arg}')")
                                    handled_cmd = False
                                    if cmd_type == "switch_tts_mode" and cmd_arg in ["local", "api", "text"]:
                                        client_settings["tts_mode"] = cmd_arg
                                        handled_cmd = True
                                    elif cmd_type == "switch_api_tts_provider" and cmd_arg in ["cartesia", "elevenlabs"]:
                                        set_active_tts_provider(cmd_arg)
                                        handled_cmd = True
                                    
                                    if handled_cmd:
                                        save_client_settings()
                                        conf_audio, conf_engine = await self.tts_handler.generate_audio(f"{cmd_type.replace('_', ' ')} set to {cmd_arg}.", persona_name="client_default")
                                        if conf_audio and conf_engine: await self.play_audio(conf_audio, conf_engine)
                                    current_transcript = None
                                    self.is_follow_up_interaction = False
                                    break 

                                laura_response_data = await self.send_to_laura_tool(current_transcript)

                                if not laura_response_data or "text" not in laura_response_data:
                                    print(f"[ERROR] Invalid/No response from LAURA. Data: {laura_response_data}")
                                    await self.display_manager.update_display("error", mood="confused", text="Server Error")
                                    await asyncio.sleep(2)
                                    current_transcript = None
                                    break

                                response_text_server = laura_response_data.get("text")
                                mood_display = laura_response_data.get("mood", "casual")
                                self.active_persona = laura_response_data.get("active_persona", self.active_persona) 

                                cleaned_tts_text = self._clean_text_for_tts(response_text_server, mood_display)
                                print(f"[DEBUG TTS Prep] Original: '{response_text_server[:100]}...' Cleaned: '{cleaned_tts_text[:100]}...' Persona: {self.active_persona}")
                                
                                await self.display_manager.update_display("speaking", mood=mood_display, text=response_text_server)
                                
                                if client_settings.get("tts_mode", "api") != "text":
                                    audio_bytes, engine = await self.tts_handler.generate_audio(cleaned_tts_text, persona_name=self.active_persona)
                                    if audio_bytes and engine:
                                        await self.play_audio(audio_bytes, engine)
                                    else:
                                        print(f"[ERROR] TTS failed. No audio generated.")
                                else:
                                    print(f"Assistant [{self.active_persona}]: {response_text_server}")
                                    await asyncio.sleep(0.1 * len(cleaned_tts_text.split()) if cleaned_tts_text else 1)

                                if self.has_conversation_hook(response_text_server, laura_response_data):
                                    print("[INFO] Hook detected. Listening for follow-up...")
                                    self.is_follow_up_interaction = True
                                    follow_up_transcript = await self.capture_speech_with_vad()
                                    if follow_up_transcript:
                                        current_transcript = follow_up_transcript
                                    else:
                                        print("[INFO] No follow-up speech. Ending conversation.")
                                        current_transcript = None
                                else:
                                    print("[INFO] No hook. Ending conversation.")
                                    current_transcript = None
                            
                            self.is_follow_up_interaction = False
                            await self.display_manager.update_display("idle", mood="neutral")
                            print("[INFO] Conversation turn ended. Waiting for new wake event.")

                        # --- If the inner 'while True' breaks (e.g., due to a client-side error not caught by a specific except),
                        #     it will exit this 'async with ClientSession' and then 'async with sse_client'
                        #     and the outer 'while True' will attempt to reconnect.

            # --- Connection-level Error Handling for the SSE connection ---
            except asyncio.CancelledError:
                print("[INFO] Main interaction loop cancelled.")
                break # Exit the outermost 'while True' loop as well, for graceful shutdown
            except ConnectionRefusedError:
                print("[ERROR] SSE Connection refused by server. Retrying in 10s...")
                await self.display_manager.update_display("error", mood="disconnected", text="Server offline?")
                await asyncio.sleep(10) # Wait before next reconnection attempt
            except Exception as e_connection_level: # Catch any other connection-level errors (e.g., unexpected disconnect)
                print(f"[ERROR] Unhandled connection-level exception: {e_connection_level}")
                traceback.print_exc()
                await self.display_manager.update_display("error", mood="error", text="Connection Error")
                await asyncio.sleep(5) # Wait before next reconnection attempt
            finally:
                # This ensures cleanup of the MCP session if the connection or inner session loop exits.
                print("[INFO] Exiting active SSE/ClientSession block. MCP session will be cleared.")
                self.mcp_session = None # Clear the active session object
                await self.display_manager.update_display("disconnected", mood="neutral") # Indicate disconnection status
                # The outer 'while True' loop will then automatically try to reconnect.
        print("[INFO] SSE client connection attempt loop has exited.") # This prints if the outermost 'while True' breaks.
        
    async def cleanup(self):
        print("[INFO] Starting PiMCPClient cleanup...")
        if self.wakeword_detector: self.wakeword_detector.cleanup()
        if self.audio_manager:
            if asyncio.iscoroutinefunction(self.audio_manager.cleanup): await self.audio_manager.cleanup()
            else: self.audio_manager.cleanup()
        if self.display_manager: self.display_manager.cleanup() # This will now call its cleanup
        if self.transcriber and hasattr(self.transcriber, 'cleanup'): self.transcriber.cleanup()
        if self.keyboard_device and hasattr(self.keyboard_device, 'close'):
            try: self.keyboard_device.close()
            except Exception as e: print(f"[ERROR] Closing keyboard device: {e}")
        print("[INFO] PiMCPClient cleanup finished.")

async def main():
    load_client_settings()

    vosk_model = VAD_SETTINGS.get("vosk_model_path", VOSK_MODEL_PATH)
    if not vosk_model or not Path(vosk_model).exists():
        print("*"*70 + f"\nERROR: Vosk model path not found: '{vosk_model}'\n" + "*"*70)
        return

    if not snowboydetect or not pyaudio:
         print("*"*70 + "\nWARN: Snowboy or PyAudio not available. Wakeword detection disabled.\n" + "*"*70)
    elif not Path(WAKEWORD_RESOURCE_FILE).exists() or not WAKE_WORDS_AND_SENSITIVITIES:
         print("*"*70 + f"\nERROR: Snowboy resource '{WAKEWORD_RESOURCE_FILE}' or WAKE_WORDS not configured.\n" + "*"*70)

    client = PiMCPClient(server_url=SERVER_URL, device_id=DEVICE_ID)
    try:
        await client.run()
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received by main().")
    except Exception as e:
        print(f"[ERROR] Unhandled exception in main(): {e}")
        traceback.print_exc()
    finally:
        print("[INFO] Main function finished. Performing final cleanup...")
        await client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[INFO] Application terminated by user (KeyboardInterrupt in __main__).")
    except Exception as e_outer:
        print(f"[FATAL] Outer exception during asyncio.run: {e_outer}")
        traceback.print_exc()
    finally:
        print("[INFO] Application shutdown complete.")
