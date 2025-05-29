#!/usr/bin/env python3
import numpy as np
import time
import re
import sys
# Ensure LAURA/snowboy path is correct if not using a virtual environment with it installed
# sys.path.append('/home/user/LAURA/snowboy') # Keep if necessary for snowboydetect
import asyncio
import os
import json
# import time # already imported
import select
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple # Added Tuple
from evdev import ecodes
import tempfile

# Import the config module with runtime TTS provider/voice handling
# client_config now handles loading its own settings dictionary
from client_config import (
    SVG_PATH, BOOT_IMG_PATH, WINDOW_SIZE, # Static paths
    client_settings, save_client_config, # Live settings dictionary and saver
    get_active_tts_provider, set_active_tts_provider, # Helpers
    WAKEWORD_DIR, WAKEWORD_RESOURCE, WAKE_WORDS, AUDIO_SAMPLE_RATE, AUDIO_CHUNK # Constants for WakeWordDetector
)
from audio_manager_vosk import AudioManager
from vosk_transcriber import VoskTranscriber
from display_manager import DisplayManager
from client_system_manager import ClientSystemManager
from client_tts_handler import TTSHandler # Corrected TTSHandler usage
import pyaudio
from evdev import InputDevice, list_devices, ecodes

from mcp import ClientSession
from mcp.client.sse import sse_client

# Utility: Find the Pi 500 keyboard
def find_pi_keyboard():
    for path in list_devices():
        dev = InputDevice(path)
        if "Pi 500" in dev.name and "Keyboard" in dev.name:
            print(f"[INFO] Found Pi 500 Keyboard: {dev.name} at {dev.path}")
            return dev
    print("[WARN] Pi 500 Keyboard not found.")
    return None



# Wakeword detector class
class WakeWordDetector:
    def __init__(self):
        # Uses constants directly from client_config import
        self.model_paths = [str(WAKEWORD_DIR / name) for name in WAKE_WORDS]
        self.sensitivities = ",".join(str(WAKE_WORDS[name]) for name in WAKE_WORDS)
        self.model_names = list(WAKE_WORDS.keys())
        
        try:
            import snowboydetect
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(
                rate=AUDIO_SAMPLE_RATE,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=AUDIO_CHUNK,
                # Consider adding input_device_index if multiple mics
            )
            self.detector = snowboydetect.SnowboyDetect(
                resource_filename=str(WAKEWORD_RESOURCE).encode(),
                model_str=",".join(self.model_paths).encode()
            )
            self.detector.SetSensitivity(self.sensitivities.encode())
            print("[INFO] WakeWordDetector initialized successfully.")
        except ImportError:
            print("[ERROR] snowboydetect module not found. Wakeword detection will be disabled.")
            self.detector = None
            self.stream = None
            self.pa = None
        except Exception as e:
            print(f"[ERROR] Failed to initialize WakeWordDetector: {e}")
            traceback.print_exc()
            self.detector = None
            self.stream = None
            self.pa = None


    def detect(self):
        if not self.detector or not self.stream:
            # print("[DEBUG] Wakeword detector not available.") # Too noisy for regular check
            return None
        try:
            data = self.stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            result = self.detector.RunDetection(data)
            if result > 0:
                return self.model_names[result - 1]
            return None
        except IOError as e: # Handle stream read errors, e.g., if mic is disconnected
            if e.errno == pyaudio.paInputOverflowed:
                # print("[WARN] WakeWordDetector: Input overflowed. Skipping frame.")
                return None # Or handle as needed
            else:
                print(f"[ERROR] WakeWordDetector IOError: {e}")
                # Consider re-initializing or stopping stream
                return None
        except Exception as e:
            print(f"[ERROR] WakeWordDetector exception: {e}")
            traceback.print_exc()
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
        print("[INFO] WakeWordDetector cleaned up.")

# System command detection
def detect_system_command(transcript):
    t = transcript.lower()
    # TTS Mode Switching
    if "enable remote tts" in t or "api tts" in t:
        return True, "switch_tts_mode", "api"
    elif "enable local tts" in t or "local tts" in t:
        return True, "switch_tts_mode", "local"
    elif "text only mode" in t or "text only" in t:
        return True, "switch_tts_mode", "text"
    # API TTS Provider Switching
    elif "switch tts provider to cartesia" in t:
        return True, "switch_api_tts_provider", "cartesia"
    elif "switch tts provider to elevenlabs" in t:
        return True, "switch_api_tts_provider", "elevenlabs"
    # STT Mode Switching
    elif "enable remote transcription" in t or "remote transcription" in t:
        return True, "switch_stt_mode", "remote"
    elif "enable local transcription" in t or "local transcription" in t:
        return True, "switch_stt_mode", "local"
    return False, None, None

class PiMCPClient:
    def __init__(self):
        self.audio_manager = AudioManager()
        self.document_queue = []
        self.display_manager = DisplayManager(
            svg_path=SVG_PATH,
            boot_img_path=BOOT_IMG_PATH,
            window_size=WINDOW_SIZE
        )

        vosk_model_p = client_settings.get("vosk_model_path")
        # VoskTranscriber no longer takes audio_manager in __init__
        self.stt = VoskTranscriber(model_path=vosk_model_p, sample_rate=client_settings.get("audio_sample_rate", 16000))

        self.wakeword = WakeWordDetector()
        self.keyboard = find_pi_keyboard()
        self.tts_handler = TTSHandler() 

        self.system_manager = ClientSystemManager() # Ensure this is correctly defined/imported
        self.server_url = client_settings.get("mcp_server_uri")
        self.device_id = client_settings.get("device_id")
        self.session_id = None
        # self.stt_mode is now read from client_settings directly where needed
        self.is_follow_up_interaction = False
        
        self.vad_params = client_settings.get("vad_settings", {}).copy()
        _fallback_vad = client_settings.get("_default_config", {}).get("vad_settings", {})
        for key, default_val in _fallback_vad.items():
            self.vad_params.setdefault(key, default_val)
        print(f"[PiMCPClient DEBUG] Effective VAD params: {self.vad_params}")

        # Define conversation hook phrases
        self.continuation_phrases = [
            ".", # Ensures most complete sentences from assistant trigger re-engagement
            "let me know", "tell me more", "what else", "anything else",
            "can i help you with anything else", "do you need more information",
            "share your thoughts", "i'm listening", "go ahead",
            "feel free to elaborate", "i'd like to hear more", "please continue",
            "what do you think", "how does that sound"
        ]

    async def upload_document(self, file_path: str, session):  # Add session parameter
        """Upload a document to the server via MCP"""
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Convert to base64 for JSON transport
            import base64
            file_b64 = base64.b64encode(file_data).decode('utf-8')
            
            upload_result = await session.call_tool(  # Now session is available
                "upload_document",
                arguments={
                    "session_id": self.session_id,
                    "filename": Path(file_path).name,
                    "content": file_b64,
                    "content_type": "application/octet-stream"
                }
            )
            print(f"[INFO] Document uploaded: {file_path}")
            return upload_result
        except Exception as e:
            print(f"[ERROR] Failed to upload {file_path}: {e}")
            return None

    async def _check_manual_stop(self):
        if not self.keyboard:
            return False
        try:
            # Using select to make read_one non-blocking if no events
            r, _, _ = select.select([self.keyboard.fd], [], [], 0)
            if r:
                for event in self.keyboard.read(): # Read all available events
                    if event.type == ecodes.EV_KEY and event.code == ecodes.KEY_LEFTMETA and event.value == 1:
                        print("[CAPTURE SPEECH] Manual stop via keyboard (Meta key press).")
                        return True
        except BlockingIOError:
            pass # No events available
        except Exception as e:
            print(f"[CAPTURE SPEECH] Keyboard check error: {e}")
        return False

    def has_conversation_hook(self, response_text: str) -> bool:
        """
        Detects if the assistant's response suggests the conversation should continue.
        """
        if not response_text or not isinstance(response_text, str):
            print("[DEBUG] No response text or invalid type for hook check.")
            return False

        normalized_response = response_text.lower().strip()

        if "[continue]" in normalized_response: # Explicit server directive
            print("[DEBUG] Hook detected: [continue] tag")
            return True
        
        if "?" in normalized_response:
            print("[DEBUG] Hook detected: Question mark")
            return True
        
        for phrase in self.continuation_phrases:
            if phrase in normalized_response:
                print(f"[DEBUG] Hook detected: Phrase '{phrase}'")
                return True
            
        print("[DEBUG] No conversation hook detected in response.")
        return False

    async def capture_speech_with_vad(self):
        await self.display_manager.update_display("listening")
        self.stt.reset()

        initial_timeout_s_key = "follow_up_initial_listen_timeout_s" if self.is_follow_up_interaction else "initial_listen_timeout_s"
        initial_timeout_s = float(self.vad_params.get(initial_timeout_s_key, 7.0))

        max_rec_time_s_key = "follow_up_max_recording_time_s" if self.is_follow_up_interaction else "max_recording_time"
        max_recording_time_s = float(self.vad_params.get(max_rec_time_s_key, 30.0))

        energy_threshold = float(self.vad_params.get("energy_threshold", 0.01))
        continued_ratio = float(self.vad_params.get("continued_threshold_ratio", 0.65))
        silence_duration_s = float(self.vad_params.get("silence_duration", 2.0))
        min_speech_duration_s = float(self.vad_params.get("min_speech_duration", 0.3))
        speech_buffer_time_s = float(self.vad_params.get("speech_buffer_time", 0.2))
        frame_history_length = int(self.vad_params.get("frame_history_length", 10))

        voice_detected_in_vad = False
        vad_is_speaking_state = False
        speech_start_time_mono = None
        silence_frames_count = 0
        # Ensure audio_manager has sample_rate and frame_length attributes
        max_silence_frames = int(silence_duration_s * self.audio_manager.sample_rate / self.audio_manager.frame_length) if self.audio_manager.sample_rate and self.audio_manager.frame_length else 0
        if max_silence_frames == 0:
            print("[WARN] VAD max_silence_frames is 0, VAD by silence might not work correctly. Check AudioManager sample_rate/frame_length.")
            max_silence_frames = int(silence_duration_s * 16000 / 512) # Fallback if not set


        frame_history_for_vad = []
        
        print(f"[CAPTURE SPEECH] Listening {'(follow-up)' if self.is_follow_up_interaction else ''}. Initial Timeout: {initial_timeout_s:.1f}s. Max Rec: {max_recording_time_s:.1f}s")
        print(f"[CAPTURE SPEECH] VAD Params: EnergyThresh={energy_threshold:.4f}, SilenceDur={silence_duration_s:.1f}s, MinSpeech={min_speech_duration_s:.1f}s")

        overall_listen_start_time_mono = time.monotonic()
        transcript = None
        
        try:
            await self.audio_manager.start_listening()
            if not self.audio_manager.audio_stream:
                print("[CAPTURE SPEECH ERROR] Audio stream not available.")
                await self.display_manager.update_display('idle' if self.is_follow_up_interaction else 'sleep')
                return None

            while True:
                current_time_mono = time.monotonic()

                if not voice_detected_in_vad and (current_time_mono - overall_listen_start_time_mono) > initial_timeout_s:
                    print(f"[CAPTURE SPEECH] VAD initial timeout ({initial_timeout_s:.1f}s). No voice detected.")
                    break

                pcm_bytes = self.audio_manager.read_audio_frame()
                if not pcm_bytes:
                    await asyncio.sleep(0.01)
                    continue
                
                if voice_detected_in_vad: # Only check for manual stop if voice has been detected
                    if await self._check_manual_stop():
                        # Check if enough speech was captured before manual stop
                        if speech_start_time_mono and (current_time_mono - speech_start_time_mono) > min_speech_duration_s:
                            print("[CAPTURE SPEECH] Manual stop triggered after sufficient speech.")
                            await asyncio.sleep(speech_buffer_time_s) # Record a small buffer
                            break 
                        else:
                            print("[CAPTURE SPEECH] Manual stop but recording too short or no speech detected yet, continuing briefly.")
                            # Allow a very short buffer then break, or just ignore and break if no speech_start_time_mono
                            if speech_start_time_mono: await asyncio.sleep(speech_buffer_time_s)
                            break


                self.stt.process_frame(pcm_bytes)

                float_data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                energy = np.sqrt(np.mean(float_data**2)) if len(float_data) > 0 else 0.0
                
                frame_history_for_vad.append(energy)
                if len(frame_history_for_vad) > frame_history_length:
                    frame_history_for_vad.pop(0)
                avg_energy = sum(frame_history_for_vad) / len(frame_history_for_vad) if frame_history_for_vad else 0.0

                if not vad_is_speaking_state:
                    if avg_energy > energy_threshold:
                        print(f"[CAPTURE SPEECH] VAD: Voice started. Energy: {avg_energy:.4f} (Thresh: {energy_threshold:.4f})")
                        voice_detected_in_vad = True
                        vad_is_speaking_state = True
                        speech_start_time_mono = current_time_mono
                        silence_frames_count = 0
                else: # vad_is_speaking_state is True
                    current_speech_duration_s = current_time_mono - (speech_start_time_mono if speech_start_time_mono else current_time_mono)

                    if avg_energy > (energy_threshold * continued_ratio):
                        silence_frames_count = 0
                    else:
                        silence_frames_count += 1
                    
                    if max_silence_frames > 0 and silence_frames_count >= max_silence_frames and current_speech_duration_s > min_speech_duration_s :
                        print(f"[CAPTURE SPEECH] VAD: End of speech by silence. Duration: {current_speech_duration_s:.2f}s")
                        await asyncio.sleep(speech_buffer_time_s)
                        break

                    if current_speech_duration_s > max_recording_time_s:
                        print(f"[CAPTURE SPEECH] VAD: Max recording time ({max_recording_time_s:.1f}s) reached.")
                        break
                
                await asyncio.sleep(0.01)

            if not voice_detected_in_vad:
                await self.display_manager.update_display('sleep' if not self.is_follow_up_interaction else 'idle')
                return None

            transcript = self.stt.get_final_text()
            print(f"[CAPTURE SPEECH] Raw transcript from Vosk: '{transcript}'")

            if transcript:
                transcript = re.sub(r'^(that were|that was)\s+', '', transcript, flags=re.IGNORECASE).strip()
                words = transcript.split()
                num_words = len(words)
                
                min_words_to_accept = int(self.vad_params.get("vosk_reject_min_words", 2))
                min_chars_single_word = int(self.vad_params.get("vosk_reject_min_chars_if_single_word", 4))

                if num_words == 0:
                    print("[CAPTURE SPEECH] Transcript empty after cleaning.")
                    transcript = None
                elif num_words < min_words_to_accept:
                    print(f"[CAPTURE SPEECH] Discarding (words): '{transcript}' ({num_words} < {min_words_to_accept})")
                    transcript = None
                elif num_words == 1 and len(words[0]) < min_chars_single_word:
                    print(f"[CAPTURE SPEECH] Discarding (chars): '{transcript}' ({len(words[0])} < {min_chars_single_word})")
                    transcript = None
            else:
                print("[CAPTURE SPEECH] No transcript returned by Vosk.")
                transcript = None
            
            if transcript:
                print(f"[CAPTURE SPEECH] Final processed transcript: '{transcript}'")
                return transcript.strip()
            else:
                # This was missing, if transcript becomes None after checks, ensure display is updated.
                print("[CAPTURE SPEECH] Transcript became None after processing.")
                await self.display_manager.update_display('idle' if self.is_follow_up_interaction else 'sleep')
                return None

        except Exception as e:
            print(f"[ERROR] Speech capture failed: {e}")
            traceback.print_exc()
            await self.display_manager.update_display('idle' if self.is_follow_up_interaction else 'sleep') # Ensure display reset
            return None
        finally:
            await self.audio_manager.stop_listening()


    async def capture_speech(self): # Kept for compatibility if anything calls it directly
        return await self.capture_speech_with_vad()

    def listen_keyboard(self) -> bool: # Synchronous check for main loop
        if not self.keyboard:
            return False
        r, _, _ = select.select([self.keyboard.fd], [], [], 0) # Non-blocking check
        if not r:
            return False
        # Read up to 5 events to catch the key press if there's a burst
        for _ in range(5): 
            event = self.keyboard.read_one() # May still block if select was wrong, but unlikely
            if event and event.type == ecodes.EV_KEY and event.code == ecodes.KEY_LEFTMETA and event.value == 1: # KEY_LEFTMETA is 125
                return True
        return False

    def listen_wakeword(self) -> Optional[str]: # Synchronous check for main loop
        return self.wakeword.detect()

    async def play_audio(self, audio_bytes: bytes, source_engine: str):
        """Plays audio bytes, saving to a temp file with an extension based on the source engine."""
        if not audio_bytes:
            print("[PLAY AUDIO] No audio bytes to play.")
            return

        temp_dir = Path(tempfile.gettempdir())
        fname_base = "assistant_response_temp"
        
        # Determine file extension based on source engine
        if source_engine.lower() == "piper":
            ext = ".wav"
        elif source_engine.lower() == "elevenlabs":
            ext = ".mp3"
        elif source_engine.lower() == "cartesia": # Assuming Cartesia outputs WAV
            ext = ".wav"
        else:
            print(f"[PLAY AUDIO WARN] Unknown source engine '{source_engine}', defaulting to .mp3 for saving.")
            ext = ".mp3" # Fallback

        fname = temp_dir / (fname_base + ext)

        try:
            with open(fname, "wb") as f:
                f.write(audio_bytes)
            print(f"[PLAY AUDIO] Playing {fname} (from {source_engine})")
            await self.audio_manager.queue_audio(audio_file=str(fname))
            await self.audio_manager.wait_for_audio_completion()
        except Exception as e:
            print(f"[ERROR] Failed to play audio from {fname}: {e}")
            traceback.print_exc()
        # finally: # Temporarily disable the finally block for diagnostics
        #     if fname.exists():
        #         try:
        #             # os.remove(fname) # <-- COMMENT THIS OUT
        #             print(f"[DEBUG] File {fname} would be removed here, but deletion is temporarily disabled.")
        #         except Exception as e_rem:
        #             print(f"[ERROR] Failed to remove temp audio file {fname}: {e_rem}")
        print(f"[DEBUG] play_audio finished for {fname}. File deletion temporarily disabled.")

    async def run(self):
        try:
            print(f"[INFO] Attempting to connect to MCP server at {self.server_url}")
            async with sse_client(self.server_url, headers={}) as (read, write):
                print("[INFO] SSE client connected. Initializing ClientSession...")
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    print(f"[INFO] ClientSession initialized. Registering device: {self.device_id}")
                    
                    register_result = await session.call_tool(
                        "register_device",
                        arguments={
                            "device_id": self.device_id,
                            "capabilities": {
                                "input": ["text", "audio"],
                                "output": ["text", "audio"],
                                "tts_mode": client_settings.get("tts_mode", "api"),
                                "api_tts_provider": get_active_tts_provider(),
                            }
                        }
                    )
                    
                    session_info = None
                    # Robust parsing of registration result
                    if hasattr(register_result, "content") and \
                       isinstance(register_result.content, list) and \
                       len(register_result.content) > 0 and \
                       hasattr(register_result.content[0], "text") and \
                       isinstance(register_result.content[0].text, str):
                        try:
                            session_info_text = register_result.content[0].text.strip()
                            if session_info_text: # Ensure not empty string
                                session_info = json.loads(session_info_text)
                                print(f"[INFO] Device registration successful. Session Info: {session_info}")
                            else:
                                print("[FATAL ERROR] Registration response text is empty.")
                                return
                        except json.JSONDecodeError as e:
                            print(f"[FATAL ERROR] Could not parse session info JSON: '{register_result.content[0].text}'. Error: {e}")
                            return
                    else:
                        print(f"[FATAL ERROR] Registration failed or invalid response structure: {register_result}")
                        return

                    if not isinstance(session_info, dict) or "session_id" not in session_info:
                        print(f"[FATAL ERROR] Registration failed (no session_id). Parsed: {session_info}")
                        return
                    self.session_id = session_info["session_id"]
                    print(f"[INFO] Session ID obtained: {self.session_id}")

                    await self.display_manager.update_display("idle")
                    
                    # --- Outer Main Interaction Loop (handles new wake events) ---
                    while True:
                        wake_model = None
                        self.is_follow_up_interaction = False # Reset for a new wake-initiated cycle
                        
                        # Listen for wake event (keyboard or wakeword)
                        try:
                            if self.listen_keyboard():
                                wake_model = "keyboard"
                                print("[INFO] Keyboard wake detected.")
                            else:
                                wm = self.listen_wakeword()
                                if wm:
                                    wake_model = wm
                                    print(f"[INFO] Wakeword '{wm}' detected.")
                        except Exception as e:
                            print(f"[ERROR] Wakeword/keyboard detection failed: {e}")
                            traceback.print_exc()
                            await asyncio.sleep(0.1) # Brief pause before retrying
                            continue # Retry wake detection

                        if not wake_model:
                            await asyncio.sleep(0.05) # Polling interval for wake detection
                            continue
                        
                        # A wake event has occurred, start processing
                        print(f"[INFO] Wake event '{wake_model}' processed. is_follow_up_interaction set to: {self.is_follow_up_interaction}")
                        
                        # Initial speech capture after wake event
                        current_processing_transcript = await self.capture_speech_with_vad()

                        # --- Inner Conversational Loop (handles back-and-forth after initial wake) ---
                        while current_processing_transcript:
                            print(f"[INFO] Processing transcript: '{current_processing_transcript}' (Follow-up status: {self.is_follow_up_interaction})")

                            # Check for new files to upload before sending to LAURA
                            query_files_dir = Path("/home/user/LAURA/query_files")
                            if query_files_dir.exists():
                                for file_path in query_files_dir.iterdir():
                                    if file_path.is_file():
                                        print(f"[INFO] Found document to upload: {file_path.name}")
                                        await self.upload_document(str(file_path), session)
                                        offload_dir = Path("/home/user/LAURA/query_offload")
                                        offload_dir.mkdir(exist_ok=True)
                                        file_path.rename(offload_dir / file_path.name)
                                        print(f"[INFO] Document {file_path.name} moved to offload.")
                            
                            # Check for system command
                            is_cmd, cmd_type, arg = detect_system_command(current_processing_transcript)
                            if is_cmd:
                                print(f"[INFO] Detected system command: {cmd_type} with arg: {arg}")
                                if cmd_type == "switch_tts_mode":
                                    client_settings["tts_mode"] = arg
                                elif cmd_type == "switch_api_tts_provider":
                                    set_active_tts_provider(arg) # Uses the helper
                                elif cmd_type == "switch_stt_mode":
                                    client_settings["stt_mode"] = arg
                                save_client_config(client_settings) # Save any changes
                                print(f"[INFO] System setting {cmd_type} updated to {arg}.")
                                
                                await self.display_manager.update_display("idle")
                                current_processing_transcript = None # End inner conversation loop
                                self.is_follow_up_interaction = False # Ensure reset
                                continue # Break from inner loop, go to outer (wait for new wake)

                            # Send transcript to LAURA tool on the server
                            response_from_laura_tool = None 
                            try:
                                await self.display_manager.update_display("thinking")
                                print(f"[INFO] Sending to LAURA tool: '{current_processing_transcript}'")
                                
                                response_from_laura_tool = await session.call_tool(
                                    "run_LAURA",
                                    arguments={
                                        "session_id": self.session_id,
                                        "input_type": "text", 
                                        "payload": {"text": current_processing_transcript},
                                        "timestamp": datetime.utcnow().isoformat() + "Z",
                                        "output_mode": ["text"] # Expecting text output for hooks
                                    }
                                )
                            except Exception as e:
                                print(f"[ERROR] Failed to send input to LAURA tool: {e}")
                                traceback.print_exc()
                                await self.display_manager.update_display("idle")
                                current_processing_transcript = None # End inner conversation loop
                                self.is_follow_up_interaction = False 
                                continue # Break from inner loop

                            # Process the response from LAURA
                            laura_response_data = None
                            # Robust parsing
                            if hasattr(response_from_laura_tool, "content") and \
                               isinstance(response_from_laura_tool.content, list) and \
                               len(response_from_laura_tool.content) > 0 and \
                               hasattr(response_from_laura_tool.content[0], "text") and \
                               isinstance(response_from_laura_tool.content[0].text, str):
                                try:
                                    laura_response_text_from_tool = response_from_laura_tool.content[0].text.strip()
                                    if laura_response_text_from_tool:
                                        laura_response_data = json.loads(laura_response_text_from_tool)
                                        print(f"[INFO] LAURA response data (parsed JSON): {laura_response_data}")
                                    else:
                                        print("[ERROR] LAURA response text from tool is empty.")
                                except json.JSONDecodeError as e:
                                    print(f"[ERROR] Could not parse LAURA response JSON: '{response_from_laura_tool.content[0].text}'. Error: {e}")
                                except Exception as e: # Catch any other parsing error
                                    print(f"[ERROR] Unexpected error parsing LAURA response JSON: {e}, Text: '{response_from_laura_tool.content[0].text}'")
                                    traceback.print_exc()
                            else:
                                print("[ERROR] LAURA response content item has no 'text' field, is not a string, or content list is empty.")
                            
                            if not isinstance(laura_response_data, dict):
                                print(f"[ERROR] Parsed LAURA response is not a valid dictionary. Ending conversation. Value: {laura_response_data}")
                                await self.display_manager.update_display("idle")
                                current_processing_transcript = None # End inner conversation loop
                                self.is_follow_up_interaction = False
                                continue

                            active_persona = laura_response_data.get("active_persona", client_settings.get("device_id", "laura"))
                            response_text_from_server = laura_response_data.get("text") # This is the text for TTS and hook check
                            mood_for_display = laura_response_data.get("mood", "casual") # Mood from structured data

                            if not response_text_from_server:
                                print("[WARN] LAURA response dict has no 'text' field. Ending conversation.")
                                current_processing_transcript = None # End inner conversation loop
                                self.is_follow_up_interaction = False
                                continue
                            
                            # Update display and prepare for TTS
                            await self.display_manager.update_display("speaking", mood=mood_for_display)
                            
                            # Clean message for TTS (mood tag stripping, newlines)
                            message_for_tts = response_text_from_server
                            mood_match = re.match(r'^\[(.*?)\](.*)', message_for_tts, re.IGNORECASE | re.DOTALL) # DOTALL for multiline mood tags
                            if mood_match:
                                # Extracted mood from text could be used or compared with structured mood
                                message_for_tts = mood_match.group(2).strip()
                                print(f"[DEBUG] Stripped mood from text. Message for TTS now: '{message_for_tts[:100]}...'")
                            
                            message_for_tts = message_for_tts.replace('\n\n', '. ') # Consolidate paragraphs
                            message_for_tts = message_for_tts.replace('\n', ' ')    # Single newlines to space
                            message_for_tts = re.sub(r'\s+', ' ', message_for_tts)  # Multiple spaces to one
                            message_for_tts = re.sub(r'\(\s*\)', '', message_for_tts) # Remove empty parentheses
                            message_for_tts = message_for_tts.strip()
                            
                            print(f"[DEBUG] Final cleaned response text for TTS: '{message_for_tts[:100]}...'")

                            # Perform TTS or print if text_only mode
                            current_tts_mode = client_settings.get("tts_mode", "api")
                            if current_tts_mode == "text":
                                print(f"Assistant ({active_persona}): {message_for_tts}")
                            else:
                                print(f"[INFO] Requesting TTS. Mode: '{current_tts_mode}', Persona: '{active_persona}'")
                                audio_bytes, successful_engine = await self.tts_handler.generate_audio(
                                    text=message_for_tts, 
                                    persona_name=active_persona 
                                )
                                if audio_bytes and successful_engine:
                                    await self.play_audio(audio_bytes, successful_engine)
                                else:
                                    print(f"[ERROR] TTS failed for mode '{current_tts_mode}' after trying available engines.")
                            
                            # IMPORTANT: Wait for audio to finish BEFORE checking hooks or listening again
                            await self.audio_manager.wait_for_audio_completion()

                            # Check for conversation hook to continue or end the inner loop
                            if self.has_conversation_hook(response_text_from_server): # Use original server text for hook detection
                                print("[INFO] Conversation hook detected. Listening for follow-up...")
                                await self.display_manager.update_display("listening")
                                self.is_follow_up_interaction = True # Set for the NEXT capture_speech_with_vad
                                
                                follow_up_transcript = await self.capture_speech_with_vad()
                                
                                if follow_up_transcript:
                                    current_processing_transcript = follow_up_transcript # Continue inner loop with new transcript
                                else:
                                    print("[INFO] No follow-up speech captured. Ending conversation.")
                                    await self.display_manager.update_display("idle")
                                    current_processing_transcript = None # End inner loop
                                    self.is_follow_up_interaction = False # Reset
                            else:
                                print("[INFO] No conversation hook detected. Ending conversation.")
                                await self.display_manager.update_display("idle")
                                current_processing_transcript = None # End inner loop
                                self.is_follow_up_interaction = False # Reset
                        # --- End of Inner Conversational Loop ---
                        
                        # After inner loop finishes (conversation ends), ensure display is idle and reset follow-up flag
                        if not current_processing_transcript: # Double check if loop exited due to no transcript
                             await self.display_manager.update_display("idle")
                        self.is_follow_up_interaction = False 
                        print("[INFO] Inner conversation loop ended. Returning to wait for new wake event.")
                    # --- End of Outer Main Interaction Loop ---
        
        except ConnectionRefusedError:
            print(f"[FATAL ERROR] Connection refused to MCP server at {self.server_url}. Is it running?")
        except (ConnectionError, Exception) as e: # Catch other connection errors or major issues
            print(f"[FATAL ERROR] Main run loop encountered an unrecoverable error: {e}")
            traceback.print_exc()
        finally:
            print("[INFO] Exiting run loop. Performing cleanup...")
            self.cleanup()

    def cleanup(self):
        print("[INFO] Starting PiMCPClient cleanup...")
        if self.wakeword:
            self.wakeword.cleanup()
        if self.audio_manager: 
            self.audio_manager.cleanup() 
        if self.stt: 
            pass # VoskTranscriber might not have specific cleanup if PyAudio is handled by AudioManager
        if self.keyboard and hasattr(self.keyboard, 'close'): 
            try:
                self.keyboard.close()
                print("[INFO] Pi 500 Keyboard device closed.")
            except Exception as e:
                print(f"[ERROR] Error closing keyboard device: {e}")
        if self.display_manager:
            self.display_manager.cleanup() 
        print("[INFO] PiMCPClient cleanup finished.")


async def main():
    client = PiMCPClient()
    try:
        await client.run()
    except KeyboardInterrupt:
        print("\nExiting due to KeyboardInterrupt...")
        # Cleanup is now handled in client.run()'s finally block
    except Exception as e:
        print(f"[FATAL ERROR] Unhandled exception in main function: {e}")
        traceback.print_exc()
        # Cleanup is now handled in client.run()'s finally block
    finally:
        print("[INFO] Main function's execution finished.")


if __name__ == "__main__":
    from client_config import load_client_config
    load_client_config() 
    
    asyncio.run(main())
