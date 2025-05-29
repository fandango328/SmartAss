#!/usr/bin/env python3

import asyncio
import json
import os
import tempfile
import traceback
from pathlib import Path
import time # For monotonic clock in VAD
from datetime import datetime # For timestamp in send_to_laura_tool

# MCP Imports
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession # Assuming this is the correct import for session management

# Local Imports
from audio_manager_vosk import AudioManager
from client_tts_handler import TTSHandler
from display_manager import DisplayManager # Assuming you have this
from wake_word_detector import WakeWordDetector # Assuming you have this
from vosk_transcriber import VoskTranscriber # Assuming you have this

from client_config import (
    SERVER_URL,
    DEVICE_ID,
    VAD_SETTINGS,
    DEFAULT_PERSONA,
    KEEP_TEMP_AUDIO_FILES,
    # Add VOSK_MODEL_PATH to client_config.py if not already there
    # e.g., VOSK_MODEL_PATH = "/path/to/your/vosk-model-small-en-us-0.15"
)
# from client_secret import YOUR_API_KEY # If needed directly by client

# For VAD
import numpy as np

class PiMCPClient:
    def __init__(self, server_url: str, device_id: str):
        self.server_url = server_url
        self.device_id = device_id
        self.session_id: str | None = None
        self.active_persona: str = DEFAULT_PERSONA
        self.is_follow_up_interaction: bool = False
        self.debug_keep_audio_files = KEEP_TEMP_AUDIO_FILES
        self.mcp_session: ClientSession | None = None # To store the active session

        # Initialize core components
        self.audio_manager = AudioManager()
        self.tts_handler = TTSHandler()
        self.display_manager = DisplayManager()
        self.wake_word_detector = WakeWordDetector()
        
        vosk_model_path = VAD_SETTINGS.get("vosk_model_path", "model") # Default if not in VAD_SETTINGS
        if not Path(vosk_model_path).exists():
            print(f"[WARN] Vosk model path '{vosk_model_path}' does not exist. Please check client_config.py or VAD_SETTINGS.")
            # Fallback or error handling might be needed here
        self.transcriber = VoskTranscriber(model_path=vosk_model_path)
        
        print(f"[PiMCPClient DEBUG] Effective VAD params: {VAD_SETTINGS}")

    async def initialize_session(self, read, write):
        """Initializes the client session with the MCP server."""
        try:
            # The ClientSession is now managed by the sse_client context manager
            # We need to call register_device using the provided session object
            print(f"[INFO] SSE client connected. Attempting to register device: {self.device_id}")
            
            # Assuming `self.mcp_session` is the `session` object yielded by `ClientSession(read, write)`
            # in the main run loop. We need to pass `read` and `write` to ClientSession.
            # The original full client had a nested ClientSession context manager here,
            # which is redundant if sse_client already provides one or if we manage it in run().
            # For now, let's assume self.mcp_session is already set by the run() loop's context.

            if not self.mcp_session:
                print("[ERROR] MCP session object not available for device registration.")
                return False

            # Call the 'register_device' tool on the server
            registration_payload = {
                "device_id": self.device_id,
                "capabilities": { # Define your client's capabilities
                    "input": ["text", "audio"],
                    "output": ["text", "audio"],
                    "tts_mode": "local", # Example
                    "api_tts_provider": "elevenlabs" # Example
                }
            }
            print(f"[INFO] Calling 'register_device' tool with payload: {registration_payload}")
            response = await self.mcp_session.call_tool("register_device", **registration_payload)
            
            print(f"[INFO] 'register_device' tool response: {response}")

            if response and response.get("session_id"):
                self.session_id = response["session_id"]
                # Store capabilities if provided in response (though we sent them)
                self.capabilities = response.get("capabilities", registration_payload["capabilities"])
                print(f"[INFO] Device registration successful. Session ID: {self.session_id}, Capabilities: {self.capabilities}")
                return True
            else:
                print(f"[ERROR] Device registration failed. Response: {response}")
                return False
        except Exception as e:
            print(f"[ERROR] Error during session initialization/device registration: {e}")
            traceback.print_exc()
            return False

    async def play_audio(self, audio_bytes: bytes, source_engine: str):
        """
        Plays audio bytes by saving to a temp file, queuing with AudioManager,
        and then WAITING for AudioManager to signal its completion.
        """
        if not audio_bytes:
            print("[PiMCPClient.play_audio] No audio bytes to play.")
            return

        temp_dir = Path(tempfile.gettempdir())
        fname_base = "assistant_response_temp"
        ext = ".wav" if source_engine.lower() in ["piper", "cartesia"] else ".mp3"
        fname = temp_dir / (fname_base + ext)

        try:
            with open(fname, "wb") as f:
                f.write(audio_bytes)
            
            print(f"[PiMCPClient.play_audio] Queuing {fname} (from {source_engine}) with AudioManager.")
            await self.audio_manager.queue_audio(audio_file=str(fname))
            
            print(f"[PiMCPClient.play_audio] Waiting for AudioManager to complete playback of {fname}...")
            await self.audio_manager.wait_for_audio_completion()
            print(f"[PiMCPClient.play_audio] AudioManager reports playback complete for {fname}.")

        except Exception as e:
            print(f"[ERROR] PiMCPClient.play_audio: Failed to play audio from {fname}: {e}")
            traceback.print_exc()
        finally:
            if os.path.exists(fname) and not self.debug_keep_audio_files:
                try:
                    os.remove(fname)
                except Exception as e_del:
                    print(f"[WARN] PiMCPClient.play_audio: Failed to delete temp audio file {fname}: {e_del}")
            print(f"[DEBUG] PiMCPClient.play_audio finished its own operations for {fname}.")

    async def capture_speech_with_vad(self) -> str | None:
        """
        Captures audio using VAD loop with Vosk for transcription.
        """
        await self.audio_manager.initialize_input()
        audio_stream = await self.audio_manager.start_listening()
        if not audio_stream:
            print("[ERROR] Failed to start audio stream for VAD.")
            return None

        print(f"[CAPTURE SPEECH] Listening {'(follow-up)' if self.is_follow_up_interaction else ''}. "
              f"Initial Timeout: {VAD_SETTINGS['follow_up_initial_listen_timeout_s'] if self.is_follow_up_interaction else VAD_SETTINGS['initial_listen_timeout_s']:.1f}s. "
              f"Max Rec: {VAD_SETTINGS['follow_up_max_recording_time_s'] if self.is_follow_up_interaction else VAD_SETTINGS['max_recording_time']:.1f}s")
        
        print(f"[CAPTURE SPEECH] VAD Params: EnergyThresh={VAD_SETTINGS['energy_threshold']:.4f}, "
              f"SilenceDur={VAD_SETTINGS['silence_duration']:.1f}s, MinSpeech={VAD_SETTINGS['min_speech_duration']:.1f}s")

        self.transcriber.reset()

        overall_listen_start_time_mono = time.monotonic()
        initial_timeout = VAD_SETTINGS['follow_up_initial_listen_timeout_s'] if self.is_follow_up_interaction else VAD_SETTINGS['initial_listen_timeout_s']
        max_recording_time = VAD_SETTINGS['follow_up_max_recording_time_s'] if self.is_follow_up_interaction else VAD_SETTINGS['max_recording_time']
        
        energy_thresh = VAD_SETTINGS['energy_threshold']
        continued_ratio = VAD_SETTINGS['continued_threshold_ratio']
        silence_dur_s = VAD_SETTINGS['silence_duration']
        min_speech_s = VAD_SETTINGS['min_speech_duration']
        frame_history_len = VAD_SETTINGS['frame_history_length']

        frames_per_second = self.audio_manager.sample_rate / self.audio_manager.frame_length
        silence_frames_needed = int(silence_dur_s * frames_per_second)
        
        frame_history = []
        silence_frames_count = 0
        voice_started = False
        speech_start_time_mono = 0

        try:
            while True:
                current_mono_time = time.monotonic()
                if not voice_started and (current_mono_time - overall_listen_start_time_mono > initial_timeout):
                    print(f"[CAPTURE SPEECH] VAD initial timeout ({initial_timeout:.1f}s). No voice detected.")
                    return None
                if voice_started and (current_mono_time - speech_start_time_mono > max_recording_time):
                    print(f"[CAPTURE SPEECH] VAD max recording time ({max_recording_time:.1f}s) reached.")
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
                        speech_start_time_mono = current_mono_time
                        silence_frames_count = 0
                        print(f"[CAPTURE SPEECH] VAD: Voice started. Energy: {avg_energy:.4f} (Thresh: {energy_thresh:.4f})")
                else:
                    if avg_energy > (energy_thresh * continued_ratio):
                        silence_frames_count = 0
                    else:
                        silence_frames_count += 1

                    speech_duration_s = current_mono_time - speech_start_time_mono
                    if silence_frames_count >= silence_frames_needed and speech_duration_s >= min_speech_s:
                        print(f"[CAPTURE SPEECH] VAD: End of speech by silence. Duration: {speech_duration_s:.2f}s")
                        break
                    if is_final_vosk and speech_duration_s >= min_speech_s:
                        print(f"[CAPTURE SPEECH] VAD: End of speech by Vosk final. Duration: {speech_duration_s:.2f}s")
                        break
                
                if partial_text_vosk:
                    if not hasattr(self, "last_partial_print_time") or (current_mono_time - getattr(self, "last_partial_print_time", 0) > 2.0):
                        print(f"[CAPTURE SPEECH] Partial: {partial_text_vosk}")
                        self.last_partial_print_time = current_mono_time

            final_transcript = self.transcriber.get_final_text()
            print(f"[CAPTURE SPEECH] Raw transcript from Vosk: '{final_transcript}'")

            if final_transcript:
                final_transcript = final_transcript.strip()
                if not final_transcript: return None
                num_words = len(final_transcript.split())
                min_chars_single_word = VAD_SETTINGS.get('vosk_reject_min_chars_if_single_word', 4)
                min_words_overall = VAD_SETTINGS.get('vosk_reject_min_words', 2)

                if num_words == 1 and len(final_transcript) < min_chars_single_word:
                    print(f"[CAPTURE SPEECH] Rejecting very short single word: '{final_transcript}'")
                    return None
                # Adjusted logic for min_words: if it's less than min_words_overall, but not a single short word, it might be okay for some cases.
                # The original logic was a bit unclear. Let's assume if it passes single word check, and has at least one word, it's okay for now.
                # If stricter multi-word check is needed, re-add:
                # if num_words < min_words_overall and num_words > 0:
                #     print(f"[CAPTURE SPEECH] Transcript too short (words: {num_words} < {min_words_overall}): '{final_transcript}'")
                #     return None
                print(f"[CAPTURE SPEECH] Final processed transcript: '{final_transcript}'")
                return final_transcript
            return None
        except Exception as e:
            print(f"[ERROR] Error during VAD/transcription: {e}")
            traceback.print_exc()
            return None
        finally:
            await self.audio_manager.stop_listening()
            if hasattr(self, "last_partial_print_time"):
                del self.last_partial_print_time

    async def send_to_laura_tool(self, transcript: str) -> dict | None:
        """Sends transcript to the LAURA/MCP server by calling the 'run_LAURA' tool."""
        if not self.session_id or not self.mcp_session:
            print("[ERROR] Session not initialized. Cannot send message to LAURA tool.")
            return {"text": "Error: Client session not ready.", "mood": "error"}
        try:
            print(f"[INFO] Calling 'run_LAURA' tool with transcript: '{transcript}'")

            tool_call_args = {
                "session_id": self.session_id,
                "input_type": "text",
                "payload": {"text": transcript},
                "output_mode": ["audio", "text"], 
                "broadcast": False,
                "timestamp": datetime.utcnow().isoformat()
            }

            response_payload = await self.mcp_session.call_tool("run_LAURA", **tool_call_args)
            
            print(f"[INFO] 'run_LAURA' tool response payload: {response_payload}")

            if response_payload:
                if "text" in response_payload: # Assuming server returns at least 'text'
                    return response_payload
                elif "error" in response_payload:
                    print(f"[ERROR] 'run_LAURA' tool returned an error: {response_payload.get('error')}")
                    return {"text": response_payload.get("text", f"Sorry, an error occurred: {response_payload.get('error', 'Unknown error')}"), 
                            "mood": "error", 
                            **response_payload}
                else:
                    print(f"[WARN] 'run_LAURA' tool response has unexpected structure: {response_payload}")
                    return {"text": "Sorry, I received an unexpected response.", "mood": "confused"}
            print(f"[WARN] No valid response payload from 'run_LAURA' tool for transcript: '{transcript}'")
            return {"text": "Sorry, no response received from LAURA tool.", "mood": "error"}

        except Exception as e:
            print(f"[ERROR] Failed to call 'run_LAURA' tool or process its response: {e}")
            traceback.print_exc()
            return {"text": "Sorry, there was a communication problem with the LAURA tool.", "mood": "error"}

    def has_conversation_hook(self, response_text: str | None) -> bool:
        if not response_text: return False
        if "?" in response_text: return True
        response_lower = response_text.lower()
        # Using original hook logic which included "."
        # Consider if "." is too broad for a hook.
        continuation_phrases = [
            "let me know", "tell me", "what do you think", "anything else", 
            "can i help with anything else", ".", "what's next" 
        ]
        for phrase in continuation_phrases:
            if phrase in response_lower:
                # Special handling for "." to avoid triggering on every sentence.
                # Only trigger if it's at the end or implies continuation.
                if phrase == ".":
                    if response_text.endswith(".") and not response_text.endswith("..?"): # Avoid triggering on ellipses or question marks ending with a period.
                         # Simple check: if it's just a period, might be a short statement.
                         # A more complex check might be needed if "." is a common sentence ender.
                         # For now, let's assume if it ends with a period AND is in the list, it's a hook.
                         # This part of the logic might need refinement based on typical AI responses.
                         # Let's be more specific: only if it's a question-like phrase ending with period.
                         pass # Original logic was to return True here. Re-evaluating.
                else: # For other phrases
                    return True
        # If only "." was found, and it wasn't part of a longer hook phrase, decide if it's a hook.
        # For simplicity, if "?" is the primary hook, let's make "." less aggressive.
        # If you want "." to be a strong hook, return True if response_text.endswith(".")
        return False # Default if no strong hooks found. "?" is handled above.

    async def run(self):
        print("[INFO] PiMCPClient run loop started.")
        await self.display_manager.update_display("idle")

        # The sse_client context manager handles the connection.
        # ClientSession is instantiated within this loop per connection attempt.
        async for read, write in sse_client(self.server_url, headers={}): # Add bearer token header if server requires for SSE
            print("[INFO] SSE connection established. Creating ClientSession...")
            try:
                async with ClientSession(read, write) as session: # MCP library's ClientSession
                    self.mcp_session = session # Store the active session
                    print("[INFO] ClientSession active.")

                    if not await self.initialize_session(read, write): # Pass read/write if needed, though session obj is key
                        print("[ERROR] Failed to initialize session with server. Retrying SSE connection...")
                        await asyncio.sleep(5)
                        continue # Go to next iteration of sse_client

                    print("[INFO] Session established with server. Entering main interaction loop.")
                    # Inner loop for interactions within an active session
                    while True: 
                        await self.display_manager.update_display("idle", mood="neutral")
                        print("[INFO] Waiting for wake word...")
                        
                        detected_wake_word = await self.wake_word_detector.detect()
                        if not detected_wake_word:
                            await asyncio.sleep(0.1)
                            continue

                        print(f"[INFO] Wakeword '{detected_wake_word}' detected.")
                        self.is_follow_up_interaction = False
                        await self.display_manager.update_display("listening", mood="curious")
                        
                        transcript = await self.capture_speech_with_vad()
                        if not transcript:
                            print("[INFO] No transcript from initial capture. Returning to wake word.")
                            continue

                        await self.display_manager.update_display("thinking", mood="thoughtful")
                        laura_response_data = await self.send_to_laura_tool(transcript)

                        if not laura_response_data or "text" not in laura_response_data:
                            print(f"[ERROR] Invalid or no response from LAURA tool. Response: {laura_response_data}")
                            await self.display_manager.update_display("error", mood="confused")
                            await asyncio.sleep(2)
                            continue

                        response_text_from_server = laura_response_data.get("text")
                        mood_for_display = laura_response_data.get("mood", "casual")
                        self.active_persona = laura_response_data.get("active_persona", self.active_persona)

                        cleaned_response_for_tts = response_text_from_server
                        # Simple mood stripping, server should ideally send clean text for TTS
                        if mood_for_display and response_text_from_server.startswith(f"[{mood_for_display.lower()}]"):
                             cleaned_response_for_tts = response_text_from_server[len(mood_for_display)+2:].lstrip()
                        elif mood_for_display and response_text_from_server.startswith(f"[{mood_for_display.capitalize()}]"):
                             cleaned_response_for_tts = response_text_from_server[len(mood_for_display)+2:].lstrip()


                        print(f"[DEBUG] Final cleaned response text for TTS: '{cleaned_response_for_tts[:80]}...'")
                        
                        await self.display_manager.update_display("speaking", mood=mood_for_display)
                        audio_bytes, successful_engine = await self.tts_handler.generate_audio(cleaned_response_for_tts)

                        if audio_bytes and successful_engine:
                            print(f"[INFO] PiMCPClient.run is about to call self.play_audio for assistant response.")
                            await self.play_audio(audio_bytes, successful_engine)
                            print(f"[INFO] PiMCPClient.run: self.play_audio has completed.")
                        else:
                            print(f"[ERROR] TTS failed or no audio bytes, skipping playback.")
                        
                        # Conversation Hook and Follow-up Loop
                        while self.has_conversation_hook(response_text_from_server):
                            print("[INFO] Conversation hook detected. Listening for follow-up...")
                            await self.display_manager.update_display("listening", mood=mood_for_display)
                            self.is_follow_up_interaction = True
                            
                            follow_up_transcript = await self.capture_speech_with_vad()
                            if not follow_up_transcript:
                                print("[INFO] No follow-up speech captured. Ending conversation.")
                                break 

                            print(f"[INFO] Processing follow-up transcript: '{follow_up_transcript}'")
                            await self.display_manager.update_display("thinking", mood="thoughtful")
                            laura_response_data = await self.send_to_laura_tool(follow_up_transcript)

                            if not laura_response_data or "text" not in laura_response_data:
                                print(f"[ERROR] Invalid or no response for follow-up. Data: {laura_response_data}")
                                break 
                            
                            response_text_from_server = laura_response_data.get("text")
                            mood_for_display = laura_response_data.get("mood", "casual")
                            self.active_persona = laura_response_data.get("active_persona", self.active_persona)

                            cleaned_response_for_tts = response_text_from_server
                            if mood_for_display and response_text_from_server.startswith(f"[{mood_for_display.lower()}]"):
                                cleaned_response_for_tts = response_text_from_server[len(mood_for_display)+2:].lstrip()
                            elif mood_for_display and response_text_from_server.startswith(f"[{mood_for_display.capitalize()}]"):
                                cleaned_response_for_tts = response_text_from_server[len(mood_for_display)+2:].lstrip()

                            await self.display_manager.update_display("speaking", mood=mood_for_display)
                            audio_bytes, successful_engine = await self.tts_handler.generate_audio(cleaned_response_for_tts)

                            if audio_bytes and successful_engine:
                                await self.play_audio(audio_bytes, successful_engine)
                            else:
                                print(f"[ERROR] TTS failed for follow-up, skipping playback.")
                        
                        print("[INFO] Inner conversation loop ended. Returning to wait for new wake event.")

            except asyncio.CancelledError:
                print("[INFO] Main interaction loop (while True) cancelled.")
                break # Exit the sse_client loop for this connection
            except ConnectionRefusedError:
                print("[ERROR] SSE Connection refused by server. Retrying in 5s...")
                await asyncio.sleep(5)
                # The outer sse_client loop will handle reconnection.
            except Exception as e_main_loop:
                print(f"[ERROR] Unhandled exception in main interaction loop: {e_main_loop}")
                traceback.print_exc()
                await self.display_manager.update_display("error", mood="confused")
                await asyncio.sleep(5)
                break # Exit sse_client loop on other major errors
            finally:
                print("[INFO] Exiting main interaction loop block (active session).")
                self.mcp_session = None # Clear session on exit from this block

        print("[INFO] SSE client loop has exited.")


    async def cleanup(self):
        print("[INFO] Starting PiMCPClient cleanup...")
        if hasattr(self, 'wake_word_detector') and self.wake_word_detector and \
           hasattr(self.wake_word_detector, 'cleanup') and callable(self.wake_word_detector.cleanup):
            self.wake_word_detector.cleanup() # Assuming sync
            print("[INFO] WakeWordDetector cleaned up.")
        
        if hasattr(self, 'audio_manager') and self.audio_manager and \
           hasattr(self.audio_manager, 'cleanup') and callable(self.audio_manager.cleanup):
            if asyncio.iscoroutinefunction(self.audio_manager.cleanup):
                await self.audio_manager.cleanup()
            else:
                self.audio_manager.cleanup()
            print("[INFO] AudioManager cleaned up.")

        if hasattr(self, 'display_manager') and self.display_manager and \
           hasattr(self.display_manager, 'cleanup') and callable(self.display_manager.cleanup):
            self.display_manager.cleanup() # Assuming sync
            print("[INFO] DisplayManager cleaned up.")
        
        if hasattr(self, 'transcriber') and self.transcriber and \
           hasattr(self.transcriber, 'cleanup') and callable(self.transcriber.cleanup):
            self.transcriber.cleanup() # Assuming sync
            print("[INFO] Transcriber cleaned up.")
        print("[INFO] PiMCPClient cleanup finished.")

async def main():
    # Ensure VOSK_MODEL_PATH is set in client_config or VAD_SETTINGS correctly
    # Example: VAD_SETTINGS["vosk_model_path"] = "/home/user/LAURA/models/vosk-model-small-en-us-0.15"
    if not VAD_SETTINGS.get("vosk_model_path") or not Path(VAD_SETTINGS["vosk_model_path"]).exists():
        print("*********************************************************************")
        print("ERROR: Vosk model path not found or not configured in client_config.py (via VAD_SETTINGS['vosk_model_path']).")
        print(f"Attempted path: {VAD_SETTINGS.get('vosk_model_path')}")
        print("Please ensure 'vosk_model_path' is correctly set in VAD_SETTINGS within client_config.py")
        print("*********************************************************************")
        return

    client = PiMCPClient(server_url=SERVER_URL, device_id=DEVICE_ID)
    try:
        await client.run()
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received by main(). Exiting...")
    except Exception as e:
        print(f"[ERROR] Unhandled exception in main(): {e}")
        traceback.print_exc()
    finally:
        print("[INFO] Main function's execution finished. Performing final cleanup...")
        await client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[INFO] Application terminated by user (KeyboardInterrupt in __main__ block).")
    except Exception as e_outer: # Catch any other unexpected errors during asyncio.run
        print(f"[FATAL] Outer exception during asyncio.run: {e_outer}")
        traceback.print_exc()
    finally:
        print("[INFO] Application shutdown sequence complete.")
