#!/usr/bin/env python3

import os
import asyncio
import traceback
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Union, Any
import config
from config import MOOD_MAPPINGS, ACTIVE_PERSONA

async def capture_speech(is_follow_up=False):
    """
    Unified function to capture and transcribe speech, replacing both
    handle_voice_query and conversation_mode.

    Args:
        is_follow_up (bool): Whether this is a follow-up question (affects timeout)

    Returns:
        str or None: Transcribed text if speech was detected, None otherwise
    """
    try:
        # Determine appropriate timeouts based on context
        if is_follow_up:
            # For follow-up questions, check if we have specific config values
            initial_timeout = VAD_SETTINGS.get("follow_up_timeout", 4.0)
            max_recording_time = VAD_SETTINGS.get("follow_up_max_recording", 45.0)
        else:
            # For initial queries, use the primary configuration values
            initial_timeout = VOICE_START_TIMEOUT
            max_recording_time = VAD_SETTINGS["max_recording_time"]

        waiting_message = f"\n{Fore.MAGENTA}Waiting for response...{Fore.WHITE}" if is_follow_up else f"{Fore.BLUE}Listening...{Fore.WHITE}"
        print(waiting_message)

        # Ensure audio playback is complete before starting listening (for follow-ups)
        if is_follow_up:
            await audio_manager.wait_for_audio_completion()
            await asyncio.sleep(0.5)  # Small buffer delay

        # Start listening
        if TRANSCRIPTION_MODE == "local":
            # Reset the transcriber state
            transcriber.reset()

            audio_stream, _ = await audio_manager.start_listening()
            voice_detected = False

            # Keep processing audio frames until we detect end of speech
            print(f"{Fore.MAGENTA}Waiting for voice...{Fore.WHITE}")
            start_time = time.time()

            # For Vosk, we need a different approach
            if TRANSCRIPTION_ENGINE == "vosk":
                # Get VAD settings
                energy_threshold = VAD_SETTINGS["energy_threshold"]
                continued_ratio = VAD_SETTINGS["continued_threshold_ratio"]
                silence_duration = VAD_SETTINGS["silence_duration"]
                min_speech_duration = VAD_SETTINGS["min_speech_duration"]
                speech_buffer_time = VAD_SETTINGS["speech_buffer_time"]

                # Manual stop tracking
                manual_stop = False

                # Calculate frames needed for silence duration
                max_silence_frames = int(silence_duration * 16000 / audio_manager.frame_length)

                # State tracking
                silence_frames = 0
                last_partial_time = time.time()
                frame_history = []
                is_speaking = False
                speech_start_time = None

                while True:
                    # Check for initial timeout
                    if not voice_detected and (time.time() - start_time) > initial_timeout:
                        if not is_follow_up:  # More verbose for initial queries
                            print(f"DEBUG: Voice not detected - energy threshold: {energy_threshold:.6f}")
                        print(f"{Fore.YELLOW}{'No response detected' if is_follow_up else 'Voice start timeout'}{Fore.WHITE}")
                        break

                    # Read audio frame
                    pcm_bytes = audio_manager.read_audio_frame()
                    if not pcm_bytes:
                        await asyncio.sleep(0.01)
                        continue

                    # Check for manual stop if voice was detected
                    if voice_detected:
                        manual_stop = await check_manual_stop()
                        if manual_stop:
                            speech_duration = time.time() - speech_start_time
                            if speech_duration > min_speech_duration:
                                print(f"{Fore.MAGENTA}Manual stop triggered after {speech_duration:.1f}s{Fore.WHITE}")
                                await asyncio.sleep(speech_buffer_time)
                                break
                            else:
                                print(f"{Fore.YELLOW}Recording too short for manual stop ({speech_duration:.1f}s < {min_speech_duration}s){Fore.WHITE}")
                                manual_stop = False

                    # Process with Vosk
                    is_end, is_speech, partial_text = transcriber.process_frame(pcm_bytes)

                    # Calculate energy level (RMS)
                    float_data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    energy = np.sqrt(np.mean(float_data**2)) if len(float_data) > 0 else 0

                    # Add to frame history for smoother energy detection
                    frame_history.append(energy)
                    if len(frame_history) > 10:
                        frame_history.pop(0)

                    # Calculate average energy
                    avg_energy = sum(frame_history) / len(frame_history) if frame_history else 0

                    # Display partial results (less frequently for follow-ups)
                    current_time = time.time()
                    partial_display_interval = 5 if is_follow_up else 2
                    if partial_text and (current_time - last_partial_time) > partial_display_interval:
                        last_partial_time = current_time
                        print(f"Partial: {partial_text}")

                    # VAD STATE MACHINE
                    if avg_energy > energy_threshold and not is_speaking:
                        # Speech just started
                        voice_color = Fore.GREEN if is_follow_up else Fore.BLUE
                        print(f"{voice_color}Voice detected{' (energy: ' + str(avg_energy)[:6] + ')' if not is_follow_up else ''}{Fore.WHITE}")
                        voice_detected = True
                        is_speaking = True
                        speech_start_time = time.time()
                        silence_frames = 0

                    elif is_speaking:
                        speech_duration = time.time() - speech_start_time

                        # Check if energy is above the continued threshold
                        if avg_energy > (energy_threshold * continued_ratio):
                            silence_frames = 0
                        else:
                            silence_frames += 1

                        # Check for end conditions
                        if (silence_frames >= max_silence_frames and
                            speech_duration > min_speech_duration):
                            print(f"{Fore.MAGENTA}End of {'response' if is_follow_up else 'speech'} detected{Fore.WHITE}")
                            await asyncio.sleep(speech_buffer_time)
                            break

                        # Check for maximum duration
                        if speech_duration > max_recording_time:
                            print(f"{Fore.RED}Maximum recording time reached{Fore.WHITE}")
                            break

                    # Reduced CPU usage
                    await asyncio.sleep(0.01)

                # Update display state based on context
                if not voice_detected:
                    if is_follow_up:
                        await display_manager.update_display('idle')
                    else:
                        await display_manager.update_display('sleep')
                    return None

                # Get final transcription
                transcript = transcriber.get_final_text()
                print(f"Raw transcript: '{transcript}'")

                # Apply cleanup for common Vosk errors
                if transcript:
                    # Fix "that were" at beginning which is a common Vosk error
                    transcript = re.sub(r'^that were\s+', '', transcript)
                    transcript = re.sub(r'^that was\s+', '', transcript)

                    # Reject single-word responses as likely noise
                    min_word_length = 1 if is_follow_up else 4
                    if len(transcript.split()) <= 1 and len(transcript) < min_word_length:
                        print(f"Discarding too-short transcript: '{transcript}'")
                        return None

            else:
                # Handle Whisper transcription
                recording_complete = False
                is_speech = False

                while not recording_complete:
                    if not voice_detected and (time.time() - start_time > initial_timeout):
                        print(f"{Fore.YELLOW}{'No response detected' if is_follow_up else 'Voice start timeout'}{Fore.WHITE}")
                        break

                    pcm = audio_manager.read_audio_frame()
                    if not pcm:
                        await asyncio.sleep(0.01)
                        continue

                    recording_complete, is_speech = transcriber.process_frame(pcm)

                    if is_speech and not voice_detected:
                        voice_color = Fore.GREEN if is_follow_up else Fore.BLUE
                        print(f"{voice_color}Voice detected{Fore.WHITE}")
                        voice_detected = True
                        start_time = time.time()  # Reset timeout

                if not voice_detected:
                    print("No voice detected")
                    return None

                print(f"{Fore.MAGENTA}Transcribing {'conversation' if is_follow_up else ''}...{Fore.WHITE}")
                # Get final transcription
                transcript = transcriber.transcribe()
                print(f"Raw transcript: '{transcript}'")

            if not transcript:
                print("No transcript returned")
                return None

            print(f"Transcription: {transcript}")
            return transcript

        else:  # Remote transcription
            audio_stream, _ = await audio_manager.start_listening()

            recording = []
            start_time = time.time()
            voice_detected = False

            # Get VAD settings
            energy_threshold = VAD_SETTINGS["energy_threshold"]
            continued_ratio = VAD_SETTINGS["continued_threshold_ratio"]
            silence_duration = VAD_SETTINGS["silence_duration"]

            # Initial detection phase - different timeouts based on context
            print(f"{Fore.MAGENTA}Waiting for voice...{Fore.WHITE}")
            while (time.time() - start_time) < initial_timeout:
                pcm_bytes = audio_manager.read_audio_frame()
                if not pcm_bytes:
                    await asyncio.sleep(0.01)
                    continue

                # Convert bytes to int16 values
                pcm = np.frombuffer(pcm_bytes, dtype=np.int16)

                # Calculate energy (RMS)
                float_data = pcm.astype(np.float32) / 32768.0
                energy = np.sqrt(np.mean(float_data**2)) if len(float_data) > 0 else 0

                # Check if this is speech
                if energy > energy_threshold:
                    color = Fore.GREEN if is_follow_up else Fore.BLUE
                    print(f"{color}Voice detected{Fore.WHITE}")
                    voice_detected = True
                    recording.extend(pcm)
                    break

            # If no voice detected in initial phase, return
            if not voice_detected:
                print(f"No voice detected in {'' if is_follow_up else 'initial '}phase")
                return None

            # Continuous recording phase
            print(f"{Fore.MAGENTA}Recording...{Fore.WHITE}")
            silence_frames = 0
            silence_frame_threshold = int(silence_duration * audio_manager.sample_rate / audio_manager.frame_length)

            while True:
                pcm_bytes = audio_manager.read_audio_frame()
                if not pcm_bytes:
                    await asyncio.sleep(0.01)
                    continue

                # Convert bytes to int16 values
                pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
                recording.extend(pcm)

                # Calculate energy
                float_data = pcm.astype(np.float32) / 32768.0
                energy = np.sqrt(np.mean(float_data**2)) if len(float_data) > 0 else 0

                # Check if this frame has voice
                if energy > (energy_threshold * continued_ratio):
                    silence_frames = 0
                else:
                    silence_frames += 1

                # End recording conditions
                current_length = len(recording) / audio_manager.sample_rate

                if silence_frames >= silence_frame_threshold:
                    print(f"{Fore.MAGENTA}{'End of response' if is_follow_up else 'Silence'} detected, ending recording (duration: {current_length:.2f}s){Fore.WHITE}")
                    break
                elif current_length >= max_recording_time:
                    print(f"{Fore.MAGENTA}Maximum recording time reached{Fore.WHITE}")
                    break

            if recording:
                audio_array = np.array(recording, dtype=np.float32) / 32768.0
                transcript = await remote_transcriber.transcribe(audio_array)

        # Common post-processing for both transcription methods
        end_phrases = [
            "thank you for watching",
            "thanks for watching",
            "thank you for watching!",
            "thanks for watching!",
            "thanks you for watching",
            "thanks you for watching!"
        ]

        # Handle case where transcript might be a dictionary
        if isinstance(transcript, dict) and 'text' in transcript:
            transcript = transcript['text']

        if not transcript:
            print("No transcript returned from transcriber")
            return None

        if not isinstance(transcript, str):
            print(f"Invalid transcript type: {type(transcript)}")
            return None

        if transcript.lower().strip() in end_phrases:
            print("End phrase detected, ignoring...")
            return None

        # Output recognized speech
        if is_follow_up:
            print(f"\n{Style.BRIGHT}You said:{Style.NORMAL} {transcript}\n")

        # Prevent rejection of valid short phrases
        if len(transcript.strip()) > 0:
            print(f"Final transcript: '{transcript}'")
            return transcript.strip()
        else:
            print("Empty transcript after processing")
            return None

    finally:
        await audio_manager.stop_listening()


async def run_vad_calibration(system_manager, display_manager):
    """Run VAD calibration process with proper manager coordination"""
    try:
        print("Starting VAD calibration...")
        
        # Show calibration image through system manager
        await system_manager.show_calibration_image()
        
        # Create and run calibration process
        calibration_script = Path(__file__).parent / "vad_calib.py"
        process = await asyncio.create_subprocess_exec(
            'python3', str(calibration_script),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        output = stdout.decode()
        error = stderr.decode()
        
        # Check success and update display
        if "CALIBRATION_COMPLETE" in output:
            print("VAD calibration completed successfully")
            await display_manager.update_display('listening')
            return True
        else:
            print("VAD calibration failed")
            print(f"Output: {output}")
            print(f"Error: {error}")
            return False
            
    except Exception as e:
        print(f"Error during calibration: {e}")
        traceback.print_exc()
        return False

async def handle_calendar_query(calendar_manager, query_type, **kwargs):
    """Handle calendar queries with proper error handling"""
    try:
        if query_type == "next_event":
            return calendar_manager.get_next_event()
        elif query_type == "day_schedule":
            return calendar_manager.get_day_schedule()
        else:
            raise ValueError(f"Unsupported calendar query type: {query_type}")
            
    except Exception as e:
        print(f"Calendar query error: {e}")
        traceback.print_exc()
        return f"Error processing calendar query: {str(e)}"

async def process_response_content(content, chat_log, system_manager, display_manager, audio_manager, notification_manager):
    """
    Process API response content for voice generation and chat log storage.
    Handles mood detection, content formatting, ensures original message is saved, and robustly extracts text from 
    structured content (blocks, dicts, etc.). Provides deep debug logging for every step.
    
    Args:
        content: Raw response from API (can be str, dict, or structured content blocks)
        chat_log: Global chat history
        system_manager: System manager instance for validation
        display_manager: Display manager for mood updates
        audio_manager: Audio manager for state checking
        notification_manager: Notification manager for queue checks
        
    Returns:
        str: Formatted message ready for voice generation
    """
    print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')}] === Response Processing Debug ===")
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Incoming content type: {type(content)}")
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Incoming content raw: {repr(content)}")
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Audio state - Speaking: {getattr(audio_manager, 'is_speaking', None)}, Playing: {getattr(audio_manager, 'is_playing', None)}")
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Notification queue size: {getattr(notification_manager.notification_queue, 'qsize', lambda: 'N/A')()}")

    def extract_text_from_content(content, depth=0):
        indent = "  " * depth
        print(f"{indent}Extracting text from: {type(content)}: {repr(content)[:80]}")

        # Anthropic block objects (e.g., TextBlock), prefer .text field
        if hasattr(content, "text") and isinstance(getattr(content, "text", None), str):
            print(f"{indent}Block object with .text: {content.text[:80]}")
            return content.text
        if hasattr(content, "content"):
            print(f"{indent}Block object with .content, recursing...")
            return extract_text_from_content(content.content, depth+1)
        if isinstance(content, str):
            print(f"{indent}Returning string directly.")
            return content
        elif isinstance(content, dict):
            if "type" in content:
                print(f"{indent}Block type: {content.get('type')}")
            # Prefer 'text', then 'content'
            if 'text' in content and isinstance(content['text'], str):
                print(f"{indent}Found 'text' field.")
                return content['text']
            elif 'content' in content and isinstance(content['content'], str):
                print(f"{indent}Found 'content' field (str).")
                return content['content']
            elif 'content' in content and isinstance(content['content'], list):
                print(f"{indent}Found 'content' field (list), recursing...")
                return extract_text_from_content(content['content'], depth+1)
            else:
                # Fallback: join all string values in dict
                join_strs = [str(v) for v in content.values() if isinstance(v, str)]
                print(f"{indent}Fallback join of string values in dict: {join_strs}")
                return " ".join(join_strs)
        elif isinstance(content, list):
            print(f"{indent}Content is a list of length {len(content)}")
            result = []
            for idx, block in enumerate(content):
                print(f"{indent}Processing list item {idx}: {type(block)}")
                result.append(extract_text_from_content(block, depth+1))
            joined = " ".join([x for x in result if x])
            print(f"{indent}Joined list result: {repr(joined)[:80]}")
            return joined
        else:
            print(f"{indent}Unknown content type, using str().")
            return str(content)

    try:
        # Step 1: Parse content into usable text
        text = extract_text_from_content(content)
        print(f"\nDEBUG - Full response text after extraction:\n{text}\n")
        
        # Step 2: Parse mood and preserve complete message
        original_message = text  # Store original message before any processing
        mood_match = re.match(r'^\[(.*?)\]([\s\S]*)', text, re.IGNORECASE)
        if mood_match:
            raw_mood = mood_match.group(1)         # Extract mood from [mood]
            clean_message = mood_match.group(2)     # Get complete message content
            mapped_mood = None
            if 'MOOD_MAPPINGS' in globals():
                mapped_mood = MOOD_MAPPINGS.get(raw_mood.lower(), None)
            else:
                mapped_mood = raw_mood.lower()
            if mapped_mood and display_manager:
                print(f"DEBUG - Mood detected and updating display: {mapped_mood}")
                await display_manager.update_display('speaking', mood=mapped_mood)
        else:
            clean_message = text
            print("DEBUG - No mood detected in message")

        # Step 3: Format message for voice generation
        formatted_message = clean_message

        # Deep debug before formatting
        print(f"DEBUG - Pre-format message for voice: {repr(formatted_message)}")
        
        # Convert all newlines to spaces for continuous speech
        formatted_message = formatted_message.replace('\n', ' ')
        # Convert list markers to natural speech transitions
        formatted_message = re.sub(r'(\d+)\.\s*', r'Number \1: ', formatted_message)
        formatted_message = re.sub(r'^\s*-\s*', 'Also, ', formatted_message)
        # Clean up any multiple spaces from formatting
        formatted_message = re.sub(r'\s+', ' ', formatted_message)
        # Add natural pauses after sentences
        formatted_message = re.sub(r'(?<=[.!?])\s+(?=[A-Z])', '. ', formatted_message)
        # Remove any trailing/leading whitespace
        formatted_message = formatted_message.strip()
        
        print(f"DEBUG - Formatted message for voice (full):\n{formatted_message}\n")
        
        # Step 4: Return formatted message for voice generation
        return original_message, formatted_message

    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Content processing error: {e}")
        traceback.print_exc()
        return "I apologize, but I encountered an error processing the response."
        
async def execute_calendar_query(tool_call: dict, calendar_service, notification_manager) -> str:
    """
    Execute calendar queries through the tool registry pattern.
    Coordinates with notification system for calendar updates.
    
    Args:
        tool_call: Dictionary containing query parameters
        calendar_service: Google Calendar service instance
        notification_manager: Notification manager for calendar alerts
        
    Returns:
        str: Formatted calendar response
    """
    try:
        query_type = tool_call.get("query_type")
        if not query_type:
            raise ValueError("Missing query_type in calendar query")
            
        now = datetime.now(timezone.utc)
        
        if query_type == "next_event":
            events_result = calendar_service.events().list(
                calendarId='primary',
                timeMin=now.isoformat(),
                maxResults=1,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            if not events:
                return "No upcoming events found."
                
            event = events[0]
            start = event['start'].get('dateTime', event['start'].get('date'))
            
            # Register event for notification tracking
            if notification_manager:
                await notification_manager.register_calendar_event(
                    event_id=event['id'],
                    summary=event['summary'],
                    start_time=start
                )
            
            return format_calendar_event(event, now)
            
        elif query_type == "day_schedule":
            date_str = tool_call.get("date")
            target_date = datetime.strptime(date_str, "%Y-%m-%d") if date_str else now
            
            start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
            
            events_result = calendar_service.events().list(
                calendarId='primary',
                timeMin=start.isoformat() + 'Z',
                timeMax=end.isoformat() + 'Z',
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            return format_day_schedule(events_result, start)
        
        else:
            raise ValueError(f"Unsupported calendar query type: {query_type}")
            
    except Exception as e:
        print(f"Calendar query error: {e}")
        traceback.print_exc()
        return f"Error processing calendar query: {str(e)}"

def format_calendar_event(event: dict, now: datetime) -> str:
    """Format a single calendar event with timing information."""
    start = event['start'].get('dateTime', event['start'].get('date'))
    
    if 'T' in start:  # DateTime
        start_time = datetime.fromisoformat(start.replace('Z', '+00:00'))
        time_until = start_time - now
        if time_until.days > 0:
            timing = f"in {time_until.days} days"
        elif time_until.seconds > 3600:
            hours = time_until.seconds // 3600
            timing = f"in {hours} hours"
        else:
            minutes = (time_until.seconds % 3600) // 60
            timing = f"in {minutes} minutes"
    else:  # Date only
        timing = f"on {start}"
        
    return f"Your next event is '{event['summary']}' {timing}"

def format_day_schedule(events_result: dict, target_date: datetime) -> str:
    """Format a day's schedule of events."""
    events = events_result.get('items', [])
    if not events:
        return f"No events found for {target_date.strftime('%Y-%m-%d')}"
        
    schedule = [f"Schedule for {target_date.strftime('%A, %B %d')}:"]
    
    for event in events:
        event_start = event['start'].get('dateTime', event['start'].get('date'))
        if 'T' in event_start:
            time_str = datetime.fromisoformat(
                event_start.replace('Z', '+00:00')
            ).strftime('%I:%M %p')
        else:
            time_str = "All day"
        schedule.append(f"- {time_str}: {event['summary']}")
        
    return "\n".join(schedule)
          
def validate_llm_response(content) -> str:
    """
    Validate and sanitize LLM response content.
    Prevents binary data contamination and ensures proper content type.
    
    Args:
        content: Raw response from LLM API
        
    Returns:
        str: Validated and sanitized text content
        
    Raises:
        ValueError: If content contains binary data or invalid formats
    """
    if isinstance(content, (bytes, bytearray)):
        raise ValueError("Binary content detected in LLM response")
        
    if isinstance(content, str):
        text = content
    elif hasattr(content, 'text'):
        text = content.text
    elif isinstance(content, list):
        text = ""
        for block in content:
            if hasattr(block, 'text'):
                text += block.text
            elif isinstance(block, dict) and block.get('type') == 'text':
                text += block.get('text', '')
            elif isinstance(block, str):
                text += block
    else:
        raise ValueError(f"Unsupported content type: {type(content)}")
        
    if any(indicator in text for indicator in [
        'LAME3', '\xFF\xFB',  # MP3 headers
        '\x89PNG',            # PNG header
        '\xFF\xD8\xFF'       # JPEG header
    ]):
        raise ValueError("Binary data detected in text content")
        
    return text.strip()

async def handle_tool_sequence(tool_response: str, system_manager, display_manager, audio_manager) -> bool:
    """
    Coordinate tool use sequence with proper state transitions and audio sync.
    
    Args:
        tool_response: Response from tool execution
        system_manager: System manager instance for validation
        display_manager: Display manager for state updates
        audio_manager: Audio manager for playback
        
    Returns:
        bool: Success status of the sequence
    """
    try:
        # Use new state: 'tool_use' for main tool execution feedback
        await display_manager.update_display('tool_use', specific_image='use')
        
        # Persona-specific sound path
        persona = config.ACTIVE_PERSONA.lower()
        audio_folder = f"/home/user/LAURA/sounds/{persona}/tool_sentences/use"
        if os.path.exists(audio_folder):
            mp3_files = [f for f in os.listdir(audio_folder) if f.endswith('.mp3')]
            if mp3_files:
                audio_file = os.path.join(audio_folder, random.choice(mp3_files))
                await audio_manager.play_audio(audio_file)
                await audio_manager.wait_for_audio_completion()
        
        # Process response
        validated_content = validate_llm_response(tool_response)
        if not validated_content:
            raise RuntimeError("Failed to process tool response content")
            
        # Generate and play TTS
        tts_audio = await system_manager.tts_handler.generate_audio(validated_content)
        if not tts_audio:
            raise RuntimeError("Failed to generate TTS audio")
            
        with open("speech.mp3", "wb") as f:
            f.write(tts_audio)
            
        await display_manager.update_display('speaking', mood='casual')
        await audio_manager.play_audio("speech.mp3")
        await audio_manager.wait_for_audio_completion()
        
        await display_manager.update_display('listening')
        return True
        
    except Exception as e:
        print(f"Error in tool sequence: {e}")
        traceback.print_exc()
        await display_manager.update_display('listening')
        return False

def get_random_audio(category: str, subtype: str = None) -> Optional[str]:
    """
    Get a random audio file from the specified category and optional subtype/context.
    Persona-aware: uses persona subdirectory for all categories.
    
    Args:
        category (str): Main audio category (e.g., 'tool', 'timeout', 'wake')
        subtype (str, optional): Subcategory, context, or specific type (e.g., 'use', 'enabled', model name)
        
    Returns:
        Optional[str]: Path to random audio file if found, None otherwise
    """
    try:
        import random
        from pathlib import Path

        persona = config.ACTIVE_PERSONA.lower()
        # Base directory for all sounds, persona-aware
        base_sound_dir = Path("/home/user/LAURA/sounds") / persona
        audio_path = None

        # Special handling for known folder structures
        if category == "file" and subtype:
            audio_path = base_sound_dir / "file_sentences" / subtype
            print(f"Looking in file category path: {audio_path}")

        elif category == "tool" and subtype:
            if subtype == "use":
                audio_path = base_sound_dir / "tool_sentences" / "use"
            elif subtype in ["enabled", "disabled"]:
                audio_path = base_sound_dir / "tool_sentences" / "status" / subtype
            else:
                audio_path = base_sound_dir / "tool_sentences" / subtype
            print(f"Looking in tool category path: {audio_path}")

        elif category == "wake" and subtype in ["Laura.pmdl", "Wake_up_Laura.pmdl", "GD_Laura.pmdl"]:
            context_map = {
                "Laura.pmdl": "standard",
                "Wake_up_Laura.pmdl": "sleepy", 
                "GD_Laura.pmdl": "frustrated"
            }
            folder = context_map.get(subtype, "standard")
            audio_path = base_sound_dir / "wake_sentences" / folder
            print(f"Looking for wake audio in: {audio_path}")

        else:
            # Default to main category folder for timeout, calibration, etc.
            audio_path = base_sound_dir / f"{category}_sentences"
            if subtype and (audio_path / subtype).exists():
                audio_path = audio_path / subtype
            print(f"Looking for audio in category folder: {audio_path}")

        # Find audio files in the specified path
        audio_files = []
        if audio_path.exists():
            audio_files = list(audio_path.glob('*.mp3')) + list(audio_path.glob('*.wav'))

        if audio_files:
            chosen_file = str(random.choice(audio_files))
            print(f"Found and selected audio file: {chosen_file}")
            return chosen_file
        else:
            print(f"WARNING: No audio files found in {audio_path}")

            # Fallback to parent directory for empty subfolders
            if subtype and f"{category}_sentences" in str(audio_path):
                parent_path = base_sound_dir / f"{category}_sentences"
                if parent_path.exists():
                    parent_files = list(parent_path.glob('*.mp3')) + list(parent_path.glob('*.wav'))
                    if parent_files:
                        print(f"Found fallback files in parent directory: {parent_path}")
                        return str(random.choice(parent_files))

            return None

    except Exception as e:
        print(f"Error in get_random_audio: {str(e)}")
        return None
