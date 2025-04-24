#!/usr/bin/env python3

import os
import asyncio
import traceback
import re
from datetime import datetime
from pathlib import Path

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
    
    Args:
        content: Raw response from API
        chat_log: Global chat history
        system_manager: System manager instance for validation
        display_manager: Display manager for mood updates
        audio_manager: Audio manager for state checking
        notification_manager: Notification manager for queue checks
        
    Returns:
        str: Formatted message ready for voice generation
    """
    print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')}] === Response Processing Debug ===")
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Content type: {type(content)}")
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Audio state - Speaking: {audio_manager.is_speaking}, Playing: {audio_manager.is_playing}")
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Notification queue size: {notification_manager.notification_queue.qsize()}")
    
    try:
        # Validate content
        text = system_manager._validate_llm_response(content)
        formatted_message = text
        
        # Text normalization
        formatted_message = formatted_message.replace('\n', ' ')
        formatted_message = re.sub(r'(\d+)\.\s*', r'Number \1: ', formatted_message)
        formatted_message = re.sub(r'^\s*-\s*', 'Also, ', formatted_message)
        formatted_message = re.sub(r'\s+', ' ', formatted_message)
        formatted_message = re.sub(r'(?<=[.!?])\s+(?=[A-Z])', '. ', formatted_message)
        
        # Mood processing
        mood_match = re.match(r'^\[(.*?)\]([\s\S]*)', formatted_message, re.IGNORECASE)
        if mood_match:
            raw_mood = mood_match.group(1)
            clean_message = mood_match.group(2)
            mapped_mood = system_manager.map_mood(raw_mood)
            
            if mapped_mood:
                await display_manager.update_display('speaking', mood=mapped_mood)
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Mood detected: {mapped_mood}")
            
            # Save with mood intact
            chat_log.append({
                "role": "assistant",
                "content": formatted_message
            })
            
            # Strip mood for voice
            formatted_message = clean_message
        else:
            chat_log.append({
                "role": "assistant",
                "content": formatted_message
            })
        
        formatted_message = formatted_message.strip()
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Formatted for voice - Length: {len(formatted_message)}")
        
        return formatted_message
        
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
        await display_manager.update_display('tools', specific_image='use')
        
        # Queue and play pre-recorded acknowledgment
        audio_folder = f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/tool_sentences/use"
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
