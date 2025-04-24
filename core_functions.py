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