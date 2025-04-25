#!/usr/bin/env python3

# =============================================================================
# Standard Library Imports - Core
# =============================================================================
import os
import io
import base64
import re
import gc
import psutil
import json
import time
import base64
import struct
import random
import asyncio
import textwrap
import threading
import requests
import traceback
import webbrowser
import select
import importlib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Dict, Tuple, List, Union, Optional, Any
from enum import Enum
from evdev import InputDevice, ecodes, list_devices

# =============================================================================
# Standard Library Imports - File Operations
# =============================================================================
import glob
import wave

# =============================================================================
# Warning Suppression
# =============================================================================
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # Hide pygame welcome message
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# Third-Party Imports - Audio Processing
# =============================================================================
import pyaudio
import whisper
import websockets
import snowboydetect
import numpy as np
from PIL import Image
from mutagen.mp3 import MP3
from elevenlabs.client import ElevenLabs
from colorama import Fore, Style

# =============================================================================
# Third-Party Imports - API Clients
# =============================================================================
from anthropic import (
    Anthropic,
    APIError,
    APIConnectionError,
    BadRequestError,
    InternalServerError
)

# =============================================================================
# Third-Party Imports - Google Services
# =============================================================================
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# =============================================================================
# Local Module Imports
# =============================================================================
from laura_tools import AVAILABLE_TOOLS as TOOLS
from tts_handler import TTSHandler
from display_manager import DisplayManager
from audio_manager_vosk import AudioManager
from whisper_transcriber import WhisperCppTranscriber
from vosk_transcriber import VoskTranscriber
from system_manager import SystemManager
from email_manager import EmailManager
from secret import (
    GOOGLE_MAPS_API_KEY,
    OPENROUTER_API_KEY,
    PV_ACCESS_KEY,
    ELEVENLABS_KEY,
    ANTHROPIC_API_KEY
)
from tool_analyzer import get_tools_for_query
from tool_context import TOOL_CONTEXTS
from token_manager import TokenManager
from document_manager import DocumentManager
from notification_manager import NotificationManager
import config
from config import (
    TRANSCRIPTION_SERVER,
    TRANSCRIPTION_MODE,
    VOSK_MODEL_PATH,
    TRANSCRIPTION_ENGINE,
    WHISPER_MODEL_PATH,
    VAD_SETTINGS,
    VOICE,
    ACTIVE_PERSONA,
    MOODS,
    MOOD_MAPPINGS,
    USE_GOOGLE,
    CONVERSATION_END_SECONDS,
    ELEVENLABS_MODEL,
    VOICE_TIMEOUT,
    VOICE_START_TIMEOUT,
    SYSTEM_PROMPT,
    SOUND_PATHS,
    WAKE_WORDS,
    CALENDAR_NOTIFICATION_INTERVALS,
    DEBUG_CALENDAR,
    CHAT_LOG_MAX_TOKENS,
    CHAT_LOG_RECOVERY_TOKENS,
    CHAT_LOG_DIR,
    SYSTEM_STATE_COMMANDS,
    TTS_ENGINE,
    ANTHROPIC_MODEL,
    CALENDAR_NOTIFICATION_SENTENCES
)
from function_definitions import (
    get_current_time,
    get_location,
    create_calendar_event,
    update_calendar_event,
    cancel_calendar_event,
    manage_tasks,
    create_task_from_email,
    create_task_for_event,
    load_recent_context,
    get_calendar_service,
    save_to_log_file,
    sanitize_messages_for_api  
)

from core_functions import get_random_audio, process_response_content
from tool_registry import tool_registry


#BASE_URL = "https://openrouter.ai/api/v1/chat/completions"  # This is for using openrouter, right now i have it configured to use anthropic for handling query
AUDIO_FILE = "speech.mp3" #gets saved-to/overwritten by elevenlabs after each completed voice generation and delivery

SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.labels",
    "https://www.googleapis.com/auth/gmail.settings.basic",
    "https://www.googleapis.com/auth/gmail.settings.sharing",
    "https://mail.google.com/",
    "https://www.googleapis.com/auth/contacts",
    "https://www.googleapis.com/auth/contacts.readonly",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/tasks"
]

# Email importance configuration - update this with information about people you care about


class CommandType(Enum):
    SYSTEM_STATE = "system_state"     # enable tools, load files, calibrate
    TOOL_USAGE = "tool_usage"         # get time, draft email, etc
    CONVERSATION = "conversation"      # regular chat

class WhisperTranscriber:
    def __init__(self, model_size="medium"):
        print(f"{Fore.YELLOW}Loading Whisper {model_size} model...{Fore.WHITE}")
        self.model = whisper.load_model(model_size)
        print(f"{Fore.GREEN}Whisper model loaded!{Fore.WHITE}")

    def transcribe(self, audio_data, sample_rate=16000):
        """
        Transcribe audio data to text.
        audio_data: List of PCM data
        sample_rate: Sample rate of the audio (default 16000 for Whisper)
        """
        temp_file = "temp_audio.wav"
        try:
            # Convert PCM to WAV
            with wave.open(temp_file, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(np.array(audio_data).tobytes())

            # Transcribe
            result = self.model.transcribe(
                temp_file,
                language="en",
                fp16=True,
                initial_prompt="This is a voice command for a smart assistant"
            )

            return result["text"].strip()

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def cleanup(self):
        # Add any cleanup code here if needed
        pass

class RemoteTranscriber:
    def __init__(self):
        self.server_url = f"ws://{TRANSCRIPTION_SERVER['host']}:{TRANSCRIPTION_SERVER['port']}"
        self.websocket = None
        self.max_retries = 3
        self.retry_delay = 2
        self.send_timeout = 60
        self.receive_timeout = 120
        self.ping_interval = 20
        self.ping_timeout = 30

    async def check_connection(self):
        try:
            if not self.websocket:
                return False
            pong_waiter = await self.websocket.ping()
            await asyncio.wait_for(pong_waiter, timeout=10)
            return True
        except:
            return False

    async def connect(self):
        retries = 0
        while retries < self.max_retries:
            try:
                if not self.websocket:
                    self.websocket = await websockets.connect(
                        self.server_url,
                        max_size=100 * 1024 * 1024,
                        ping_interval=self.ping_interval,
                        ping_timeout=self.ping_timeout
                    )
                return True
            except Exception as e:
                print(f"Connection attempt {retries + 1} failed: {e}")
                retries += 1
                if retries < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
        return False

    async def transcribe(self, audio_data):
        try:
            # Check and reconnect if needed
            if self.websocket and not await self.check_connection():
                await self.cleanup()
                await self.connect()
            elif not self.websocket and not await self.connect():
                return None

            # Get audio length
            audio_length = len(audio_data) / 160001

            # NEW: For long audio, process immediately in chunks
            if audio_length > 15.0:
                print(f"Processing {audio_length:.2f}s audio in parallel chunks")
                chunk_size = 16000 * 10  # 10 second chunks
                overlap = 16000 * 1      # 1 second overlap

                # Create tasks for all chunks
                tasks = []
                chunk_id = 0

                for start in range(0, len(audio_data), chunk_size - overlap):
                    chunk_id += 1
                    end = min(start + chunk_size, len(audio_data))
                    chunk = audio_data[start:end]

                    # Skip chunks that are too short
                    if len(chunk) < 16000 * 2:  # At least 2 seconds
                        continue

                    # Process each chunk as a separate task
                    task = asyncio.create_task(
                        self._process_chunk(chunk, chunk_id, chunk_id == 1)
                    )
                    tasks.append(task)

                # Wait for all chunks to complete
                results = await asyncio.gather(*tasks)

                # Filter out None results and combine
                valid_results = [r for r in results if r]
                if valid_results:
                    combined_text = " ".join(valid_results)
                    print(f"Combined transcription: {combined_text}")
                    return combined_text
                else:
                    return None

            else:
                # For shorter audio, process normally
                return await self._process_chunk(audio_data, 1, True)

        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed unexpectedly: {e}")
            self.websocket = None
            return None
        except Exception as e:
            print(f"Transcription error: {type(e).__name__}: {e}")
            if self.websocket:
                await self.cleanup()
            return None

    async def _process_chunk(self, audio_chunk, chunk_id=1, is_first_chunk=True):
        """Process a single audio chunk"""
        try:
            chunk_length = len(audio_chunk) / 16000
            print(f"Processing chunk {chunk_id}: {chunk_length:.2f}s")

            # Send chunk with timeout
            await asyncio.wait_for(
                self.websocket.send(json.dumps({
                    "audio": audio_chunk.tolist(),
                    "is_chunk": True,
                    "chunk_id": chunk_id,
                    "is_first_chunk": is_first_chunk
                })),
                timeout=self.send_timeout
            )

            # Get response with timeout
            response = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=self.receive_timeout
            )

            result = json.loads(response)
            if "error" in result:
                print(f"Server error: {result['error']}")
                return None

            transcript = result["transcript"]

            # Check for end phrases
            if transcript:
                normalized_text = transcript.lower().strip()
                end_phrases = [
                    "thank you for watching",
                    "thanks for watching",
                    "thank you watching",
                    "thanks watching",
                ]

                for phrase in end_phrases:
                    if phrase in normalized_text.replace('!', '').replace('.', '').replace(',', ''):
                        print(f"End phrase detected: {transcript}")
                        return None

            if transcript and transcript.strip():
                print(f"Chunk {chunk_id} result: {transcript}")
                return transcript
            return None

        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")
            return None

        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed unexpectedly: {e}")
            self.websocket = None
            return None
        except Exception as e:
            print(f"Transcription error: {type(e).__name__}: {e}")
            if self.websocket:
                await self.cleanup()
            return None

    async def cleanup(self):
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                print(f"Error during cleanup: {e}")
            finally:
                self.websocket = None


# Global variables
# =============================================================================
# Core System Components - Global Declarations
# =============================================================================
# Initialize all global variables at module level
keyboard_device: Optional[Any] = None
keyboard_path: Optional[str] = None
transcriber: Optional[Union[VoskTranscriber, WhisperCppTranscriber]] = None
remote_transcriber: Optional[RemoteTranscriber] = None
audio_manager: Optional[AudioManager] = None
display_manager: Optional[DisplayManager] = None
document_manager: Optional[DocumentManager] = None
tts_handler: Optional[TTSHandler] = None
anthropic_client: Optional[Anthropic] = None
token_manager: Optional[TokenManager] = None
system_manager: Optional[SystemManager] = None
email_manager: Optional[EmailManager] = None  # Add this line
snowboy = None  # Type hint not added as snowboydetect types aren't standard
system_manager_lock = asyncio.Lock()


# =============================================================================
# Google Integration Setup  - DO NOT FUCK WITH THIS AT ALL!!!  THIS SETS UP THE CHROMIUM BROWSER TO PROVIDE WITH A REFREST TOKEN AFTER YOU LOG IN, YOU SHOULD NOT HAVE A BROWSER POP-UP WHEN RELOGGING IN.
# =============================================================================
creds = None  # Global declaration
try:
    webbrowser.register('chromium', None, webbrowser.Chrome('/usr/bin/chromium'))

    if USE_GOOGLE:
        if os.path.exists("token.json"):
            try:
                creds = Credentials.from_authorized_user_file("token.json", SCOPES)  #This is your refresh token.  As long as you don't switch wifi networks the refresh token should log you back in automatically if you need to restart the script, or the device
                print("Loaded existing credentials from token.json")
            except Exception as e:
                print(f"Error loading credentials: {e}")
                if os.path.exists("token.json"):
                    os.remove("token.json")
                creds = None

        if not creds or not creds.valid:
            try:
                if creds and creds.expired and creds.refresh_token:
                    print("Refreshing expired credentials")
                    creds.refresh(Request())
                else:
                    print("Initiating new OAuth2 flow")
                    # Create a new InstalledAppFlow with custom OAuth2 session
                    from google_auth_oauthlib.flow import InstalledAppFlow
                    from requests_oauthlib import OAuth2Session

                    flow = InstalledAppFlow.from_client_secrets_file(
                        "credentials.json",
                        scopes=SCOPES,
                    )

                    # Configure the session with offline access
                    session = OAuth2Session(
                        client_id=flow.client_config['client_id'],
                        scope=SCOPES,
                        redirect_uri='http://localhost:8080/'
                    )

                    # Update flow's session with our configured one
                    flow.oauth2session = session

                    # Generate authorization URL with offline access
                    auth_url, _ = flow.authorization_url(
                        access_type='offline',
                        include_granted_scopes='true',
                        prompt='consent'  # Force consent screen
                    )

                    print("Opening browser for authentication...")
                    creds = flow.run_local_server(
                        host='localhost',
                        port=8080,
                        access_type='offline',
                        prompt='consent',
                        authorization_prompt_message="Please complete authentication in the opened browser window",
                        success_message="Authentication completed successfully. You may close this window."
                    )

                    print("\nDetailed Token Information:")
                    print(f"Valid: {creds.valid}")
                    print(f"Expired: {creds.expired}")
                    print(f"Has refresh_token attribute: {hasattr(creds, 'refresh_token')}")
                    if hasattr(creds, 'refresh_token'):
                        print(f"Refresh token value present: {bool(creds.refresh_token)}")
                        print(f"Token expiry: {creds.expiry}")

                if creds and creds.valid:
                    if not hasattr(creds, 'refresh_token') or not creds.refresh_token:
                        print("\nWARNING: No refresh token received!")
                        print("This might require re-authentication on next run.")
                        print("Try revoking access at https://myaccount.google.com/permissions")
                        print("Then delete token.json and run again.")
                    else:
                        print("\nSaving credentials with refresh token to token.json")
                        with open("token.json", "w") as token:
                            token.write(creds.to_json())
                        print("Credentials saved successfully")
                else:
                    print("Warning: Invalid credentials state")

            except Exception as e:
                print(f"Error during Google authentication: {e}")
                traceback.print_exc()
                if os.path.exists("token.json"):
                    os.remove("token.json")
                creds = None

except Exception as e:
    print(f"Error setting up Google integration: {e}")
    traceback.print_exc()
    USE_GOOGLE = False
# =============================================================================
# Manager Initialization
# =============================================================================
try:
    # Initialize notification manager first
    notification_manager = NotificationManager(audio_manager)
    
    # Initialize email manager with Google credentials
    email_manager = EmailManager(creds) if USE_GOOGLE else None
    
    # Initialize document manager
    document_manager = DocumentManager()  # Document manager no longer needs email_manager
    
    # Now initialize system manager with all available components
    async def init_system_manager():
        global system_manager
        async with system_manager_lock:
            system_manager = SystemManager(
                email_manager=email_manager,
                display_manager=display_manager,
                audio_manager=audio_manager,
                document_manager=document_manager,
                notification_manager=notification_manager,
                token_manager=token_manager,
                tts_handler=tts_handler,
                anthropic_client=anthropic_client
            )
        return system_manager

    # Run system manager initialization
    asyncio.get_event_loop().run_until_complete(init_system_manager())
    print("Manager initialization completed successfully")
    
except Exception as e:
    print(f"Error during manager initialization: {e}")
    traceback.print_exc()
    raise

# =============================================================================
# Core Component Initialization
# =============================================================================
try:
    if TRANSCRIPTION_MODE == "local":
        if TRANSCRIPTION_ENGINE == "vosk":
            transcriber = VoskTranscriber(VOSK_MODEL_PATH)
        else:
            transcriber = WhisperCppTranscriber(WHISPER_MODEL_PATH, VAD_SETTINGS)
    else:
        remote_transcriber = RemoteTranscriber()

    audio_manager = AudioManager(PV_ACCESS_KEY if TRANSCRIPTION_MODE == "remote" else None)
    tts_config = {
    "TTS_ENGINE": config.TTS_ENGINE,
    "ELEVENLABS_KEY": ELEVENLABS_KEY,
    "VOICE": config.VOICE,
    "ELEVENLABS_MODEL": ELEVENLABS_MODEL,
}
    tts_handler = TTSHandler(tts_config)
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
except Exception as e:
    print(f"Critical error during core component initialization: {e}")
    raise

# Notification tracking structures
NOTIFICATION_STATES = {
    "calendar": {
        # event_id: {last_notified: datetime, count: int, intervals: set(), recurring: bool}
    },
    "configurable": {
        # notification_id: {last_notified: datetime, cleared: bool, reminder_interval: int}
    }
}

# Track notification clear states
PENDING_NOTIFICATIONS = {
    # notification_id: {
    #     "type": str ("calendar" or "configurable"),
    #     "message": str,
    #     "created": datetime,
    #     "requires_clear": bool,
    #     "last_reminder": datetime,
    #     "reminder_interval": int (minutes)
    # }
}
# =============================================================================
# State Management
# =============================================================================
chat_log = []
last_interaction = datetime.now()
last_interaction_check = datetime.now()
initial_startup = True
calendar_notified_events = set()
notification_queue = asyncio.Queue()  # Queue for pending notifications during conversation

# =============================================================================
# Session Management
# =============================================================================
LAST_TASKS_RESULT = {
    "tasks": [],
    "timestamp": None
}


# =============================================================================
# Functions
# =============================================================================

def read_keyboard_events(device, max_events=5):
    """
    Read keyboard events in a non-blocking way with proper error handling.

    Args:
        device: InputDevice instance
        max_events: Maximum number of events to read at once

    Returns:
        list: Events read from device or empty list if none/error
    """
    if not device:
        return []

    events = []
    try:
        # Use select with a very short timeout for non-blocking reads
        r, w, x = select.select([device.fd], [], [], 0)

        if not r:
            return []  # No data available

        # Read available events (up to max_events)
        for i in range(max_events):
            try:
                event = device.read_one()
                if event is None:
                    break  # No more events
                events.append(event)
            except BlockingIOError:
                break  # No more events available

    except (OSError, IOError) as e:
        print(f"Error reading keyboard: {e}")

    return events
def map_mood(mood):
    return MOOD_MAPPINGS.get(mood.lower(), "casual")

def verify_token_manager_setup():
    """Verify token tracker initialization status"""
    print(f"\n{Fore.CYAN}=== Pre-Initialization Check ===")
    print(f"Token Tracker Status: {'Defined' if 'token_manager' in globals() else 'Missing'}")
    print(f"Anthropic Client Status: {'Initialized' if anthropic_client else 'Missing'}")
    print(f"System Prompt Status: {'Defined' if SYSTEM_PROMPT else 'Missing'}{Fore.WHITE}\n")

async def handle_system_command(transcript):
    """
    Handle system-level commands using SystemManager
    Returns True if command was handled, False otherwise
    """
    try:
        # Check if command matches any system patterns
        is_command, command_type, action, arguments = system_manager.detect_command(transcript)

        if is_command:
            # Execute the command through system manager
            # This will handle:
            # - Document loading/unloading (showing document states from /system/document/{load,unload})
            # - Calibration (showing calibration image from /system/calibration/)
            # - Persona switching (with transition animations from /system/persona/out and /in)
            # - Tool enabling/disabling
            success = await system_manager.handle_command(command_type, action, arguments)
            return success

        return False

    except Exception as e:
        print(f"Error handling system command: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

async def execute_tool(tool_call):
    """Execute a tool and return its result."""
    try:
        # Get the tool handler from TokenManager
        handler = token_manager.tool_handlers.get(tool_call.name)
        if not handler:
            return "Unsupported tool called"
            
        tool_args = tool_call.input
        
        # Special handling for calendar query
        if tool_call.name == "calendar_query":
            if tool_args["query_type"] == "next_event":
                return await handler(get_next_event)
            elif tool_args["query_type"] == "day_schedule":
                return await handler(get_day_schedule)
            else:
                return "Unsupported query type"
        
        # Handle update_calendar_event and cancel_calendar_event special cases
        if tool_call.name in ["update_calendar_event", "cancel_calendar_event"]:
            if "event_description" in tool_args and not tool_args.get("event_id"):
                matching_events = find_calendar_event(tool_args["event_description"])
                if matching_events:
                    if len(matching_events) == 1:
                        tool_args["event_id"] = matching_events[0]["id"]
                        tool_args.pop("event_description", None)
                        return await handler(**tool_args)
                    else:
                        event_list = "\n".join([
                            f"{i+1}. {event['summary']} ({event['start_formatted']})"
                            for i, event in enumerate(matching_events)
                        ])
                        return f"I found multiple matching events:\n\n{event_list}\n\nPlease specify which event you'd like to {'update' if tool_call.name == 'update_calendar_event' else 'cancel'}."
                else:
                    return "I couldn't find any calendar events matching that description."
        
        # Execute the tool with its arguments
        result = await handler(**tool_args) if asyncio.iscoroutinefunction(handler) else handler(**tool_args)
        
        # Record successful tool usage
        token_manager.record_tool_usage(tool_call.name)
        
        return result
        
    except Exception as e:
        print(f"DEBUG: Tool execution error: {e}")
        return f"Error executing tool: {str(e)}"

async def generate_response(query: str) -> str:
    """
    Generate a response using the LLM with proper tool coordination and state management.
    
    Args:
        query: User's input query
        
    Returns:
        str: Processed response ready for TTS
    """
    global chat_log, last_interaction, last_interaction_check

    system_manager = tool_registry.get_manager('system')
    if not system_manager:
        raise RuntimeError("System manager not registered with tool registry")
        
    display_manager = tool_registry.get_manager('display')
    audio_manager = tool_registry.get_manager('audio')
    token_manager = tool_registry.get_manager('token')
    notification_manager = tool_registry.get_manager('notification')

    now = datetime.now()
    last_interaction = now
    last_interaction_check = now
    token_manager.start_interaction()

    try:
        was_command, command_response = token_manager.handle_tool_command(query)
        if was_command:
            success = command_response.get('success', False)
            mood = command_response.get('mood', 'casual')
            await display_manager.update_display('speaking', mood=mood if success else 'excited')
            if command_response.get('state') in ['enabled', 'disabled']:
                status_type = command_response['state'].lower()
                audio_file = get_random_audio('tool', f'status/{status_type}')
                if audio_file:
                    await display_manager.update_display('tools', specific_image=status_type)
                    await audio_manager.queue_audio(audio_file=audio_file, priority=True)
                    await audio_manager.wait_for_queue_empty()
            return "[CONTINUE]"

        # Save user message
        chat_message = {"role": "user", "content": query}
        try:
            save_to_log_file(chat_message)
        except Exception as e:
            print(f"Warning: Failed to save chat message to log: {e}")
            traceback.print_exc()
        
        already_added = (
            len(chat_log) > 0 and 
            chat_log[-1]["role"] == "user" and
            chat_log[-1]["content"] == query
        )
        if not already_added:
            chat_log.append(chat_message)

        use_tools = token_manager.tools_are_active()
        relevant_tools = []
        if use_tools:
            tools_needed, relevant_tools = token_manager.get_tools_for_query(query)
            if tools_needed and relevant_tools:
                query_lower = query.lower()
                tool_to_execute = None
                tool_args = {}
                if any(t['name'] == 'get_current_time' for t in relevant_tools) and ('time' in query_lower or 'clock' in query_lower):
                    tool_to_execute = 'get_current_time'
                    format_type = 'both'
                    if 'date' in query_lower and 'time' not in query_lower:
                        format_type = 'date'
                    elif 'time' in query_lower and 'date' not in query_lower:
                        format_type = 'time'
                    tool_args = {"format": format_type}
                if tool_to_execute:
                    from collections import namedtuple
                    ToolCall = namedtuple('ToolCall', ['name', 'id', 'input'])
                    tool_call = ToolCall(name=tool_to_execute, id="direct-execution", input=tool_args)
                    return await system_manager.execute_tool_with_feedback(tool_call)

        # Prepare sanitized messages for API
        sanitized_messages = sanitize_messages_for_api(chat_log)
        system_content = config.SYSTEM_PROMPT

        document_manager = tool_registry.get_manager('document')
        if document_manager and document_manager.files_loaded:
            loaded_content = document_manager.get_loaded_content()
            if loaded_content["system_content"]:
                system_content += "\n\nLoaded Documents:\n" + loaded_content["system_content"]

        if use_tools and relevant_tools:
            system_content += "\n\nAvailable Tools:\n" + json.dumps(relevant_tools, indent=2)

        if document_manager and document_manager.files_loaded:
            loaded_content = document_manager.get_loaded_content()
            if loaded_content["image_content"]:
                api_message_content = loaded_content["image_content"]
                api_message_content.append({"type": "text", "text": query})
                if sanitized_messages and sanitized_messages[-1]["role"] == "user":
                    sanitized_messages[-1]["content"] = api_message_content
                else:
                    sanitized_messages.append({
                        "role": "user",
                        "content": api_message_content
                    })
            else:
                if sanitized_messages and sanitized_messages[-1]["role"] == "user":
                    sanitized_messages[-1]["content"] = query
                else:
                    sanitized_messages.append({"role": "user", "content": query})
        else:
            if sanitized_messages and sanitized_messages[-1]["role"] == "user":
                sanitized_messages[-1]["content"] = query
            else:
                sanitized_messages.append({"role": "user", "content": query})

        api_params = {
            "model": config.ANTHROPIC_MODEL,
            "system": system_content,
            "messages": sanitized_messages,
            "max_tokens": 1024,
            "temperature": 0.8
        }
        if use_tools and relevant_tools:
            api_params["tools"] = relevant_tools
            api_params["tool_choice"] = {"type": "auto"}

        # Main API call
        response = system_manager.anthropic_client.messages.create(**api_params)
        if not response or not response.content:
            raise ValueError("Empty response from API")

        # Handle tool use
        if response.stop_reason == "tool_use":
            # 1. Append initial assistant response (acknowledgment) to chat_log and log file
            initial_assistant_msg = None
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    initial_assistant_msg = getattr(block, "text", "")
                    break
            if not initial_assistant_msg:
                initial_assistant_msg = "Processing your request..."

            assistant_msg = {"role": "assistant", "content": initial_assistant_msg}
            chat_log.append(assistant_msg)
            try:
                save_to_log_file(assistant_msg)
            except Exception as e:
                print(f"Warning: Failed to save assistant message to log: {e}")

            # 2. Find the tool_call
            tool_call = next((block for block in response.content if getattr(block, "type", None) == "tool_use"), None)
            if not tool_call:
                raise ValueError("Tool use indicated but no tool call found")

            # 3. Execute tool and append tool_result as user message
            tool_result = await system_manager.execute_tool_with_feedback(tool_call)
            if tool_result:
                tool_result_msg = {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": tool_result["result"] if isinstance(tool_result, dict) and "result" in tool_result else tool_result,
                    }]
                }
                chat_log.append(tool_result_msg)
                try:
                    save_to_log_file(tool_result_msg)
                except Exception as e:
                    print(f"Warning: Failed to save tool_result to log: {e}")

                # 4. Persona-aware pre-recorded audio (tool_sentences/use)
                current_persona = getattr(config, "ACTIVE_PERSONA", "laura").lower()
                audio_folder = f"/home/user/LAURA/sounds/{current_persona}/tool_sentences/use"
                tool_sound = None
                if os.path.exists(audio_folder):
                    mp3_files = [f for f in os.listdir(audio_folder) if f.endswith('.mp3')]
                    if mp3_files:
                        tool_sound = os.path.join(audio_folder, random.choice(mp3_files))
                if tool_sound:
                    await audio_manager.queue_audio(audio_file=tool_sound, priority=True)
                    await audio_manager.wait_for_queue_empty()

                # 5. Secondary API call with updated chat log
                final_response = system_manager.anthropic_client.messages.create(
                    model=config.ANTHROPIC_MODEL,
                    system=system_content,
                    messages=sanitize_messages_for_api(chat_log),
                    max_tokens=1024,
                    temperature=0.7
                )

                # 6. Process and TTS the final response
                if final_response and final_response.content:
                    processed_content = await process_response_content(
                        final_response.content, chat_log, system_manager, display_manager, audio_manager, notification_manager
                    )
                    try:
                        final_audio = system_manager.tts_handler.generate_audio(str(processed_content))
                        with open(AUDIO_FILE, "wb") as f:
                            f.write(final_audio)
                        await display_manager.update_display('speaking', mood='casual')
                        await audio_manager.queue_audio(audio_file=AUDIO_FILE)
                        await audio_manager.wait_for_queue_empty()
                        return processed_content
                    except Exception as e:
                        print(f"Error in final response audio generation: {e}")
                        raise

        # Non-tool flow
        processed_content = await process_response_content(
            response.content, chat_log, system_manager, display_manager, audio_manager, notification_manager
        )
        final_audio = system_manager.tts_handler.generate_audio(str(processed_content))
        with open(AUDIO_FILE, "wb") as f:
            f.write(final_audio)
        await display_manager.update_display('speaking', mood='casual')
        await audio_manager.queue_audio(audio_file=AUDIO_FILE)
        await audio_manager.wait_for_queue_empty()
        return processed_content

    except Exception as e:
        print(f"Error in generate_response: {e}")
        traceback.print_exc()
        try:
            await display_manager.update_display('speaking', mood='disappointed')
            error_msg = "I apologize, but I encountered an error processing your request"
            error_audio = system_manager.tts_handler.generate_audio(error_msg)
            with open(AUDIO_FILE, "wb") as f:
                f.write(error_audio)
            await audio_manager.queue_audio(audio_file=AUDIO_FILE)
            await audio_manager.wait_for_queue_empty()
            return error_msg
        except Exception as audio_err:
            print(f"Error handling error notification: {audio_err}")
            return "An unexpected error occurred"
                        
async def wake_word():
    """Wake word detection with notification-aware breaks"""
    global last_interaction_check, last_interaction

    # One-time initialization
    if not hasattr(wake_word, 'detector'):
        try:
            print(f"{Fore.YELLOW}Initializing wake word detector...{Fore.WHITE}")

            resource_path = Path("resources/common.res")
            model_paths = [
                Path("GD_Laura.pmdl"),
                Path("Wake_up_Laura.pmdl"),
                Path("Laura.pmdl")
            ]

            for path in [resource_path] + model_paths:
                if not path.exists():
                    print(f"ERROR: File not found at {path.absolute()}")
                    return None

            wake_word.detector = snowboydetect.SnowboyDetect(
                resource_filename=str(resource_path.absolute()).encode(),
                model_str=",".join(str(p.absolute()) for p in model_paths).encode()
            )
            wake_word.detector.SetSensitivity(b"0.5,0.5,0.45")
            wake_word.model_names = [p.name for p in model_paths]
            wake_word.pa = pyaudio.PyAudio()
            wake_word.stream = None
            wake_word.last_break = time.time()
            print(f"{Fore.GREEN}Wake word detector initialized{Fore.WHITE}")
        except Exception as e:
            print(f"Error initializing wake word detection: {e}")
            return None

    try:
        # Create/restart stream if needed
        if not wake_word.stream or not wake_word.stream.is_active():
            wake_word.stream = wake_word.pa.open(
                rate=16000,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=2048
            )
            wake_word.stream.start_stream()

        # Periodic breaks for notifications (every 20 seconds)
        current_time = time.time()
        if (current_time - wake_word.last_break) >= 20:
            wake_word.last_break = current_time
            if wake_word.stream:
                wake_word.stream.stop_stream()
                await asyncio.sleep(1)  # 1-second break
                wake_word.stream.start_stream()
            return None

        # Read audio with error handling
        try:
            data = wake_word.stream.read(2048, exception_on_overflow=False)
            if len(data) == 0:
                print("Warning: Empty audio frame received")
                return None
        except (IOError, OSError) as e:
            print(f"Stream read error: {e}")
            if wake_word.stream:
                wake_word.stream.stop_stream()
                wake_word.stream.close()
                wake_word.stream = None
            return None

        result = wake_word.detector.RunDetection(data)
        if result > 0:
            print(f"{Fore.GREEN}Wake word detected! (Model {result}){Fore.WHITE}")
            last_interaction = datetime.now()
            last_interaction_check = datetime.now()
            return wake_word.model_names[result-1] if result <= len(wake_word.model_names) else None

        # Occasionally yield to event loop (much less frequently)
        if random.random() < 0.01:
            await asyncio.sleep(0)

        return None

    except Exception as e:
        print(f"Error in wake word detection: {e}")
        traceback.print_exc()
        if hasattr(wake_word, 'stream') and wake_word.stream:
            wake_word.stream.stop_stream()
            wake_word.stream.close()
            wake_word.stream = None
        return None



def has_conversation_hook(response):
    """
    Default to True for conversation continuation unless explicitly ended.
    Only returns False if specific end-conversation phrases are detected.
    """
    if response is None:
        print("Warning: Received None response in has_conversation_hook")
        return False

    if isinstance(response, str):
        # Check for control signals first
        if response.startswith("[CONTINUE]"):
            return True

        # End conversation phrases
        end_phrases = [
            "no need for follow up",
            "i'm done for now",
            "that will be all",
            "no further questions needed",
            "end of conversation"
        ]

        return not any(phrase in response.lower() for phrase in end_phrases)

    return True

async def handle_conversation_loop(initial_response):
    """Handle ongoing conversation with calendar notification interrupts"""
    global chat_log, last_interaction

    res = initial_response
    while has_conversation_hook(res):
        # Now enter listening state for follow-up
        await display_manager.update_display('listening')

        # Capture the follow-up speech
        follow_up = await capture_speech(is_follow_up=True)

        if follow_up == "[CONTINUE]":
            continue

        elif follow_up:
            # Update last interaction time
            last_interaction = datetime.now()

            # Check if this is a system command
            cmd_result = system_manager.detect_command(follow_up)
            if cmd_result and cmd_result[0]:
                is_cmd, cmd_type, action, args = cmd_result  # Now unpacking 4 values
                success = await system_manager.handle_command(cmd_type, action, args)

                if success:
                    await display_manager.update_display('listening')
                    continue
                else:
                    # Clear conversation hook by setting res to empty
                    res = ""
                    await display_manager.update_display('idle')
                    print(f"{Fore.MAGENTA}Conversation ended, returning to idle state...{Fore.WHITE}")
                    break

            # Generate response for follow-up
            await display_manager.update_display('thinking')
            res = await generate_response(follow_up)

            if res == "[CONTINUE]":
                print("DEBUG - Detected [CONTINUE] control signal, skipping voice generation")
                continue

            await display_manager.update_display('speaking')
            await generate_voice(res)

            if not has_conversation_hook(res):
                await audio_manager.wait_for_audio_completion()
                await display_manager.update_display('idle')
                print(f"{Fore.MAGENTA}Conversation ended, returning to idle state...{Fore.WHITE}")
                break

            await audio_manager.wait_for_audio_completion()
        else:
            # No follow-up detected or manual stop, clear conversation hook
            res = ""  # Clear the conversation hook
            timeout_audio = get_random_audio("timeout")
            if timeout_audio:
                await audio_manager.play_audio(timeout_audio)
            else:
                await generate_voice("No input detected. Feel free to ask for assistance when needed")
            await audio_manager.wait_for_audio_completion()
            await display_manager.update_display('idle')
            print(f"{Fore.MAGENTA}Conversation ended due to timeout or manual stop, returning to idle state...{Fore.WHITE}")
            break

async def check_manual_stop():
    if keyboard_device:
        try:
            r, w, x = select.select([keyboard_device.fd], [], [], 0)
            if r:
                events = read_keyboard_events(keyboard_device)
                for event in events:
                    if event.type == ecodes.EV_KEY and event.code == 125 and event.value == 1:
                        print(f"{Fore.GREEN}Manual recording termination via LEFTMETA{Fore.WHITE}")
                        return True
        except Exception as e:
            print(f"Keyboard check error in recording: {e}")
    return False

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


async def print_response(chat):
    """Print response before voice generation"""
    wrapper = textwrap.TextWrapper(width=70)
    paragraphs = chat.split('\n')
    wrapped_chat = "\n".join([wrapper.fill(p) for p in paragraphs if p.strip()])
    print(wrapped_chat)

async def generate_voice(chat):
    """
    Generate voice audio from formatted text.
    """
    print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')}] === Voice Generation Debug ===")
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Input type: {type(chat)}")
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Input preview: {str(chat)[:100]}")
    
    # Skip voice generation for control signals
    if not chat or chat == "[CONTINUE]":
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Skipping voice generation - control signal")
        return

    try:
        # Normalize the text
        chat = ' '.join(chat.split())
        
        if not chat.strip():
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Warning: Empty text after normalization")
            return

        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Generating TTS for: {chat[:100]}...")

        # Generate audio with error handling
        try:
            audio = system_manager.tts_handler.generate_audio(chat)
            if not audio:
                raise Exception("TTS handler returned empty audio data")
        except Exception as tts_error:
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Error in TTS generation: {tts_error}")
            raise

        # Save and play the audio
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Saving audio to {AUDIO_FILE}")
        with open(AUDIO_FILE, "wb") as f:
            f.write(audio)

        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Queueing audio for playback")
        await audio_manager.play_audio(AUDIO_FILE)

    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Error in generate_voice: {e}")
        traceback.print_exc()



async def play_queued_notifications():
    """
    Play queued notifications and handle notification state management.
    Supports both calendar events and configurable notifications.
    """
    global NOTIFICATION_STATES, PENDING_NOTIFICATIONS

    if notification_queue.empty() and not PENDING_NOTIFICATIONS:
        return

    previous_state = display_manager.current_state
    previous_mood = display_manager.current_mood
    current_time = datetime.now()

    # First handle any immediate notifications in the queue
    while not notification_queue.empty():
        notification_data = await notification_queue.get()

        # Expected format: dict with type, id, message, requires_clear, reminder_interval
        notification_type = notification_data.get('type', 'calendar')
        notification_id = notification_data.get('id')
        message = notification_data.get('message', '')
        requires_clear = notification_data.get('requires_clear', False)
        reminder_interval = notification_data.get('reminder_interval', 10)  # Default 10 minutes

        # Add to pending if requires clear
        if requires_clear:
            PENDING_NOTIFICATIONS[notification_id] = {
                "type": notification_type,
                "message": message,
                "created": current_time,
                "requires_clear": True,
                "last_reminder": current_time,
                "reminder_interval": reminder_interval
            }

        try:
            await audio_manager.wait_for_audio_completion()
            await display_manager.update_display('speaking', mood='casual')

            audio = tts_handler.generate_audio(message)
            with open("notification.mp3", "wb") as f:
                f.write(audio)

            await audio_manager.play_audio("notification.mp3")
            await audio_manager.wait_for_audio_completion()

        except Exception as e:
            print(f"Error playing notification: {e}")

    # Handle pending notifications that need reminders
    pending_to_remove = set()
    for notification_id, notification_data in PENDING_NOTIFICATIONS.items():
        if not notification_data['requires_clear']:
            pending_to_remove.add(notification_id)
            continue

        time_since_last = (current_time - notification_data['last_reminder']).total_seconds() / 60
        if time_since_last >= notification_data['reminder_interval']:
            reminder_message = f"Reminder: {notification_data['message']}"

            try:
                await audio_manager.wait_for_audio_completion()
                await display_manager.update_display('speaking', mood='casual')

                audio = tts_handler.generate_audio(reminder_message)
                with open("notification.mp3", "wb") as f:
                    f.write(audio)

                await audio_manager.play_audio("notification.mp3")
                await audio_manager.wait_for_audio_completion()

                # Update last reminder time
                PENDING_NOTIFICATIONS[notification_id]['last_reminder'] = current_time

            except Exception as e:
                print(f"Error playing reminder: {e}")

    # Remove cleared notifications
    for notification_id in pending_to_remove:
        PENDING_NOTIFICATIONS.pop(notification_id, None)

    # Restore previous display state
    await display_manager.update_display(previous_state, mood=previous_mood)

async def check_upcoming_events():
    """Calendar notification system that can interrupt conversation flow"""
    global calendar_notified_events, notification_queue

    while True:
        try:
            service = get_calendar_service()
            if not service:
                await asyncio.sleep(30)
                continue

            now = datetime.now(timezone.utc)
            max_minutes = max(CALENDAR_NOTIFICATION_INTERVALS) + 1
            timeMin = now.isoformat()
            timeMax = (now + timedelta(minutes=max_minutes)).isoformat()

            events_result = service.events().list(
                calendarId='primary',
                timeMin=timeMin,
                timeMax=timeMax,
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            events = events_result.get('items', [])

            # Check for notifications regardless of state
            for event in events:
                event_id = event['id']
                summary = event.get('summary', 'Unnamed event')
                start = event['start'].get('dateTime', event['start'].get('date'))

                if 'T' in start:
                    start_time = datetime.fromisoformat(start.replace('Z', '+00:00'))
                else:
                    start_time = datetime.strptime(start, '%Y-%m-%d').replace(tzinfo=timezone.utc)

                seconds_until = (start_time - now).total_seconds()
                minutes_until = int(seconds_until / 60)

                for interval in CALENDAR_NOTIFICATION_INTERVALS:
                    notification_key = f"{event_id}_{interval}"

                    if (abs(minutes_until * 60 - interval * 60) <= 15 and
                        notification_key not in calendar_notified_events):
                        calendar_notified_events.add(notification_key)

                        notification_text = random.choice(CALENDAR_NOTIFICATION_SENTENCES).format(
                            minutes=interval,
                            event=summary
                        )

                        # Add to notification queue instead of pending_notifications list
                        await notification_queue.put(notification_text)
                        print(f"DEBUG: Added calendar notification to queue: {notification_text[:50]}...")

                # Cleanup old notifications
                if minutes_until < 0:
                    for cleanup_interval in CALENDAR_NOTIFICATION_INTERVALS:
                        calendar_notified_events.discard(f"{event_id}_{cleanup_interval}")

            await asyncio.sleep(30)

        except Exception as e:
            print(f"Error in calendar check: {e}")
            await asyncio.sleep(30)

async def heartbeat(remote_transcriber):
    while True:
        try:
            if remote_transcriber.websocket:
                await remote_transcriber.websocket.ping()
            await asyncio.sleep(30)  # Check every 30 seconds
        except:
            remote_transcriber.websocket = None

async def get_random_audio_async(category, context=None):
    """Asynchronous wrapper for get_random_audio"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_random_audio, category, context)

def sanitize_tool_interactions(chat_history):
    """
    Fix mismatched tool_use and tool_result pairs in chat history.
    """
    print("DEBUG: Sanitizing tool interactions in chat history")

    # STEP 1: Build comprehensive tool_use ID mapping
    tool_use_info = {}  # Maps tool_use ID to (message_index, item_index)

    for msg_idx, message in enumerate(chat_history):
        if message.get("role") == "assistant" and isinstance(message.get("content"), list):
            for item_idx, item in enumerate(message.get("content", [])):
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    tool_id = item.get("id")
                    if tool_id:
                        tool_use_info[tool_id] = (msg_idx, item_idx)

    # STEP 2: Check and fix tool_result references
    valid_history = []

    for message in chat_history:
        # Handle non-list content directly
        if not isinstance(message.get("content"), list):
            valid_history.append(message)
            continue

        # Process list content for tool results
        content = message.get("content", [])
        valid_content = []
        modified = False

        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_result":
                tool_id = item.get("tool_use_id")
                # Only keep tool_results with valid references
                if tool_id in tool_use_info:
                    valid_content.append(item)
                else:
                    modified = True
                    print(f"DEBUG: Removed orphaned tool_result with ID: {tool_id}")
            else:
                valid_content.append(item)

        if content and not valid_content:
            # Skip entirely empty messages
            print(f"DEBUG: Skipping message with only invalid tool results")
            modified = True
            continue

        # Only create a new message if we modified content
        if modified:
            valid_message = dict(message)
            valid_message["content"] = valid_content
            valid_history.append(valid_message)
        else:
            valid_history.append(message)

    # STEP 3: Ensure all tool_use entries have matching results
    has_orphaned_tool_use = False

    # Map tool_results back to their tool_use entries
    result_mapping = {}
    for msg_idx, message in enumerate(valid_history):
        if message.get("role") == "user" and isinstance(message.get("content"), list):
            for item in message.get("content", []):
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    tool_id = item.get("tool_use_id")
                    if tool_id:
                        result_mapping[tool_id] = True

    # Check for orphaned tool_use entries
    for tool_id in tool_use_info:
        if tool_id not in result_mapping:
            has_orphaned_tool_use = True
            print(f"DEBUG: Found orphaned tool_use with ID: {tool_id}")

    # If we have orphaned tool_use entries, consider more aggressive trimming or repair
    if has_orphaned_tool_use:
        print("WARNING: Conversation contains orphaned tool_use entries")
        # For severe cases, we might want to:
        # 1. Remove all tool-related messages to start fresh
        # 2. Keep only the most recent N non-tool messages

    return valid_history

async def main():
    """
    Main entry point for LAURA (Language & Automation User Response Agent).
    
    Manages the initialization and lifecycle of core system components:
    - Token Management: Handles API token usage and limits
    - Document Management: Controls file loading and processing
    - Transcription: Manages speech-to-text (local or remote)
    - Display: Controls visual feedback and animations
    - Calendar: Monitors and notifies of upcoming events

    The system uses asyncio for concurrent operation of multiple tasks:
    - Background animations
    - Main interaction loop
    - Calendar monitoring
    - Heartbeat monitoring (for remote transcription)

    Last Updated: 2025-04-24 16:36:20 UTC
    Author: fandango328
    """
    global remote_transcriber, display_manager, transcriber, token_manager
    global chat_log, document_manager, system_manager, keyboard_device
    global audio_manager, tts_handler, anthropic_client
    tasks = []

    try:
        # CORE MANAGER INITIALIZATION PHASE
        print("\n=== Core Manager Initialization ===")
        
        # 1. Display Manager (Primary UI System)
        print("\nInitializing Display Manager...")
        try:
            display_manager = DisplayManager()
        except Exception as e:
            print(f"Critical Error: Display initialization failed: {e}")
            return

        # 2. Audio System
        print("\nInitializing Audio Manager...")
        try:
            audio_manager = AudioManager()
        except Exception as e:
            print(f"Critical Error: Audio initialization failed: {e}")
            return

        # 3. Anthropic Client and Token Management
        print("\nInitializing Token Management...")
        try:
            if not anthropic_client:
                from anthropic import Anthropic
                anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
            token_manager = TokenManager(anthropic_client=anthropic_client)
            token_manager.start_session()
        except Exception as e:
            print(f"Critical Error: Token management initialization failed: {e}")
            return

        # 4. TTS Handler
        print("\nInitializing TTS Handler...")
        try:
            from config import TTS_ENGINE, VOICE, ELEVENLABS_MODEL
            from secret import ELEVENLABS_KEY
            
            tts_config = {
                "TTS_ENGINE": config.TTS_ENGINE,
                "ELEVENLABS_KEY": ELEVENLABS_KEY,
                "VOICE": config.VOICE,
                "ELEVENLABS_MODEL": config.ELEVENLABS_MODEL,
            }
            tts_handler = TTSHandler(tts_config)
            
            print("TTS Configuration:")
            print(f"- Engine: {tts_config['TTS_ENGINE']}")
            print(f"- Voice: {tts_config['VOICE']}")
            print(f"- Model: {tts_config['ELEVENLABS_MODEL']}")
            
            tts_handler = TTSHandler(tts_config)
            print("TTS Handler initialized successfully")
            
        except ImportError as e:
            print(f"Critical Error: Failed to import TTS configuration: {e}")
            print("Ensure ElevenLabs parameters are defined in config.py")
            return
        except Exception as e:
            print(f"Critical Error: TTS initialization failed: {e}")
            print(f"Config attempted: {tts_config}")
            return

        # KEYBOARD INITIALIZATION PHASE
        print("\n=== Keyboard Initialization ===")
        keyboard_devices = []
        
        for path in list_devices():
            try:
                device = InputDevice(path)
                try:
                    select.select([device.fd], [], [], 0)
                    if "Pi 500" in device.name and "Keyboard" in device.name and "Mouse" not in device.name:
                        try:
                            device.grab()
                            device.ungrab()
                            keyboard_devices.append(device)
                        except Exception:
                            device.close()
                    else:
                        device.close()
                except Exception:
                    device.close()
            except Exception as e:
                print(f"Error with device {path}: {e}")

        if keyboard_devices:
            keyboard_device = keyboard_devices[0]
            print(f"{Fore.GREEN}Using keyboard device: {keyboard_device.path}{Fore.WHITE}")
            print(f"{Fore.GREEN}Using keyboard without exclusive access to allow normal typing{Fore.WHITE}")
        else:
            print(f"{Fore.YELLOW}No valid Pi 500 Keyboard found{Fore.WHITE}")
            keyboard_device = None

        # SUPPORTING MANAGERS INITIALIZATION
        print("\n=== Supporting Managers Initialization ===")
        
        # Document Manager
        print("\nInitializing Document Manager...")
        try:
            document_manager = DocumentManager()
        except Exception as e:
            print(f"Error: Document manager initialization failed: {e}")
            return

        # Notification Manager
        print("\nInitializing Notification Manager...")
        try:
            notification_manager = NotificationManager(audio_manager)
            await notification_manager.start()
        except Exception as e:
            print(f"Error: Notification manager initialization failed: {e}")
            return

        # SYSTEM MANAGER INITIALIZATION
        print("\nInitializing System Manager...")
        try:
            system_manager = SystemManager(
                email_manager=email_manager,
                display_manager=display_manager,
                audio_manager=audio_manager,
                document_manager=document_manager,
                notification_manager=notification_manager,
                token_manager=token_manager,
                tts_handler=tts_handler,
                anthropic_client=anthropic_client
            )
            await system_manager.register_managers()
            
            is_init, missing = system_manager.is_initialized()
            if not is_init:
                raise RuntimeError(f"System initialization incomplete. Missing: {', '.join(missing)}")
                
        except Exception as e:
            print(f"Critical Error: System manager initialization failed: {e}")
            return

        # TRANSCRIPTION SETUP
        print("\n=== Transcription System Initialization ===")
        if TRANSCRIPTION_MODE == "remote":
            remote_transcriber = RemoteTranscriber()
            print(f"{Fore.YELLOW}Using remote transcription service{Fore.WHITE}")
        else:
            print(f"{Fore.YELLOW}Using local transcription{Fore.WHITE}")
            if not transcriber:
                if TRANSCRIPTION_ENGINE == "vosk":
                    print("Initializing Vosk transcriber...")
                    transcriber = VoskTranscriber(VOSK_MODEL_PATH)
                else:
                    print("Initializing Whisper transcriber...")
                    transcriber = WhisperCppTranscriber(WHISPER_MODEL_PATH, VAD_SETTINGS)

        # CHAT LOG INITIALIZATION
        print("\n=== Chat Log Initialization ===")
        chat_log = load_recent_context(token_manager=token_manager)
        print(f"Loaded {len(chat_log)} messages from previous conversation")

        # TASK CREATION AND EXECUTION
        print("\n=== Task Management ===")
        tasks = [
            asyncio.create_task(display_manager.rotate_background()),
            asyncio.create_task(run_main_loop()),
            asyncio.create_task(check_upcoming_events())
        ]

        if TRANSCRIPTION_MODE == "remote":
            tasks.append(asyncio.create_task(heartbeat(remote_transcriber)))

        # Set initial display state
        await display_manager.update_display('sleep')
        
        # Execute all tasks
        await asyncio.gather(*tasks, return_exceptions=True)

    except Exception as e:
        print(f"Critical error in main function: {e}")
        traceback.print_exc()
        
    finally:
        # CLEANUP PHASE
        print("\n=== Cleanup Phase ===")
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if TRANSCRIPTION_MODE == "remote" and remote_transcriber:
            await remote_transcriber.cleanup()
        elif transcriber:
            transcriber.cleanup()

        if display_manager:
            display_manager.cleanup()

        if audio_manager:
            await audio_manager.reset_audio_state()

        if keyboard_device:
            keyboard_device.close()
            print(f"{Fore.GREEN}Keyboard device closed successfully{Fore.WHITE}")

        print(f"{Fore.GREEN}All resources released{Fore.WHITE}")

async def run_main_loop():
    """
    Core interaction loop managing LAURA's conversation flow.
    """
    global document_manager, chat_log, system_manager, keyboard_device, notification_manager
    iteration_count = 0

    # Initialize SystemManager if not already initialized
    async with system_manager_lock:
        if system_manager is None:
            print("\n=== System Manager Initialization Debug ===")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Verify and debug required global managers
            required_managers = {
                'display_manager': display_manager,
                'audio_manager': audio_manager,
                'token_manager': token_manager,
                'document_manager': document_manager
            }
            
            print("\nManager Status:")
            for name, manager in required_managers.items():
                status = "✓ Available" if manager is not None else "✗ Missing"
                print(f"{name}: {status}")
                if manager is not None:
                    print(f"  Type: {type(manager).__name__}")
            
            missing_managers = [name for name, manager in required_managers.items() 
                              if manager is None]
            
            if missing_managers:
                print(f"\nError: Required managers not found: {', '.join(missing_managers)}")
                print("Initialization aborted.")
                return
            
            try:
                print("\n--- Notification Manager Initialization ---")
                if notification_manager is None:
                    print("Status: Creating new instance")
                    try:
                        notification_manager = NotificationManager(audio_manager)
                        await notification_manager.start()
                        print("Status: Successfully initialized")
                    except Exception as notif_error:
                        print(f"Status: Failed - {str(notif_error)}")
                        traceback.print_exc()
                        return
                else:
                    print("Status: Using existing instance")

                print("\n--- System Manager Creation ---")
                try:
                    print("Status: Creating new instance")
                    system_manager = SystemManager(
                        email_manager=email_manager,
                        display_manager=display_manager,
                        audio_manager=audio_manager,
                        document_manager=document_manager,
                        notification_manager=notification_manager,
                        token_manager=token_manager,
                        tts_handler=tts_handler,
                        anthropic_client=anthropic_client
                    )
                    
                    print("Status: Registering managers")
                    await system_manager.register_managers()
                    
                    print("Status: Verifying initialization")
                    is_init, missing = system_manager.is_initialized()
                    
                    if not is_init:
                        print("\nInitialization verification failed!")
                        print(f"Missing components: {', '.join(missing)}")
                        return
                    
                    print("\n=== System Manager Initialization Complete ===")
                    print(f"{Fore.MAGENTA}Listening for wake word or press Raspberry button to begin...{Fore.WHITE}")
                    
                except Exception as sys_error:
                    print(f"\nSystem Manager creation failed: {sys_error}")
                    traceback.print_exc()
                    return
                    
            except Exception as e:
                print(f"\nCritical initialization error: {e}")
                traceback.print_exc()
                return
    
    while True:
        # Start of iteration - clear variables
        wake_detected = False
        detected_model = None

        iteration_count += 1

        try:
            current_state = display_manager.current_state

            # Sleep/Idle states - check for wake events and notifications
            if current_state in ['sleep', 'idle']:
                # Process any pending notifications during idle periods
                if await notification_manager.has_pending_notifications():
                    await notification_manager.process_pending_notifications()

                # Quick non-blocking keyboard check
                if keyboard_device:
                    try:
                        r, w, x = select.select([keyboard_device.fd], [], [], 0)
                        if r:
                            events = read_keyboard_events(keyboard_device)
                            for event in events:
                                if event.type == ecodes.EV_KEY and event.code == 125 and event.value == 1:
                                    print(f"{Fore.GREEN}LEFTMETA key detected - activating{Fore.WHITE}")
                                    wake_detected = True
                                    break
                    except (IOError, BlockingIOError):
                        pass
                    except Exception as e:
                        print(f"Keyboard check error: {e}")

                # Quick wake word check if no keyboard wake
                if not wake_detected:
                    detected_model = await wake_word()
                    if detected_model:
                        wake_detected = True

                # If no wake event detected, quick sleep and continue
                if not wake_detected:
                    await asyncio.sleep(0.1)
                    continue

                # Handle wake event
                if wake_detected:
                    # Only play wake audio if wake word was used (not keyboard)
                    if detected_model:
                        await display_manager.update_display('wake')                        
                        wake_audio = get_random_audio('wake', detected_model)

                        if wake_audio:
                            await audio_manager.queue_audio(audio_file=wake_audio, priority=True)
                            await audio_manager.wait_for_queue_empty()

                    # Transition to listening state and pause wake word detection
                    await display_manager.update_display('listening')

                    # Properly manage wake word detection state
                    wake_word_active = False
                    if hasattr(wake_word, 'state') and wake_word.state.get('stream'):
                        wake_word.state['stream'].stop_stream()
                        wake_word_active = True

                    try:
                        transcript = await capture_speech(is_follow_up=False)
                    finally:
                        # Always restore wake word detection regardless of transcript result
                        if wake_word_active:
                            await asyncio.sleep(0.1)  # Small buffer to prevent immediate re-trigger
                            wake_word.state['stream'].start_stream()

                    if not transcript:
                        print(f"No input detected. Returning to sleep state.")
                        await display_manager.update_display('sleep')
                        await asyncio.sleep(0.1)  # Small buffer before next iteration
                        continue

                    # Process the transcript
                    await audio_manager.wait_for_queue_empty()
                    transcript_lower = transcript.lower().strip()

                    # Check for system commands
                    command_result = system_manager.detect_command(transcript)
                    if command_result:
                        print(f"\nDEBUG: System command detected: {command_result}")
                        is_command, command_type, action, arguments = command_result
                        if is_command:
                            try:
                                await display_manager.update_display('tools')
                                success = await system_manager.handle_command(command_type, action, arguments)
                                
                                # Wait for any queued audio to complete
                                await audio_manager.wait_for_queue_empty()
                                
                                if success:
                                    await display_manager.update_display('listening')
                                    
                                    # Handle follow-up conversation
                                    follow_up = await capture_speech(is_follow_up=True)
                                    while follow_up:
                                        cmd_result = system_manager.detect_command(follow_up)
                                        if cmd_result and cmd_result[0]:
                                            is_cmd, cmd_type, action, args = cmd_result
                                            await display_manager.update_display('tools')
                                            success = await system_manager.handle_command(cmd_type, action, args)
                                            await audio_manager.wait_for_queue_empty()
                                            
                                            if success:
                                                await display_manager.update_display('listening')
                                                follow_up = await capture_speech(is_follow_up=True)
                                                continue
                                            break
                                        
                                        # Handle non-command follow-up
                                        user_message = {"role": "user", "content": follow_up}
                                        if not any(msg["role"] == "user" and msg["content"] == follow_up 
                                                 for msg in chat_log[-2:]):
                                            chat_log.append(user_message)
                                        
                                        await display_manager.update_display('thinking')
                                        formatted_response = await generate_response(follow_up)
                                        
                                        if formatted_response == "[CONTINUE]":
                                            print("DEBUG - Detected control signal in follow-up, continuing")
                                            continue
                                            
                                        try:
                                            await display_manager.update_display('speaking')
                                            await generate_voice(formatted_response)
                                            
                                            # Check notifications before state transition
                                            if await notification_manager.has_pending_notifications():
                                                await notification_manager.process_pending_notifications()
                                                
                                            if isinstance(formatted_response, str) and has_conversation_hook(formatted_response):
                                                await audio_manager.wait_for_queue_empty()
                                                await display_manager.update_display('listening')
                                                await handle_conversation_loop(formatted_response)
                                                break
                                            else:
                                                await audio_manager.wait_for_queue_empty()
                                                break
                                        except Exception as voice_error:
                                            print(f"Error during follow-up voice generation: {voice_error}")
                                            await display_manager.update_display('sleep')
                                            break
                                else:
                                    await display_manager.update_display('sleep')
                            except Exception as e:
                                print(f"Error handling command: {e}")
                                traceback.print_exc()
                                await display_manager.update_display('sleep')
                            continue

                    # Normal conversation flow
                    try:
                        # Make sure any previous audio has completed
                        await audio_manager.wait_for_queue_empty()
                        
                        # Set thinking state and generate response
                        await display_manager.update_display('thinking')
                        formatted_response = await generate_response(transcript)

                        if formatted_response == "[CONTINUE]":
                            print("DEBUG - Detected control signal, skipping voice generation")
                            continue

                        # Ensure audio is complete before state transitions
                        await audio_manager.wait_for_queue_empty()
                        
                        # Transition to speaking state
                        await display_manager.update_display('speaking')
                        await generate_voice(formatted_response)

                        # Wait for speech to complete
                        await audio_manager.wait_for_queue_empty()

                        # Check and process any pending notifications
                        if await notification_manager.has_pending_notifications():
                            await notification_manager.process_pending_notifications()

                        # Always transition to listening and continue conversation unless explicitly ended
                        await display_manager.update_display('listening')
                        await handle_conversation_loop(formatted_response)

                    except Exception as voice_error:
                        print(f"Error during voice generation: {voice_error}")
                        traceback.print_exc()
                        await display_manager.update_display('sleep')
                        continue

            else:
                # If we're in idle state, check for timeout to sleep
                if current_state == 'idle':
                    now = datetime.now()
                    if (now - last_interaction).total_seconds() > CONVERSATION_END_SECONDS:
                        print(f"{Fore.CYAN}Conversation timeout reached ({CONVERSATION_END_SECONDS} seconds), transitioning to sleep state...{Fore.WHITE}")
                        await display_manager.update_display('sleep')
                    await asyncio.sleep(0.1)
                    continue
                # If in any other state (besides sleep), ensure we're tracking last_interaction
                elif current_state not in ['sleep']:
                    last_interaction = datetime.now()

        except Exception as e:
            print(f"Error in main loop: {e}")
            traceback.print_exc()
            now = datetime.now()
            if (now - last_interaction).total_seconds() > CONVERSATION_END_SECONDS:
                await display_manager.update_display('sleep')
            else:
                await display_manager.update_display('idle')
            await asyncio.sleep(0.1)
            continue

if __name__ == "__main__":
    try:
        display_manager = DisplayManager()
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting Virtual Assistant - Cleaning up resources...")

        # First stop any active processes
        if 'audio_manager' in globals() and audio_manager:
            try:
                print("Cleaning up audio manager...")
                asyncio.run(audio_manager.cleanup())
            except Exception as e:
                print(f"Error cleaning up audio manager: {e}")

        # Release the transcriber resources explicitly
        if 'transcriber' in globals() and transcriber:
            try:
                print("Cleaning up transcriber...")
                transcriber.cleanup()
                transcriber = None  # Explicitly clear the reference
            except Exception as e:
                print(f"Error cleaning up transcriber: {e}")

        # Clean up remaining components
        if 'remote_transcriber' in globals() and remote_transcriber:
            try:
                asyncio.run(remote_transcriber.cleanup())
            except Exception as e:
                print(f"Error cleaning up remote transcriber: {e}")

        if 'display_manager' in globals() and display_manager:
            try:
                display_manager.cleanup()
            except Exception as e:
                print(f"Error cleaning up display manager: {e}")

        # Final garbage collection pass
        import gc
        gc.collect()
        print("Cleanup complete!")
