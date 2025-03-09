﻿#!/usr/bin/env python3

# =============================================================================
# Standard Library Imports
# =============================================================================
import os
import re
import time
import json
import base64
import random
import struct
import textwrap
import threading
import traceback
import asyncio
import websockets
import snowboydetect
import random
import glob
from pathlib import Path
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Dict, Tuple, List, Union, Optional
from whisper_transcriber import WhisperCppTranscriber
import struct

# =============================================================================
# Third-Party Imports
# =============================================================================
from PIL import Image
from mutagen.mp3 import MP3

from anthropic import Anthropic, APIError, APIConnectionError, BadRequestError, InternalServerError

import pyaudio
import whisper
import collections
import numpy as np
import wave

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from colorama import Fore, Style
from elevenlabs.client import ElevenLabs
from vosk_transcriber import VoskTranscriber

# =============================================================================
# Local Module Imports and Configuration
# =============================================================================
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from display_manager import DisplayManager
from audio_manager_whisper import AudioManager
from secret import GOOGLE_MAPS_API_KEY, OPENROUTER_API_KEY, PV_ACCESS_KEY, ELEVENLABS_KEY, ANTHROPIC_API_KEY

from config_transcription_v2 import (
    TRANSCRIPTION_SERVER,
    TRANSCRIPTION_MODE,
    VOSK_MODEL_PATH,
    TRANSCRIPTION_ENGINE,
    WHISPER_MODEL_PATH,
    VAD_SETTINGS,
    VOICE,
    MOODS,
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
    TTS_ENGINE,
    ANTHROPIC_MODEL,
    CALENDAR_NOTIFICATION_SENTENCES
)

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

config = {
    "TTS_ENGINE": TTS_ENGINE,
    "ELEVENLABS_KEY": ELEVENLABS_KEY,
    "VOICE": VOICE,
    "ELEVENLABS_MODEL": ELEVENLABS_MODEL, 
}
# Email importance configuration - update this with information about people you care about
EMAIL_IMPORTANCE_CONFIG = {
    # Important senders (exact matches or domains)
    "important_senders": [
        "boss@company.com",  # Replace with actual emails
        "vp@company.com",
        "@executive-team.company.com",  # Anyone from this domain
    ],
    
    # Keywords that indicate action items or urgency
    "action_keywords": [
        "urgent",
        "action required", 
        "action item",
        "please review",
        "deadline",
        "asap",
        "by tomorrow",
        "by eod",
        "assigned to you",
    ],
    
    # Project or topic importance (customize based on your priorities)
    "important_topics": [
        "quarterly review",
        "performance review",
        "key project",
        "budget approval",
    ]
}

#Tools are currently structured to work with anthropic api docs for "tool use" function calling.  These need to be updated to fix your LLM interface requirements (i.e. you must update the way this is structured to fit your llm interface, anthropic is differente from openai, which is different from openrouter, which will also be different if you host on a local llm.  You must ensure that these are structured appropriately!!!) 
TOOLS = [
    {
        "name": "draft_email",
        "description": "Draft a new email with a recipient and content. This tool creates a draft email in the user's Gmail account. It should be used when the user wants to compose or prepare an email message. The tool will create the draft but will not send it automatically.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "Email address of who will receive the email"
                },
                "subject": {
                    "type": "string",
                    "description": "The topic or subject line of the email"
                },
                "content": {
                    "type": "string",
                    "description": "The main body content of the email"
                }
            },
            "required": ["subject", "content"]
        }
    },
    {
        "name": "calendar_query",
        "description": "Get information about calendar events. Can retrieve next event or full day schedule.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["next_event", "day_schedule", "check_availability"],
                    "description": "Type of calendar query ('next_event' for next upcoming event, 'day_schedule' for full day's schedule)"
                },
                "date": {
                    "type": "string",
                    "description": "Date for query (optional)"
                }
            },
            "required": ["query_type"]
        }
    },
    {
        "name": "get_current_time",
        "description": "Get the current time in the local timezone",
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["time", "date", "both"],
                    "description": "Format of time information to return"
                }
            },
            "required": ["format"]
        }
    },
    {
        "name": "create_calendar_event",
        "description": "Create a new calendar event",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Event title"
                },
                "start_time": {
                    "type": "string",
                    "description": "Start time in ISO format"
                },
                "end_time": {
                    "type": "string",
                    "description": "End time in ISO format"
                },
                "description": {
                    "type": "string",
                    "description": "Event description"
                },
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of attendee email addresses"
                }
            },
            "required": ["summary", "start_time", "end_time"]
        }
    },
    {
        "name": "get_location",
        "description": "Get current location based on WiFi networks. This tool scans nearby WiFi networks and uses them to determine the current geographic location. It can return the location in different formats including coordinates or street address.",
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["coordinates", "address", "both"],
                    "description": "Format of location information to return"
                }
            },
            "required": ["format"]
        }
    },
    {
        "name": "calibrate_voice_detection",
        "description": "Calibrate the voice detection system to improve speech recognition. This tool runs a calibration process that measures background noise and voice levels to optimize how the system detects when you're speaking.",
        "input_schema": {
            "type": "object",
            "properties": {
                "confirm": {
                    "type": "boolean",
                    "description": "Confirmation to run calibration"
                }
            },
            "required": ["confirm"]
        }
    },
    {
        "name": "read_emails",
        "description": "Retrieve and read emails from the user's Gmail inbox with various filtering options. Can identify important emails based on sender, content, and urgency.",
        "input_schema": {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of emails to retrieve (default: 5)"
                },
                "unread_only": {
                    "type": "boolean",
                    "description": "Whether to only return unread emails"
                },
                "query": {
                    "type": "string",
                    "description": "Gmail search query (e.g., 'from:john@example.com', 'subject:meeting', 'after:2023/04/01')"
                },
                "include_content": {
                    "type": "boolean",
                    "description": "Whether to include the full email content or just headers"
                },
                "mark_as_read": {
                    "type": "boolean", 
                    "description": "Whether to mark retrieved emails as read"
                }
            }
        }
    },
    {
        "name": "email_action",
        "description": "Perform actions on specific emails like archive, delete, or mark as read/unread",
        "input_schema": {
            "type": "object",
            "properties": {
                "email_id": {
                    "type": "string",
                    "description": "ID of the email to act upon"
                },
                "action": {
                    "type": "string",
                    "enum": ["archive", "trash", "mark_read", "mark_unread", "star", "unstar"],
                    "description": "Action to perform on the email"
                }
            },
            "required": ["email_id", "action"]
        }
    },
    {
        "name": "update_calendar_event",
        "description": "Update an existing calendar event's details such as time, location, or attendees",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "ID of the calendar event to update"
                },
                "summary": {
                    "type": "string",
                    "description": "New title for the event (optional)"
                },
                "start_time": {
                    "type": "string",
                    "description": "New start time in ISO format (optional)"
                },
                "end_time": {
                    "type": "string",
                    "description": "New end time in ISO format (optional)"
                },
                "description": {
                    "type": "string",
                    "description": "New description for the event (optional)"
                },
                "location": {
                    "type": "string",
                    "description": "New location for the event (optional)"
                },
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Updated list of attendee email addresses (optional)"
                }
            },
            "required": ["event_id"]
        }
    },
    {
        "name": "cancel_calendar_event",
        "description": "Cancel an existing calendar event and optionally notify attendees",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "ID of the calendar event to cancel"
                },
                "notify_attendees": {
                    "type": "boolean",
                    "description": "Whether to send cancellation emails to attendees"
                },
                "cancellation_message": {
                    "type": "string",
                    "description": "Optional message to include in the cancellation notification"
                }
            },
            "required": ["event_id"]
        }
    },
    {
        "name": "manage_tasks",
        "description": "Create, update, list, or complete tasks in Google Tasks",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "update", "list", "complete", "delete"],
                    "description": "Action to perform on tasks"
                },
                "title": {
                    "type": "string",
                    "description": "Title of the task (for create/update)"
                },
                "notes": {
                    "type": "string",
                    "description": "Additional notes or details for the task"
                },
                "due_date": {
                    "type": "string",
                    "description": "Due date for the task in ISO format or natural language (e.g., 'tomorrow')"
                },
                "task_id": {
                    "type": "string",
                    "description": "ID of the task to update/complete/delete"
                },
                "status": {
                    "type": "string",
                    "enum": ["needsAction", "completed"],
                    "description": "Status of the task"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of tasks to return when listing"
                },
                "list_id": {
                    "type": "string",
                    "description": "ID of the task list (optional, uses default if not specified)"
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "create_task_from_email",
        "description": "Create a task based on an email",
        "input_schema": {
            "type": "object",
            "properties": {
                "email_id": {
                    "type": "string",
                    "description": "ID of the email to convert to a task"
                },
                "title": {
                    "type": "string",
                    "description": "Custom title for the task (optional, will use email subject if not provided)"
                },
                "due_date": {
                    "type": "string",
                    "description": "Due date for the task (optional)"
                },
                "priority": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Priority level for the task"
                }
            },
            "required": ["email_id"]
        }
    },
    {
        "name": "create_task_for_event",
        "description": "Create preparation or follow-up tasks for a calendar event",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "ID of the calendar event to create tasks for"
                },
                "task_type": {
                    "type": "string",
                    "enum": ["preparation", "follow_up", "both"],
                    "description": "Type of tasks to create"
                },
                "days_before": {
                    "type": "integer",
                    "description": "Days before the event for preparation tasks"
                },
                "days_after": {
                    "type": "integer",
                    "description": "Days after the event for follow-up tasks"
                },
                "custom_titles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Custom titles for tasks (optional)"
                }
            },
            "required": ["event_id", "task_type"]
        }
    },
    {
        "name": "manage_contacts",
        "description": "Manage contact information for people and organizations",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "create", "update", "list", "search"],
                    "description": "Action to perform on contacts"
                },
                "name": {
                    "type": "string",
                    "description": "Full name of the contact"
                },
                "email": {
                    "type": "string",
                    "description": "Email address of the contact"
                },
                "phone": {
                    "type": "string",
                    "description": "Phone number of the contact"
                },
                "company": {
                    "type": "string",
                    "description": "Organization or company name"
                },
                "relationship": {
                    "type": "string",
                    "description": "Relationship to the user (e.g., 'manager', 'colleague', 'client')"
                },
                "importance": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Importance level of the contact"
                },
                "query": {
                    "type": "string",
                    "description": "Search term for finding contacts"
                }
            },
            "required": ["action"]
        }
    }
]

#end of tools, start of classes

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

class TokenTracker:
    def __init__(self, anthropic_client):
        self.client = anthropic_client
        self.session_tokens = {
            "input": {
                "total": 0,
                "system_prompt": 0,
                "queries": 0,
                "chat_history": 0,
                "tool_overhead": 0
            },
            "output": {
                "total": 0,
                "responses": 0,
                "tool_results": 0
            },
            "costs": {
                "input": 0.0,
                "output": 0.0,
                "total": 0.0
            }
        }
        self.TOOL_OVERHEAD = 346
        self._initialize_system_prompt_count()

    def _initialize_system_prompt_count(self) -> None:
        try:
            if SYSTEM_PROMPT:
                count = self.client.messages.count_tokens(
                    model=ANTHROPIC_MODEL,
                    system=SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": "test"
                    }]
                ).input_tokens
                self.session_tokens["input"]["system_prompt"] = count
                self.session_tokens["input"]["total"] += count
                self._update_costs("input", count)
        except Exception as e:
            print(f"Error counting system prompt tokens: {e}")
            
    def count_message_tokens(self, messages: List[Dict[str, str]], include_system: bool = False) -> int:
        try:
            if not messages:
                return 0
                
            if include_system and SYSTEM_PROMPT:
                all_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
            else:
                all_messages = messages
                
            return self.client.messages.count_tokens(messages=all_messages).tokens
        except Exception as e:
            print(f"Error counting message tokens: {e}")
            return 0

    def update_query_tokens(self, query: str, chat_history: List[Dict[str, str]], using_tools: bool = False) -> None:
        try:
            messages = [{"role": "user", "content": query}]
            query_tokens = self.count_message_tokens(messages)
            
            history_tokens = self.count_message_tokens(chat_history) if chat_history else 0
            tool_overhead = self.TOOL_OVERHEAD if using_tools else 0
            
            self.session_tokens["input"]["queries"] += query_tokens
            self.session_tokens["input"]["chat_history"] += history_tokens
            self.session_tokens["input"]["tool_overhead"] += tool_overhead
            
            total_input = query_tokens + history_tokens + tool_overhead
            self.session_tokens["input"]["total"] += total_input
            self._update_costs("input", total_input)
            
        except Exception as e:
            print(f"Error updating query tokens: {e}")

    def update_response_tokens(self, response_content: Union[str, List[Dict]], is_tool_result: bool = False) -> None:
        try:
            if isinstance(response_content, str):
                response_message = {"role": "assistant", "content": response_content}
            else:
                response_message = {"role": "assistant", "content": json.dumps(response_content)}
            
            tokens = self.count_message_tokens([response_message])
            
            if is_tool_result:
                self.session_tokens["output"]["tool_results"] += tokens
            else:
                self.session_tokens["output"]["responses"] += tokens
                
            self.session_tokens["output"]["total"] += tokens
            self._update_costs("output", tokens)
            
        except Exception as e:
            print(f"Error updating response tokens: {e}")

    def _update_costs(self, token_type: str, token_count: int) -> None:
        cost_per_1k = 0.003 if token_type == "input" else 0.015
        cost = (token_count / 1000) * cost_per_1k
        self.session_tokens["costs"][token_type] += cost
        self.session_tokens["costs"]["total"] = (
            self.session_tokens["costs"]["input"] + 
            self.session_tokens["costs"]["output"]
        )

    def print_metrics(self, label: str = "") -> None:
        print(f"\n=== Token Metrics {label} ===")
        print("Input Tokens:")
        print(f"  System Prompt: {self.session_tokens['input']['system_prompt']}")
        print(f"  Queries: {self.session_tokens['input']['queries']}")
        print(f"  Chat History: {self.session_tokens['input']['chat_history']}")
        print(f"  Tool Overhead: {self.session_tokens['input']['tool_overhead']}")
        print(f"  Total Input: {self.session_tokens['input']['total']}")
        print(f"  Input Cost: ${self.session_tokens['costs']['input']:.4f}")
        
        print("\nOutput Tokens:")
        print(f"  Responses: {self.session_tokens['output']['responses']}")
        print(f"  Tool Results: {self.session_tokens['output']['tool_results']}")
        print(f"  Total Output: {self.session_tokens['output']['total']}")
        print(f"  Output Cost: ${self.session_tokens['costs']['output']:.4f}")
        
        print(f"\nTotal Session Cost: ${self.session_tokens['costs']['total']:.4f}")

class TTSHandler:
    def __init__(self, config):
        self.config = config  # Store config in instance variable
        self.tts_engine = config["TTS_ENGINE"]
        self.eleven = None
        if self.tts_engine == "elevenlabs":
            self.eleven = ElevenLabs(api_key=config["ELEVENLABS_KEY"])
    
    def generate_audio(self, text):
        if self.tts_engine == "elevenlabs":
            return self._generate_elevenlabs(text)
        else:
            return self._generate_alltalk(text)
    
    def _generate_elevenlabs(self, text):
        audio = b"".join(self.eleven.generate(
            text=text,
            voice=self.config["VOICE"],
            model=self.config["ELEVENLABS_MODEL"],  # Changed from model_id to model
            output_format="mp3_44100_128"  # Optional: specify output format
        ))
        return audio
    
    def _generate_alltalk(self, text):
        try:
            response = requests.post(
                f"{self.config['ALLTALK_HOST']}/api/tts",
                json={
                    "text": text,
                    "voice": self.config["ALLTALK_VOICE"],
                    "model": self.config["ALLTALK_MODEL"]
                },
                timeout=30
            )
            response.raise_for_status()
            
            if response.status_code == 200:
                return response.content
            else:
                raise Exception(f"AllTalk API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"AllTalk API request failed: {str(e)}")
            if self.eleven:
                print("Falling back to ElevenLabs...")
                return self._generate_elevenlabs(text)
            raise

# Global variables
transcriber = None  # Initialize to None first
if TRANSCRIPTION_MODE == "local":
    if TRANSCRIPTION_ENGINE == "vosk":
        transcriber = VoskTranscriber(VOSK_MODEL_PATH)
    else:
        transcriber = WhisperCppTranscriber(WHISPER_MODEL_PATH, VAD_SETTINGS)
else:
    transcriber = None  # For remote mode
    
remote_transcriber = None if TRANSCRIPTION_MODE == "local" else RemoteTranscriber()
audio_manager = AudioManager(PV_ACCESS_KEY if TRANSCRIPTION_MODE == "remote" else None)
display_manager = None
chat_log = []
last_interaction = datetime.now()
tts_handler = TTSHandler(config)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
initial_startup = True
last_interaction_check = datetime.now()
calendar_notified_events = set()
# google setup - still part of Global variables
import webbrowser
webbrowser.register('chromium', None, webbrowser.Chrome('/usr/bin/chromium'))
creds = None
if USE_GOOGLE:
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing: {e}")
                os.remove("token.json") if os.path.exists("token.json") else None
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", 
                    SCOPES
                )
                creds = flow.run_local_server(
                    port=8080,
                    host='localhost',
                    open_browser=True,
                    browser_path='/usr/bin/chromium'
                )
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", 
                SCOPES
            )
            creds = flow.run_local_server(
                port=8080,
                host='localhost',
                open_browser=True,
                browser_path='/usr/bin/chromium'
            )
    
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

# Global variables to help with email session management
LAST_EMAIL_RESULTS = {
    "emails": [],
    "timestamp": None
}
# Global variables to store task references
LAST_TASKS_RESULT = {
    "tasks": [],
    "timestamp": None
}
token_tracker = None

#end of global variables, start of function definitions
# Add this just after your global variables, before function definitions:
def verify_token_tracker_setup():
    """Verify token tracker initialization status"""
    print(f"\n{Fore.CYAN}=== Pre-Initialization Check ===")
    print(f"Token Tracker Status: {'Defined' if 'token_tracker' in globals() else 'Missing'}")
    print(f"Anthropic Client Status: {'Initialized' if anthropic_client else 'Missing'}")
    print(f"System Prompt Status: {'Defined' if SYSTEM_PROMPT else 'Missing'}{Fore.WHITE}\n")

def determine_email_importance(email, config=EMAIL_IMPORTANCE_CONFIG):
    """
    Analyzes an email and determines its importance level with explanation.
    
    Returns:
        tuple: (importance_level, explanation)
            importance_level: int (0-5, where 5 is highest importance)
            explanation: str (reason why this email is important)
    """
    importance_score = 0
    reasons = []
    
    # 1. Check sender importance
    sender_email = email.get('sender_email', '').lower()
    
    # Direct match for specific important people
    if any(important_sender.lower() == sender_email for important_sender in config["important_senders"] 
           if not important_sender.startswith('@')):
        importance_score += 3
        sender_name = email.get('sender_name', sender_email)
        reasons.append(f"From important person: {sender_name}")
    
    # Domain match for important groups
    elif any(domain.lower() in sender_email for domain in config["important_senders"] if domain.startswith('@')):
        importance_score += 2
        reasons.append(f"From important team")
    
    # 2. Check for action items/urgency in subject and body
    email_text = (email.get('subject', '') + ' ' + email.get('snippet', '')).lower()
    
    action_matches = [keyword for keyword in config["action_keywords"] 
                     if keyword.lower() in email_text]
    if action_matches:
        importance_score += 2
        reasons.append(f"Contains action keywords: {', '.join(action_matches[:2])}")
    
    # 3. Check for important projects/topics
    topic_matches = [topic for topic in config["important_topics"] 
                    if topic.lower() in email_text]
    if topic_matches:
        importance_score += 1
        reasons.append(f"Related to important topic: {', '.join(topic_matches[:2])}")
    
    # Normalize score to 0-5 range
    final_score = min(5, importance_score)
    
    # Compile explanation if there are reasons
    explanation = "; ".join(reasons) if reasons else "No special importance detected"
    
    return (final_score, explanation)

def read_emails(max_results=5, unread_only=False, query=None, include_content=False, mark_as_read=False):
    """
    Retrieve and read emails from Gmail with importance detection
    """
    global LAST_EMAIL_RESULTS
    
    try:
        service = build("gmail", "v1", credentials=creds)
        
        # Build query string
        query_parts = []
        if unread_only:
            query_parts.append("is:unread")
        if query:
            query_parts.append(query)
        
        final_query = " ".join(query_parts) if query_parts else None
        
        # Get messages
        messages_result = service.users().messages().list(
            userId="me",
            q=final_query,
            maxResults=max_results
        ).execute()
        
        messages = messages_result.get("messages", [])
        
        if not messages:
            return "No emails found matching your criteria."
        
        # Process each email
        emails = []
        important_emails = []
        regular_emails = []
        
        for message in messages:
            msg = service.users().messages().get(
                userId="me", 
                id=message["id"],
                format="full" if include_content else "metadata"
            ).execute()
            
            # Extract headers
            headers = {header["name"].lower(): header["value"] 
                      for header in msg["payload"]["headers"]}
            
            # Extract key information
            email_data = {
                "id": msg["id"],
                "thread_id": msg["threadId"],
                "subject": headers.get("subject", "(No subject)"),
                "sender_name": headers.get("from", "").split("<")[0].strip(),
                "sender_email": headers.get("from", ""),
                "date": headers.get("date", ""),
                "to": headers.get("to", ""),
                "cc": headers.get("cc", ""),
                "labels": msg["labelIds"],
                "snippet": msg.get("snippet", ""),
                "unread": "UNREAD" in msg["labelIds"]
            }
            
            # Extract sender email from format "Name <email>"
            if "<" in email_data["sender_email"] and ">" in email_data["sender_email"]:
                email_data["sender_email"] = email_data["sender_email"].split("<")[1].split(">")[0]
            
            # Extract full content if requested
            if include_content:
                email_data["body"] = extract_email_body(msg["payload"])
            
            # Determine importance
            importance_score, importance_reason = determine_email_importance(email_data)
            email_data["importance"] = importance_score
            email_data["importance_reason"] = importance_reason
            
            emails.append(email_data)
            
            # Categorize by importance
            if importance_score >= 3:
                important_emails.append(email_data)
            else:
                regular_emails.append(email_data)
        
        # Mark as read if requested
        if mark_as_read:
            email_ids_to_mark = [email["id"] for email in emails if email["unread"]]
            if email_ids_to_mark:
                service.users().messages().batchModify(
                    userId="me",
                    body={
                        "ids": email_ids_to_mark,
                        "removeLabelIds": ["UNREAD"]
                    }
                ).execute()
                
                # Update our local status too
                for email in emails:
                    if email["id"] in email_ids_to_mark:
                        email["unread"] = False
                        if "UNREAD" in email["labels"]:
                            email["labels"].remove("UNREAD")
        
        # Store results for future reference
        LAST_EMAIL_RESULTS = {
            "emails": emails,
            "timestamp": datetime.now(),
            "important": important_emails,
            "regular": regular_emails
        }
        
        # Build response for the assistant
        response = ""
        
        # Summarize important emails
        if important_emails:
            response += f"You have {len(important_emails)} important email{'s' if len(important_emails) > 1 else ''}:\n\n"
            
            for i, email in enumerate(important_emails, 1):
                response += f"{i}. From: {email['sender_name']} - {email['subject']}\n"
                response += f"   {email['importance_reason']}\n"
                response += f"   {email['snippet'][:100]}...\n\n"
        
        # Summarize other emails
        if regular_emails:
            if important_emails:
                response += f"You also have {len(regular_emails)} other email{'s' if len(regular_emails) > 1 else ''}:\n\n"
            else:
                response += f"You have {len(regular_emails)} email{'s' if len(regular_emails) > 1 else ''}:\n\n"
                
            for i, email in enumerate(regular_emails, 1):
                response += f"{i}. From: {email['sender_name']} - {email['subject']}\n"
                response += f"   {email['snippet'][:50]}...\n\n"
        
        # Add navigation hint
        response += "You can ask me to read any specific email in full or take actions like marking them as read or archiving."
        
        return response
        
    except Exception as e:
        print(f"Error reading emails: {e}")
        traceback.print_exc()
        return f"Sorry, I encountered an error while trying to read your emails: {str(e)}"

def extract_email_body(payload):
    """Helper function to extract email body content from the Gmail API response"""
    body = ""
    
    if "body" in payload and payload["body"].get("data"):
        # Base64 decode the email body
        body = base64.urlsafe_b64decode(payload["body"]["data"].encode("ASCII")).decode("utf-8")
    
    # If this is a multipart message, check the parts
    elif "parts" in payload:
        for part in payload["parts"]:
            if part["mimeType"] == "text/plain" and part["body"].get("data"):
                body = base64.urlsafe_b64decode(part["body"]["data"].encode("ASCII")).decode("utf-8")
                break
            # Handle nested multipart messages
            elif "parts" in part:
                body = extract_email_body(part)
                if body:
                    break
    
    return body

def email_action(email_id, action):
    """
    Perform an action on a specific email
    """
    try:
        service = build("gmail", "v1", credentials=creds)
        
        if action == "archive":
            # Remove INBOX label
            service.users().messages().modify(
                userId="me",
                id=email_id,
                body={"removeLabelIds": ["INBOX"]}
            ).execute()
            return "Email archived successfully."
            
        elif action == "trash":
            # Move to trash
            service.users().messages().trash(
                userId="me",
                id=email_id
            ).execute()
            return "Email moved to trash."
            
        elif action == "mark_read":
            # Remove UNREAD label
            service.users().messages().modify(
                userId="me",
                id=email_id,
                body={"removeLabelIds": ["UNREAD"]}
            ).execute()
            return "Email marked as read."
            
        elif action == "mark_unread":
            # Add UNREAD label
            service.users().messages().modify(
                userId="me",
                id=email_id,
                body={"addLabelIds": ["UNREAD"]}
            ).execute()
            return "Email marked as unread."
            
        elif action == "star":
            # Add STARRED label
            service.users().messages().modify(
                userId="me",
                id=email_id,
                body={"addLabelIds": ["STARRED"]}
            ).execute()
            return "Email starred."
            
        elif action == "unstar":
            # Remove STARRED label
            service.users().messages().modify(
                userId="me",
                id=email_id,
                body={"removeLabelIds": ["STARRED"]}
            ).execute()
            return "Email unstarred."
            
        else:
            return f"Unknown action: {action}"
            
    except Exception as e:
        print(f"Error performing email action: {e}")
        traceback.print_exc()
        return f"Sorry, I encountered an error while trying to {action} the email: {str(e)}"

def manage_tasks(action, title=None, notes=None, due_date=None, task_id=None, 
                status=None, max_results=10, list_id=None):
    """
    Manage Google Tasks - create, update, list, complete, or delete tasks
    """
    global LAST_TASKS_RESULT
    
    try:
        # Build the Tasks API service
        service = build('tasks', 'v1', credentials=creds)
        
        # Get default task list if not specified
        if not list_id:
            lists = service.tasklists().list().execute()
            list_id = lists['items'][0]['id'] if 'items' in lists else None
            if not list_id:
                return "No task lists found. Please create a task list first."
        
        # HANDLE DIFFERENT ACTIONS
        
        # CREATE a new task
        if action == "create":
            if not title:
                return "Task title is required for creating a task."
                
            # Prepare the task data
            task_data = {
                'title': title,
            }
            
            if notes:
                task_data['notes'] = notes
                
            if due_date:
                # Convert natural language dates to ISO format
                parsed_date = parse_natural_date(due_date)
                if parsed_date:
                    # Google Tasks API expects RFC 3339 timestamp
                    task_data['due'] = parsed_date.isoformat() + 'Z'  # UTC time
            
            # Create the task
            result = service.tasks().insert(tasklist=list_id, body=task_data).execute()
            
            return f"Task '{title}' has been created successfully."
            
        # UPDATE an existing task
        elif action == "update":
            if not task_id:
                return "Task ID is required for updating a task."
                
            # Get the current task data
            task = service.tasks().get(tasklist=list_id, task=task_id).execute()
            
            # Update the fields if provided
            if title:
                task['title'] = title
                
            if notes:
                task['notes'] = notes
                
            if due_date:
                # Convert natural language dates to ISO format
                parsed_date = parse_natural_date(due_date)
                if parsed_date:
                    task['due'] = parsed_date.isoformat() + 'Z'  # UTC time
            
            if status:
                task['status'] = status
                
            # Update the task
            result = service.tasks().update(tasklist=list_id, task=task_id, body=task).execute()
            
            return f"Task '{task['title']}' has been updated successfully."
            
        # COMPLETE a task
        elif action == "complete":
            if not task_id:
                return "Task ID is required for completing a task."
                
            # Get the current task data
            task = service.tasks().get(tasklist=list_id, task=task_id).execute()
            
            # Mark as completed
            task['status'] = 'completed'
            
            # Update the task
            result = service.tasks().update(tasklist=list_id, task=task_id, body=task).execute()
            
            return f"Task '{task['title']}' has been marked as completed."
            
        # DELETE a task
        elif action == "delete":
            if not task_id:
                return "Task ID is required for deleting a task."
                
            # Get the task title first for confirmation
            task = service.tasks().get(tasklist=list_id, task=task_id).execute()
            task_title = task.get('title', 'Unnamed task')
            
            # Delete the task
            service.tasks().delete(tasklist=list_id, task=task_id).execute()
            
            return f"Task '{task_title}' has been deleted."
            
        # LIST tasks
        elif action == "list":
            # Get tasks
            tasks_result = service.tasks().list(
                tasklist=list_id,
                maxResults=max_results,
                showCompleted=True,
                showHidden=False
            ).execute()
            
            tasks = tasks_result.get('items', [])
            
            if not tasks:
                return "No tasks found in this list."
                
            # Store tasks for later reference
            processed_tasks = []
            upcoming_tasks = []
            completed_tasks = []
            
            for task in tasks:
                task_info = {
                    'id': task['id'],
                    'title': task.get('title', 'Unnamed task'),
                    'status': task.get('status', 'needsAction'),
                    'notes': task.get('notes', ''),
                }
                
                # Parse due date if available
                if 'due' in task:
                    due_date = datetime.fromisoformat(task['due'].replace('Z', '+00:00'))
                    task_info['due_date'] = due_date.strftime('%B %d, %Y')
                else:
                    task_info['due_date'] = None
                
                processed_tasks.append(task_info)
                
                # Separate upcoming and completed tasks
                if task_info['status'] == 'completed':
                    completed_tasks.append(task_info)
                else:
                    upcoming_tasks.append(task_info)
            
            # Store in global variable for future reference
            LAST_TASKS_RESULT = {
                "tasks": processed_tasks,
                "timestamp": datetime.now(),
                "list_id": list_id
            }
            
            # Format response for voice
            response = ""
            
            if upcoming_tasks:
                response += f"You have {len(upcoming_tasks)} upcoming tasks:\n\n"
                for i, task in enumerate(upcoming_tasks, 1):
                    due_str = f" (Due: {task['due_date']})" if task['due_date'] else ""
                    response += f"{i}. {task['title']}{due_str}\n"
                    if task['notes']:
                        # Add a brief preview of notes if they exist
                        notes_preview = task['notes'][:50] + "..." if len(task['notes']) > 50 else task['notes']
                        response += f"   Note: {notes_preview}\n"
                response += "\n"
            
            if completed_tasks:
                response += f"You also have {len(completed_tasks)} completed tasks.\n\n"
            
            response += "You can ask me to create, update, complete, or delete specific tasks."
            
            return response
        
        else:
            return f"Unknown action: {action}. Please specify 'create', 'update', 'list', 'complete', or 'delete'."
            
    except Exception as e:
        print(f"Error managing tasks: {e}")
        traceback.print_exc()
        return f"Sorry, I encountered an error while trying to manage your tasks: {str(e)}"

def create_task_from_email(email_id, title=None, due_date=None, priority="medium"):
    """
    Create a task based on an email
    """
    try:
        # Get email details
        gmail_service = build("gmail", "v1", credentials=creds)
        
        msg = gmail_service.users().messages().get(
            userId="me", 
            id=email_id,
            format="metadata"
        ).execute()
        
        # Extract headers
        headers = {header["name"].lower(): header["value"] 
                  for header in msg["payload"]["headers"]}
        
        # Get email subject and sender
        subject = headers.get("subject", "(No subject)")
        sender = headers.get("from", "").split("<")[0].strip()
        
        # Create task title if not provided
        if not title:
            title = f"Email: {subject}"
        
        # Create notes with email details
        notes = f"From: {sender}\nSubject: {subject}\nEmail ID: {email_id}\n\n"
        notes += f"Snippet: {msg.get('snippet', '')}\n\n"
        
        # Add priority to notes
        if priority:
            notes += f"Priority: {priority.upper()}\n"
        
        # Create the task
        tasks_service = build('tasks', 'v1', credentials=creds)
        
        # Get default task list
        lists = tasks_service.tasklists().list().execute()
        list_id = lists['items'][0]['id'] if 'items' in lists else None
        
        if not list_id:
            return "No task lists found. Please create a task list first."
        
        # Prepare the task data
        task_data = {
            'title': title,
            'notes': notes
        }
        
        if due_date:
            # Convert natural language dates to ISO format
            parsed_date = parse_natural_date(due_date)
            if parsed_date:
                task_data['due'] = parsed_date.isoformat() + 'Z'  # UTC time
        
        # Create the task
        result = tasks_service.tasks().insert(tasklist=list_id, body=task_data).execute()
        
        # Mark the email as read and add a label if possible
        try:
            gmail_service.users().messages().modify(
                userId="me",
                id=email_id,
                body={"removeLabelIds": ["UNREAD"], "addLabelIds": ["STARRED"]}
            ).execute()
        except Exception as label_error:
            print(f"Warning: Could not modify email labels: {label_error}")
        
        return f"Task '{title}' has been created from the email. The email has been marked as read and starred."
        
    except Exception as e:
        print(f"Error creating task from email: {e}")
        traceback.print_exc()
        return f"Sorry, I encountered an error while trying to create a task from the email: {str(e)}"

def create_task_for_event(event_id, task_type="both", days_before=1, days_after=1, custom_titles=None):
    """
    Create preparation or follow-up tasks for a calendar event
    """
    try:
        # Get event details
        calendar_service = build("calendar", "v3", credentials=creds)
        
        event = calendar_service.events().get(calendarId='primary', eventId=event_id).execute()
        
        event_title = event.get('summary', 'Unnamed event')
        
        # Parse event start time
        if 'dateTime' in event['start']:
            start_time = datetime.fromisoformat(event['start']['dateTime'].replace('Z', '+00:00'))
        else:
            # All-day event
            start_time = datetime.fromisoformat(event['start']['date'])
        
        # Get tasks service
        tasks_service = build('tasks', 'v1', credentials=creds)
        
        # Get default task list
        lists = tasks_service.tasklists().list().execute()
        list_id = lists['items'][0]['id'] if 'items' in lists else None
        
        if not list_id:
            return "No task lists found. Please create a task list first."
        
        tasks_created = []
        
        # Create preparation tasks
        if task_type in ["preparation", "both"]:
            prep_due_date = start_time - timedelta(days=days_before)
            
            if custom_titles and len(custom_titles) > 0:
                prep_title = custom_titles[0]
            else:
                prep_title = f"Prepare for: {event_title}"
            
            # Create notes with event details
            notes = f"Event: {event_title}\n"
            notes += f"Date: {start_time.strftime('%B %d, %Y at %I:%M %p')}\n"
            notes += f"Calendar Event ID: {event_id}\n\n"
            
            if 'description' in event and event['description']:
                notes += f"Event Description: {event['description'][:200]}...\n\n"
                
            if 'location' in event and event['location']:
                notes += f"Location: {event['location']}\n"
                
            if 'attendees' in event and event['attendees']:
                attendees = ", ".join([attendee.get('email', '') for attendee in event['attendees'][:5]])
                notes += f"Attendees: {attendees}"
                if len(event['attendees']) > 5:
                    notes += f" and {len(event['attendees']) - 5} more"
            
            # Prepare the task data
            task_data = {
                'title': prep_title,
                'notes': notes,
                'due': prep_due_date.isoformat() + 'Z'  # UTC time
            }
            
            # Create the task
            prep_task = tasks_service.tasks().insert(tasklist=list_id, body=task_data).execute()
            tasks_created.append(prep_task['title'])
        
        # Create follow-up tasks
        if task_type in ["follow_up", "both"]:
            followup_due_date = start_time + timedelta(days=days_after)
            
            if custom_titles and len(custom_titles) > 1:
                followup_title = custom_titles[1]
            else:
                followup_title = f"Follow up on: {event_title}"
            
            # Create notes with event details
            notes = f"Follow-up for event: {event_title}\n"
            notes += f"Original Date: {start_time.strftime('%B %d, %Y at %I:%M %p')}\n"
            notes += f"Calendar Event ID: {event_id}\n\n"
            
            if 'attendees' in event and event['attendees']:
                attendees = ", ".join([attendee.get('email', '') for attendee in event['attendees'][:5]])
                notes += f"Attendees: {attendees}"
                if len(event['attendees']) > 5:
                    notes += f" and {len(event['attendees']) - 5} more"
            
            # Prepare the task data
            task_data = {
                'title': followup_title,
                'notes': notes,
                'due': followup_due_date.isoformat() + 'Z'  # UTC time
            }
            
            # Create the task
            followup_task = tasks_service.tasks().insert(tasklist=list_id, body=task_data).execute()
            tasks_created.append(followup_task['title'])
        
        # Return success message
        if len(tasks_created) == 1:
            return f"Task '{tasks_created[0]}' has been created for the event."
        else:
            return f"Tasks have been created for the event: {', '.join(tasks_created)}."
        
    except Exception as e:
        print(f"Error creating tasks for event: {e}")
        traceback.print_exc()
        return f"Sorry, I encountered an error while trying to create tasks for the event: {str(e)}"

def parse_natural_date(date_str):
    """
    Parse natural language date strings into datetime objects
    """
    try:
        # Try simple cases first
        now = datetime.now()
        
        if date_str.lower() == "today":
            return datetime.combine(now.date(), datetime.min.time())
            
        elif date_str.lower() == "tomorrow":
            return datetime.combine(now.date() + timedelta(days=1), datetime.min.time())
            
        elif date_str.lower() == "next week":
            # Next Monday
            days_ahead = 7 - now.weekday()
            return datetime.combine(now.date() + timedelta(days=days_ahead), datetime.min.time())
            
        # Try to parse as ISO format or other common formats
        try:
            if "T" in date_str:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                # Try common formats
                formats = [
                    "%Y-%m-%d",
                    "%m/%d/%Y",
                    "%B %d, %Y",
                    "%b %d, %Y",
                    "%d %B %Y",
                    "%d %b %Y"
                ]
                
                for fmt in formats:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
        except:
            pass
        
        # Fall back to more sophisticated parsing if needed
        # In a production environment, you might use libraries like dateparser
        # For this implementation, we'll just handle the most common cases
        
        return None
        
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
        return None

def manage_contacts(action, name=None, email=None, phone=None, company=None, 
                   relationship=None, importance=None, query=None):
    """
    Manage contacts using Google People API with enhanced metadata
    """
    try:
        # Build the People API service
        service = build('people', 'v1', credentials=creds)
        
        # Load custom metadata
        metadata = load_contact_metadata()
        
        if action == "create":
            if not name or not email:
                return "Name and email are required to create a contact."
                
            # Create the contact in Google
            contact_body = {
                "names": [{"givenName": name}],
                "emailAddresses": [{"value": email}]
            }
            
            if phone:
                contact_body["phoneNumbers"] = [{"value": phone}]
                
            if company:
                contact_body["organizations"] = [{"name": company}]
            
            result = service.people().createContact(
                body=contact_body
            ).execute()
            
            # Store custom metadata
            if relationship or importance:
                metadata[email] = {
                    "relationship": relationship,
                    "importance": importance
                }
                save_contact_metadata(metadata)
            
            return f"Contact for {name} ({email}) has been created successfully."
            
        elif action == "search":
            if not query:
                return "Search query is required."
                
            # Search Google Contacts
            results = service.people().searchContacts(
                query=query,
                readMask="names,emailAddresses,phoneNumbers,organizations"
            ).execute()
            
            connections = results.get("results", [])
            
            if not connections:
                return f"No contacts found matching '{query}'."
                
            # Format results for voice response
            response = f"I found {len(connections)} contacts matching '{query}':\n\n"
            
            for i, person in enumerate(connections, 1):
                person_data = person.get("person", {})
                
                # Extract name
                names = person_data.get("names", [])
                name = names[0].get("displayName", "Unnamed") if names else "Unnamed"
                
                # Extract email
                emails = person_data.get("emailAddresses", [])
                email = emails[0].get("value", "No email") if emails else "No email"
                
                # Extract company
                orgs = person_data.get("organizations", [])
                company = orgs[0].get("name", "") if orgs else ""
                
                # Get custom metadata
                meta = metadata.get(email, {})
                importance = meta.get("importance", "")
                relationship = meta.get("relationship", "")
                
                # Format entry
                response += f"{i}. {name} - {email}\n"
                if company:
                    response += f"   Company: {company}\n"
                if relationship:
                    response += f"   Relationship: {relationship}\n"
                if importance:
                    response += f"   Importance: {importance}\n"
                response += "\n"
            
            return response
            
        elif action == "list":
            # List contacts from Google
            results = service.people().connections().list(
                resourceName='people/me',
                pageSize=100,
                personFields='names,emailAddresses,organizations'
            ).execute()
            
            connections = results.get('connections', [])
            
            if not connections:
                return "You don't have any contacts saved."
                
            # Group by importance if available
            important_contacts = []
            regular_contacts = []
            
            for person in connections:
                # Extract name
                names = person.get("names", [])
                name = names[0].get("displayName", "Unnamed") if names else "Unnamed"
                
                # Extract email
                emails = person.get("emailAddresses", [])
                email = emails[0].get("value", "") if emails else ""
                
                if not email:
                    continue
                
                # Get metadata
                meta = metadata.get(email, {})
                importance = meta.get("importance", "")
                
                contact_info = {
                    "name": name,
                    "email": email,
                    "importance": importance
                }
                
                if importance == "high":
                    important_contacts.append(contact_info)
                else:
                    regular_contacts.append(contact_info)
            
            # Format response
            response = ""
            
            if important_contacts:
                response += f"You have {len(important_contacts)} important contacts:\n\n"
                for contact in important_contacts:
                    response += f"- {contact['name']} ({contact['email']})\n"
                response += "\n"
                
            response += f"You have {len(regular_contacts)} other contacts."
            
            if len(connections) > 10:
                response += " You can ask me to search for specific contacts if needed."
            
            return response
            
        elif action == "get":
            if not email:
                return "Email address is required to get contact details."
                
            # Search for the contact
            results = service.people().searchContacts(
                query=email,
                readMask="names,emailAddresses,phoneNumbers,organizations,addresses"
            ).execute()
            
            connections = results.get("results", [])
            
            if not connections:
                return f"No contact found with email '{email}'."
                
            person_data = connections[0].get("person", {})
            
            # Extract details
            names = person_data.get("names", [])
            name = names[0].get("displayName", "Unnamed") if names else "Unnamed"
            
            phones = person_data.get("phoneNumbers", [])
            phone = phones[0].get("value", "No phone") if phones else "No phone"
            
            orgs = person_data.get("organizations", [])
            company = orgs[0].get("name", "No company") if orgs else "No company"
            title = orgs[0].get("title", "") if orgs else ""
            
            addresses = person_data.get("addresses", [])
            address = addresses[0].get("formattedValue", "No address") if addresses else "No address"
            
            # Get metadata
            meta = metadata.get(email, {})
            importance = meta.get("importance", "")
            relationship = meta.get("relationship", "")
            
            # Format response
            response = f"Contact details for {name}:\n\n"
            response += f"Email: {email}\n"
            response += f"Phone: {phone}\n"
            if company:
                response += f"Company: {company}\n"
            if title:
                response += f"Title: {title}\n"
            if relationship:
                response += f"Relationship: {relationship}\n"
            if importance:
                response += f"Importance: {importance}\n"
            if address != "No address":
                response += f"Address: {address}\n"
            
            return response
            
        elif action == "update":
            if not email:
                return "Email address is required to update a contact."
                
            # Search for the contact first
            results = service.people().searchContacts(
                query=email,
                readMask="names,emailAddresses,phoneNumbers,organizations,addresses"
            ).execute()
            
            connections = results.get("results", [])
            
            if not connections:
                return f"No contact found with email '{email}'."
                
            person_data = connections[0].get("person", {})
            resource_name = person_data.get("resourceName")
            
            # Prepare fields to update
            update_person_fields = []
            contact_body = {}
            
            if name:
                contact_body["names"] = [{"givenName": name}]
                update_person_fields.append("names")
                
            if phone:
                contact_body["phoneNumbers"] = [{"value": phone}]
                update_person_fields.append("phoneNumbers")
                
            if company:
                contact_body["organizations"] = [{"name": company}]
                update_person_fields.append("organizations")
            
            # Update Google contact if we have standard fields
            if update_person_fields:
                result = service.people().updateContact(
                    resourceName=resource_name,
                    updatePersonFields=','.join(update_person_fields),
                    body=contact_body
                ).execute()
            
            # Update metadata
            updated_meta = False
            if email not in metadata:
                metadata[email] = {}
                
            if relationship:
                metadata[email]["relationship"] = relationship
                updated_meta = True
                
            if importance:
                metadata[email]["importance"] = importance
                updated_meta = True
                
            if updated_meta:
                save_contact_metadata(metadata)
            
            return f"Contact information for {name or email} has been updated successfully."
        
        else:
            return f"Unknown action: {action}. Please specify 'get', 'create', 'update', 'list', or 'search'."
            
    except Exception as e:
        print(f"Error managing contacts: {e}")
        traceback.print_exc()
        return f"Sorry, I encountered an error while trying to manage contacts: {str(e)}"

def load_contact_metadata():
    """Load custom contact metadata from JSON file"""
    try:
        metadata_file = "contacts_metadata.json"
        if not os.path.exists(metadata_file):
            # Create empty metadata file if it doesn't exist
            with open(metadata_file, "w") as f:
                json.dump({}, f)
            return {}
            
        with open(metadata_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading contact metadata: {e}")
        return {}
        
def save_contact_metadata(metadata):
    """Save custom contact metadata to JSON file"""
    try:
        with open("contacts_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error saving contact metadata: {e}")

async def generate_response(query):
    global chat_log, last_interaction, last_interaction_check, token_tracker
    
    now = datetime.now()    
    last_interaction = now
    last_interaction_check = now
    
    if (now - last_interaction).total_seconds() > CONVERSATION_END_SECONDS:    
        chat_log = []    
        await display_manager.update_display('sleep')
    
    # Add the new query to chat log
    chat_log.append({    
        "role": "user",
        "content": query
    })

    try:
        # Update token counts using TokenTracker
        token_tracker.update_query_tokens(
            query=query,
            chat_history=chat_log[:-1],
            using_tools=True
        )
        print("DEBUG: Token tracking updated for query")  # Debug print
    
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            system=SYSTEM_PROMPT,
            messages=chat_log,
            max_tokens=1024,
            temperature=0.7,
            tools=TOOLS,
            tool_choice={"type": "auto"}
        )
        
        # Check for errors first
        if hasattr(response, 'error'):    
            error_msg = f"Sorry, there was an error: {response.error}"
            print(f"API Error: {response.error}")
            await display_manager.update_display('speaking', mood='neutral')
            return error_msg
            
        # Handle tool use case
        if response.stop_reason == "tool_use":
            # Add tool overhead tokens
            token_tracker.update_query_tokens(
                query="",
                chat_history=chat_log,
                using_tools=True
            )
            
            print("DEBUG: Tool use detected!")
            await display_manager.update_display('tools')
            tool_audio = get_random_audio('tool')
            if tool_audio:
                await audio_manager.play_audio(tool_audio)
            else:
                print("WARNING: No tool audio files found, skipping audio")
            
            chat_log.append({
                "role": "assistant",
                "content": [block.model_dump() for block in response.content],
            })
            
            tool_results = []
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_call = content_block
                    print(f"Processing tool call: {tool_call.name}")
                    print(f"DEBUG: Tool args: {tool_call.input}")
        
                    tool_args = tool_call.input
        
                    try:
                        if tool_call.name == "draft_email":
                            tool_response = draft_email(**tool_args)
                        elif tool_call.name == "read_emails":
                            tool_response = read_emails(**tool_args)
                        elif tool_call.name == "email_action":
                            tool_response = email_action(**tool_args)
                        elif tool_call.name == "manage_tasks":
                            tool_response = manage_tasks(**tool_args)
                        elif tool_call.name == "create_task_from_email":
                            tool_response = create_task_from_email(**tool_args)
                        elif tool_call.name == "create_task_for_event":
                            tool_response = create_task_for_event(**tool_args)
                        elif tool_call.name == "update_calendar_event":
                            # Check if we need to find the event first
                            if "event_description" in tool_args and not tool_args.get("event_id"):
                                matching_events = find_calendar_event(tool_args["event_description"])
                                if matching_events:
                                    if len(matching_events) == 1:
                                        # Single match, use it
                                        tool_args["event_id"] = matching_events[0]["id"]
                                        # Remove the description parameter
                                        tool_args.pop("event_description", None)
                                        tool_response = update_calendar_event(**tool_args)
                                    else:
                                        # Multiple matches, return them for disambiguation
                                        event_list = "\n".join([
                                            f"{i+1}. {event['summary']} ({event['start_formatted']})"
                                            for i, event in enumerate(matching_events)
                                        ])
                                        tool_response = f"I found multiple matching events:\n\n{event_list}\n\nPlease specify which event you'd like to update."
                                else:
                                    tool_response = "I couldn't find any calendar events matching that description."
                            else:
                                tool_response = update_calendar_event(**tool_args)
                        elif tool_call.name == "cancel_calendar_event":
                            # Similar pattern for cancellation
                            if "event_description" in tool_args and not tool_args.get("event_id"):
                                matching_events = find_calendar_event(tool_args["event_description"])
                                if matching_events:
                                    if len(matching_events) == 1:
                                        tool_args["event_id"] = matching_events[0]["id"]
                                        tool_args.pop("event_description", None)
                                        tool_response = cancel_calendar_event(**tool_args)
                                    else:
                                        event_list = "\n".join([
                                            f"{i+1}. {event['summary']} ({event['start_formatted']})"
                                            for i, event in enumerate(matching_events)
                                        ])
                                        tool_response = f"I found multiple matching events:\n\n{event_list}\n\nPlease specify which event you'd like to cancel."
                                else:
                                    tool_response = "I couldn't find any calendar events matching that description."
                            else:
                                tool_response = cancel_calendar_event(**tool_args)
                        elif tool_call.name == "calendar_query":
                            if tool_args["query_type"] == "next_event":
                                tool_response = get_next_event()
                            elif tool_args["query_type"] == "day_schedule":
                                tool_response = get_day_schedule()
                            else:
                                tool_response = "Unsupported query type"
                        elif tool_call.name == "calibrate_voice_detection":
                            if tool_args.get("confirm", False):
                                calibration_success = await run_vad_calibration()
                    
                                if calibration_success:
                                    tool_response = "Voice detection calibration completed successfully. Your microphone settings have been optimized."
                                else:
                                    tool_response = "The voice detection calibration encountered an issue. You might want to try again."
                            else:
                                tool_response = "Voice detection calibration was not confirmed."
                        elif tool_call.name == "create_calendar_event":
                            tool_response = create_calendar_event(**tool_args)
                        elif tool_call.name == "get_location":
                            tool_response = get_location(**tool_args)
                        elif tool_call.name == "get_current_time":
                            tool_response = get_current_time(**tool_args)
                        else:
                            tool_response = "Unsupported tool called"
                    except Exception as e:
                        print(f"DEBUG: Tool execution error: {e}")
                        tool_response = f"Error executing tool: {str(e)}"
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": tool_response
                    })
                    
                    print(f"\nFinished tool call: {tool_call.name}")
            
            if tool_results:
                # Update token counts for tool results
                token_tracker.update_response_tokens(
                    tool_results,
                    is_tool_result=True
                )
                chat_log.append({
                    "role": "user",
                    "content": tool_results
                })
            
                # Make final API call with tool results
                final_response = anthropic_client.messages.create(
                    model=ANTHROPIC_MODEL,
                    system=SYSTEM_PROMPT,
                    messages=chat_log,
                    max_tokens=1024,
                    temperature=0.7,
                    tools=TOOLS,
                    tool_choice={"type": "auto"}
                )
            
                if hasattr(final_response, 'error'):
                    error_msg = f"Sorry, there was an error processing the tool response: {final_response.error}"
                    print(f"API Error in tool response: {final_response.error}")
                    await display_manager.update_display('speaking', mood='neutral')
                    return error_msg
            
                # Update token counts for final response
                if final_response.content:
                    token_tracker.update_response_tokens(
                        [block.model_dump() for block in final_response.content]
                    )
            
                token_tracker.print_metrics("After tool use")
            
                return await process_response_content(final_response.content)
        
        else:
            # Handle normal response path (no tool use)
            if response.content:
                token_tracker.update_response_tokens(
                    [block.model_dump() for block in response.content]
                )
            
            token_tracker.print_metrics("After response")
            
            return await process_response_content(response.content)

    except (APIError, APIConnectionError, BadRequestError, InternalServerError) as e:
        error_msg = ("I apologize, but the service is temporarily overloaded. Please try again in a moment." 
                    if "overloaded" in str(e).lower() else f"Sorry, there was a server error: {str(e)}")
        print(f"Anthropic API Error: {e}")
        await display_manager.update_display('speaking', mood='neutral')
        return error_msg
    except Exception as e:
        error_msg = f"Sorry, there was an unexpected error: {str(e)}"
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        await display_manager.update_display('speaking', mood='neutral')
        return error_msg

async def process_response_content(content):
    """Helper function to process response content and handle mood/display updates"""
    for content_block in content:
        if content_block.type == "text":
            text = content_block.text
            mood_match = re.match(r'^\[(neutral|happy|confused|disappointed|annoyed|surprised)\]', 
                                text, re.IGNORECASE)

            if mood_match:
                mood, message = mood_match.group(1), text[mood_match.end():].strip()
                mood = mood.lower()
                await display_manager.update_display('speaking', mood=mood)
                chat_log.append({"role": "assistant", "content": message})
                return message
            else:
                await display_manager.update_display('speaking', mood='neutral')
                chat_log.append({"role": "assistant", "content": text})
                return text

    await display_manager.update_display('speaking', mood='neutral')
    return "No text response found"

async def wake_word():
    """Wake word detection with context-aware breaks"""
    global last_interaction_check, last_interaction
    try:
        snowboy = None
        wake_pa = None
        wake_audio_stream = None
        
        # Track time since last break
        last_break_time = time.time()
        BREAK_INTERVAL = 30  # Take break every 30 seconds
        BREAK_DURATION = 1   # Reduced from 5 to 1 second
        
        try:
            print(f"{Fore.YELLOW}Initializing wake word detector...{Fore.WHITE}")
            
            # Keep original resource path code exactly as it was
            resource_path = Path("resources/common.res")
            model_paths = [Path("GD_Laura.pmdl"), Path("Wake_up_Laura.pmdl"), Path("Laura.pmdl")]
            
            if not resource_path.exists():
                print(f"ERROR: Resource file not found at {resource_path.absolute()}")
                return False
                
            for model_path in model_paths:
                if not model_path.exists():
                    print(f"ERROR: Model file not found at {model_path.absolute()}")
                    return False
            
            snowboy = snowboydetect.SnowboyDetect(
                resource_filename=str(resource_path.absolute()).encode(),
                model_str=",".join(str(p.absolute()) for p in model_paths).encode()
            )
            
            snowboy.SetSensitivity(b"0.5,0.5,0.45")
            
            wake_pa = pyaudio.PyAudio()
            wake_audio_stream = wake_pa.open(
                rate=16000,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=2048
            )
            
            while True:
                current_time = time.time()
                
                # MODIFIED: Context-aware break logic
                now = datetime.now()
                time_since_interaction = (now - last_interaction).total_seconds()
                
                # Only take breaks when system is idle and not during active interactions
                if (current_time - last_break_time >= BREAK_INTERVAL and 
                    time_since_interaction > 10 and  # Only break if no interaction in last 10 seconds
                    display_manager.current_state in ['idle', 'sleep']):  # Only break in idle states
                    
                    last_break_time = current_time
                    
                    # Shorter break duration
                    wake_audio_stream.stop_stream()
                    await asyncio.sleep(BREAK_DURATION)
                    wake_audio_stream.start_stream()
                    continue
                
                data = wake_audio_stream.read(2048, exception_on_overflow=False)
                if len(data) == 0:
                    print("Warning: Empty audio frame received")
                    continue
                
                result = snowboy.RunDetection(data)
                
                if result > 0:
                    print(f"{Fore.GREEN}Wake word detected! (Model {result}){Fore.WHITE}")
                    last_interaction = datetime.now()
                    last_interaction_check = datetime.now()
                    return True
                
                if random.random() < 0.01:
                    await asyncio.sleep(0)
                    
                    now = datetime.now()
                    if (now - last_interaction_check).total_seconds() > CONVERSATION_END_SECONDS:
                        last_interaction_check = now
                        if display_manager.current_state != 'sleep':
                            await display_manager.update_display('sleep')
                            
        finally:
            if wake_audio_stream:
                wake_audio_stream.stop_stream()
                wake_audio_stream.close()
            if wake_pa:
                wake_pa.terminate()
                
    except Exception as e:
        print(f"Error in wake word detection: {e}")
        traceback.print_exc()
        return False

async def handle_voice_query():
    try:
        print(f"{Fore.BLUE}Listening...{Fore.WHITE}")
        
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
                max_recording_time = VAD_SETTINGS["max_recording_time"]

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
                    if not voice_detected and (time.time() - start_time) > VOICE_START_TIMEOUT:
                        print(f"DEBUG: Voice not detected - energy: {avg_energy:.6f}, threshold: {energy_threshold:.6f}")
                        print(f"{Fore.YELLOW}Voice start timeout{Fore.WHITE}")
                        break
    
                    # Read audio frame
                    pcm_bytes = audio_manager.read_audio_frame()
                    if not pcm_bytes:
                        await asyncio.sleep(0.01)
                        continue

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

                    # Display partial results only occasionally
                    current_time = time.time()
                    if partial_text and (current_time - last_partial_time) > 2:
                        last_partial_time = current_time
                        print(f"Partial: {partial_text}")

                    # VAD STATE MACHINE
                    if avg_energy > energy_threshold and not is_speaking:
                        # Speech just started
                        print(f"{Fore.BLUE}Voice detected (energy: {avg_energy:.6f}, threshold: {energy_threshold:.6f}){Fore.WHITE}")
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
                            print(f"{Fore.MAGENTA}End of speech detected (duration: {speech_duration:.2f}s){Fore.WHITE}")
                            await asyncio.sleep(speech_buffer_time)
                            break

                        # Check for maximum duration
                        if speech_duration > max_recording_time:
                            print(f"{Fore.RED}Maximum recording time reached{Fore.WHITE}")
                            break

                    # Reduced CPU usage
                    await asyncio.sleep(0.01)

                # Check if we actually detected voice
                if not voice_detected:
                    print("No voice detected")
                    return None

                # Get final transcription
                transcript = transcriber.get_final_text()
                print(f"Raw transcript: '{transcript}'")

                # Apply cleanup for common Vosk errors
                if transcript:
                    # Fix "that were" at beginning which is a common Vosk error
                    transcript = re.sub(r'^that were\s+', '', transcript)
                    transcript = re.sub(r'^that was\s+', '', transcript)

                    # Reject single-word responses only if they're likely noise
                    if len(transcript.split()) <= 1 and len(transcript) < 4:
                        print(f"Discarding too-short transcript: '{transcript}'")
                        return None
                
            else:
                # Handle Whisper transcription
                recording_complete = False
                is_speech = False
                
                while True:
                    # Read audio frame
                    pcm_bytes = audio_manager.read_audio_frame()
                    if not pcm_bytes:
                        await asyncio.sleep(0.01)
                        continue
                    
                    # Process with WhisperCpp VAD
                    recording_complete, is_speech = transcriber.process_frame(pcm_bytes)
                    
                    # Calculate energy level for debugging
                    float_data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    energy = np.sqrt(np.mean(float_data**2)) if len(float_data) > 0 else 0
                    
                    if is_speech and not voice_detected:
                        print(f"{Fore.BLUE}Voice detected (energy: {energy:.6f}){Fore.WHITE}")
                        voice_detected = True
                        start_time = time.time()  # Reset timeout
                    
                    if recording_complete:
                        print(f"{Fore.MAGENTA}End of speech detected{Fore.WHITE}")
                        break
                        
                    # Check timeout if we're still waiting for voice
                    if not voice_detected and (time.time() - start_time) > VOICE_START_TIMEOUT:
                        print("Voice start timeout")
                        break
                
                if not voice_detected:
                    print("No voice detected")
                    return None
                
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
            
            # Initial detection phase
            print(f"{Fore.MAGENTA}Waiting for voice...{Fore.WHITE}")
            while (time.time() - start_time) < VOICE_START_TIMEOUT:
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
                    print(f"{Fore.BLUE}Voice detected (energy: {energy:.6f}, threshold: {energy_threshold:.6f}){Fore.WHITE}")
                    voice_detected = True
                    recording.extend(pcm)
                    break
                
            # If no voice detected in initial phase, return
            if not voice_detected:
                print("No voice detected in initial phase")
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
                    print(f"{Fore.MAGENTA}Silence detected, ending recording (duration: {current_length:.2f}s){Fore.WHITE}")
                    break
                elif current_length >= VAD_SETTINGS["max_recording_time"]:
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
            
        # Prevent rejection of valid short phrases
        if len(transcript.strip()) > 0:
            print(f"Final transcript: '{transcript}'")
            return transcript.strip()
        else:
            print("Empty transcript after processing")
            return None

    finally:
        await audio_manager.stop_listening()

async def conversation_mode():
    try:
        # Ensure audio playback is complete before starting listening
        await audio_manager.wait_for_audio_completion()
        
        # Add a small buffer delay before activating the microphone
        await asyncio.sleep(0.5)  
        
        if TRANSCRIPTION_MODE == "local":
            # Reset the transcriber state
            transcriber.reset()
            
            # Start listening
            audio_stream, _ = await audio_manager.start_listening()
            print(f"\n{Fore.MAGENTA}Waiting for response...{Fore.WHITE}")
            
            # Initial timeout for user to start speaking
            initial_timeout = 4.0

            # VOSK-specific conversation handling
            if TRANSCRIPTION_ENGINE == "vosk":
                # Get VAD settings
                energy_threshold = VAD_SETTINGS["energy_threshold"]
                continued_ratio = VAD_SETTINGS["continued_threshold_ratio"]
                silence_duration = VAD_SETTINGS["silence_duration"]
                min_speech_duration = VAD_SETTINGS["min_speech_duration"]
                speech_buffer_time = VAD_SETTINGS["speech_buffer_time"]
                max_recording_time = 30.0  # Shorter max time for conversation mode

                # Calculate frames needed for silence duration
                max_silence_frames = int(silence_duration * 16000 / audio_manager.frame_length)

                # State tracking
                silence_frames = 0
                last_partial_time = time.time()
                frame_history = []
                is_speaking = False
                voice_detected = False
                speech_start_time = None
                start_time = time.time()

                while True:
                    # Check for initial timeout
                    if not voice_detected and (time.time() - start_time) > initial_timeout:
                        print(f"{Fore.YELLOW}No response detected{Fore.WHITE}")
                        return None
    
                    # Read audio frame
                    pcm_bytes = audio_manager.read_audio_frame()
                    if not pcm_bytes:
                        await asyncio.sleep(0.01)
                        continue

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

                    # Display partial results less frequently
                    current_time = time.time()
                    if partial_text and (current_time - last_partial_time) > 2:
                        last_partial_time = current_time
                        print(f"Partial: {partial_text}")

                    # VAD STATE MACHINE
                    if avg_energy > energy_threshold and not is_speaking:
                        print(f"{Fore.GREEN}Voice detected{Fore.WHITE}")
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
                            print(f"{Fore.MAGENTA}End of speech detected{Fore.WHITE}")
                            await asyncio.sleep(speech_buffer_time)
                            break

                        # Check for maximum duration
                        if speech_duration > max_recording_time:
                            print(f"{Fore.RED}Maximum recording time reached{Fore.WHITE}")
                            break

                    # Reduced CPU usage
                    await asyncio.sleep(0.01)

                # Check if we actually detected voice
                if not voice_detected:
                    return None

                # Get final transcription
                transcript = transcriber.get_final_text()

                # Apply cleanup for common Vosk errors
                if transcript:
                    # Fix "that were" at beginning which is a common Vosk error
                    transcript = re.sub(r'^that were\s+', '', transcript)
                    transcript = re.sub(r'^that was\s+', '', transcript)

                    # Reject single-word responses as likely noise
                    if len(transcript.split()) <= 1:
                        print(f"Discarding too-short transcript: '{transcript}'")
                        return None

            else:
                # Whisper-specific conversation handling
                recording_complete = False
                voice_detected = False
                start_time = time.time()
                
                while not recording_complete:
                    if not voice_detected and (time.time() - start_time > initial_timeout):
                        print(f"{Fore.YELLOW}No response detected{Fore.WHITE}")
                        break
                    
                    pcm = audio_manager.read_audio_frame()
                    if not pcm:
                        await asyncio.sleep(0.01)
                        continue
                    
                    recording_complete, is_speech = transcriber.process_frame(pcm)
                    
                    if is_speech and not voice_detected:
                        print(f"{Fore.GREEN}Voice detected{Fore.WHITE}")
                        voice_detected = True
                
                print(f"{Fore.MAGENTA}Transcribing conversation...{Fore.WHITE}")
                transcript = transcriber.transcribe()
        
        else:  # Remote transcription mode
            # Start listening
            audio_stream, _ = await audio_manager.start_listening()
            print(f"\n{Fore.MAGENTA}Waiting for response...{Fore.WHITE}")
            
            # Get VAD settings
            energy_threshold = VAD_SETTINGS["energy_threshold"]
            continued_ratio = VAD_SETTINGS["continued_threshold_ratio"]
            silence_duration = VAD_SETTINGS["silence_duration"]
            
            recording = []
            start_time = time.time()
            voice_detected = False
            silence_frames = 0
            silence_frame_threshold = int(silence_duration * audio_manager.sample_rate / audio_manager.frame_length)
    
            while True:
                # Check for initial timeout
                if not voice_detected and (time.time() - start_time) > initial_timeout:
                    print(f"{Fore.YELLOW}No response detected{Fore.WHITE}")
                    return None
                
                # Read audio frame    
                pcm_bytes = audio_manager.read_audio_frame()
                if not pcm_bytes:
                    await asyncio.sleep(0.01)
                    continue
                
                # Convert bytes to int16 values
                pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
                
                # Calculate energy
                float_data = pcm.astype(np.float32) / 32768.0
                energy = np.sqrt(np.mean(float_data**2)) if len(float_data) > 0 else 0
                
                # If not yet speaking, check if this is the start of speech
                if not voice_detected:
                    if energy > energy_threshold:
                        print(f"{Fore.GREEN}Voice detected{Fore.WHITE}")
                        voice_detected = True
                        recording.extend(pcm)
                    continue
                
                # We're already recording, so add this frame
                recording.extend(pcm)
                
                # Check if this frame has speech
                if energy > (energy_threshold * continued_ratio):
                    silence_frames = 0
                else:
                    silence_frames += 1
                
                # End recording conditions
                if silence_frames >= silence_frame_threshold:
                    print(f"{Fore.MAGENTA}End of response detected{Fore.WHITE}")
                    break
    
            if recording and voice_detected:
                audio_array = np.array(recording, dtype=np.float32) / 32768.0
                transcript = await remote_transcriber.transcribe(audio_array)
            else:
                transcript = None
        
        # Common post-processing for both methods
        if transcript:
            print(f"\n{Style.BRIGHT}You said:{Style.NORMAL} {transcript}\n")
            return transcript
        return None

    finally:
        await audio_manager.stop_listening()

def draft_email(subject: str, content: str, recipient: str = "") -> str:
    global creds
    if not USE_GOOGLE:
        return "Please let the user know that Google is turned off in the script."
    try:
        service = build("gmail", "v1", credentials=creds)
        message = EmailMessage()
        message.set_content(content)
        if recipient:
            message["To"] = recipient
        message["Subject"] = subject
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {"message": {"raw": encoded_message}}
        draft = (
            service.users()
            .drafts()
            .create(userId="me", body=create_message)
            .execute()
        )
        if not draft or "message" not in draft:
            print(draft)
            raise ValueError("The request returned an invalid response. Check the console logs.")
        return "Please let the user know that the email has been drafted successfully."
    except HttpError as error:
        print(traceback.format_exc())
        return f"Please let the user know that there was an error trying to draft an email. The error is: {error}"

async def print_response(chat):
    """Print response before voice generation"""
    wrapper = textwrap.TextWrapper(width=70)
    paragraphs = chat.split('\n')
    wrapped_chat = "\n".join([wrapper.fill(p) for p in paragraphs if p.strip()])
    print(wrapped_chat)

async def generate_voice(chat):
    try:
        # Check if message still has mood prefix and strip it if present
        mood_pattern = r'^\[(.*?)\]'  # Only match at start of string
        mood_match = re.match(mood_pattern, chat, re.IGNORECASE)
        
        if mood_match:
            # Just strip the mood prefix without updating display
            _, message = mood_match.groups()
            chat = message.strip()

        # Debug print to verify full message
        #print(f"Generating voice for text: {chat}")

        audio = tts_handler.generate_audio(chat)

        with open(AUDIO_FILE, "wb") as f:
            f.write(audio)
    
        await audio_manager.play_audio(AUDIO_FILE)

    except Exception as e:
        print(f"Error in generate_voice: {e}")
        traceback.print_exc()

def get_location(format: str) -> str:
    try:
        # Get WiFi access points
        import subprocess
        
        # For Linux/Raspberry Pi
        cmd = "iwlist wlan0 scan | grep -E 'Address|Signal|ESSID'"
        output = subprocess.check_output(cmd, shell=True).decode()
        
        # Parse WiFi data
        wifi_data = []
        current_ap = {}
        
        for line in output.split('\n'):
            if 'Address' in line:
                if current_ap:
                    wifi_data.append(current_ap)
                current_ap = {'macAddress': line.split('Address: ')[1].strip()}
            elif 'Signal' in line:
                try:
                    # Handle different signal strength formats
                    signal_part = line.split('=')[1].split(' ')[0]
                    if '/' in signal_part:
                        # Handle format like "70/70"
                        numerator, denominator = map(int, signal_part.split('/'))
                        # Convert to dBm (typical range -100 to 0)
                        signal_level = -100 + (numerator * 100) // denominator
                    else:
                        # Direct dBm value
                        signal_level = int(signal_part)
                    current_ap['signalStrength'] = signal_level
                except Exception as e:
                    print(f"Error parsing signal strength: {e}")
                    current_ap['signalStrength'] = -50  # Default middle value
            elif 'ESSID' in line:
                current_ap['ssid'] = line.split('ESSID:')[1].strip('"')
        
        if current_ap:
            wifi_data.append(current_ap)
        
        if not wifi_data:
            return "No WiFi access points found"

        # Google Geolocation API request
        url = "https://www.googleapis.com/geolocation/v1/geolocate"
        params = {"key": GOOGLE_MAPS_API_KEY}  # Using key from secret.py
        data = {"wifiAccessPoints": wifi_data}
        
        try:
            response = requests.post(url, params=params, json=data)
            response.raise_for_status()  # Raise exception for bad status codes
            location = response.json()
            
            if 'error' in location:
                return f"Google API error: {location['error']['message']}"
                
            lat = location['location']['lat']
            lng = location['location']['lng']
            
            if format == "coordinates":
                return f"Current coordinates are: {lat}, {lng}"
            
            # Get address if needed
            if format in ["address", "both"]:
                geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json"
                params = {
                    "latlng": f"{lat},{lng}",
                    "key": GOOGLE_MAPS_API_KEY
                }
                
                address_response = requests.get(geocode_url, params=params)
                address_response.raise_for_status()
                address_data = address_response.json()
                
                if address_data['status'] == 'OK' and address_data['results']:
                    address = address_data['results'][0]['formatted_address']
                    
                    if format == "address":
                        return f"Current location is: {address}"
                    else:  # both
                        return f"Current location is: {address}\nCoordinates: {lat}, {lng}"
                else:
                    return f"Coordinates found ({lat}, {lng}) but could not determine address"
                    
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Google API: {str(e)}"
                
    except subprocess.CalledProcessError as e:
        return f"Error scanning WiFi networks: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def get_current_time(format: str = "both") -> str:
    """Get the current time in the local timezone"""
    current = datetime.now()
    if format == "time":
        return current.strftime("%I:%M %p")
    elif format == "date":
        return current.strftime("%B %d, %Y")
    else:  # both
        return current.strftime("%I:%M %p on %B %d, %Y")

def get_calendar_service():
    """Helper function to build and return calendar service"""
    try:
        if DEBUG_CALENDAR:
            print("DEBUG: Attempting to build calendar service")
        service = build("calendar", "v3", credentials=creds)
        if DEBUG_CALENDAR:
            print("DEBUG: Calendar service built successfully")
        return service
    except Exception as e:
        if DEBUG_CALENDAR:
            print(f"DEBUG: Error building calendar service: {e}")
        return None

async def check_upcoming_events():
    """Improved notification system with cleaner output"""
    global calendar_notified_events
    
    while True:
        try:
            if audio_manager.is_speaking:
                await audio_manager.wait_for_audio_completion()
            
            service = get_calendar_service()
            if not service:
                await asyncio.sleep(30)
                continue
                
            now = datetime.now(timezone.utc)
            max_minutes = max(CALENDAR_NOTIFICATION_INTERVALS) + 5
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

            for event in events:
                event_id = event['id']
                summary = event.get('summary', 'Unnamed event')
                start = event['start'].get('dateTime', event['start'].get('date'))

                if 'T' in start:
                    start_time = datetime.fromisoformat(start.replace('Z', '+00:00'))
                else:
                    start_time = datetime.strptime(start, '%Y-%m-%d').replace(tzinfo=timezone.utc)

                seconds_until = (start_time - now).total_seconds()
                minutes_until = round(seconds_until / 60)

                for interval in CALENDAR_NOTIFICATION_INTERVALS:
                    notification_key = f"{event_id}_{interval}"
                    
                    if abs(minutes_until - interval) <= 2 and notification_key not in calendar_notified_events:
                        calendar_notified_events.add(notification_key)
                        
                        notification_text = random.choice(CALENDAR_NOTIFICATION_SENTENCES).format(
                            minutes=interval,
                            event=summary
                        )
                        
                        previous_state = display_manager.current_state
                        previous_mood = display_manager.current_mood
                        
                        try:
                            await audio_manager.wait_for_audio_completion()
                            await display_manager.update_display('speaking')
                            
                            audio = tts_handler.generate_audio(notification_text)
                            with open("notification.mp3", "wb") as f:
                                f.write(audio)
                            
                            await audio_manager.play_audio("notification.mp3")
                            await audio_manager.wait_for_audio_completion()
                            await display_manager.update_display(previous_state, mood=previous_mood)
                            
                        except Exception as e:
                            print(f"Error during notification: {e}")
                            await display_manager.update_display(previous_state, mood=previous_mood)

                if minutes_until < -5:
                    for interval in CALENDAR_NOTIFICATION_INTERVALS:
                        calendar_notified_events.discard(f"{event_id}_{interval}")

            await asyncio.sleep(5)

        except Exception as e:
            print(f"Error in calendar check: {e}")
            await asyncio.sleep(5)

def get_day_schedule() -> str:
    """Get all events for today"""
    try:
        service = get_calendar_service()
        if not service:
            return "Failed to initialize calendar service"
            
        now = datetime.now(timezone.utc)
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        events_result = service.events().list(
            calendarId="primary",
            timeMin=start_of_day.isoformat(),
            timeMax=end_of_day.isoformat(),
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        
        events = events_result.get("items", [])
        
        if not events:
            return "No events scheduled for today."
            
        schedule = []
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            start_time = datetime.fromisoformat(start.replace('Z', '+00:00'))
            schedule.append(f"- {start_time.strftime('%I:%M %p')}: {event['summary']}")
            
        return "Today's schedule:\n" + "\n".join(schedule)
        
    except Exception as e:
        if DEBUG_CALENDAR:
            print(f"Error getting day schedule: {e}")
        return f"Error retrieving schedule: {str(e)}"

def update_calendar_event(event_id, summary=None, start_time=None, end_time=None, 
                         description=None, location=None, attendees=None):
    """
    Update an existing calendar event
    """
    try:
        service = build("calendar", "v3", credentials=creds)
        
        # Get the current event
        event = service.events().get(calendarId='primary', eventId=event_id).execute()
        
        # Update fields if provided
        if summary:
            event['summary'] = summary
            
        if description:
            event['description'] = description
            
        if location:
            event['location'] = location
            
        if start_time:
            if 'dateTime' in event['start']:
                event['start']['dateTime'] = start_time
            else:
                # Convert all-day event to timed event
                event['start'] = {'dateTime': start_time, 'timeZone': 'America/Los_Angeles'}
                
        if end_time:
            if 'dateTime' in event['end']:
                event['end']['dateTime'] = end_time
            else:
                # Convert all-day event to timed event
                event['end'] = {'dateTime': end_time, 'timeZone': 'America/Los_Angeles'}
                
        if attendees:
            event['attendees'] = [{'email': email} for email in attendees]
            
        # Update the event
        updated_event = service.events().update(
            calendarId='primary',
            eventId=event_id,
            body=event,
            sendUpdates='all'
        ).execute()
        
        # Format a nice response
        event_time = ""
        if 'dateTime' in updated_event['start']:
            start_dt = datetime.fromisoformat(updated_event['start']['dateTime'].replace('Z', '+00:00'))
            start_time_str = start_dt.strftime('%I:%M %p on %B %d, %Y')
            event_time = f"at {start_time_str}"
        else:
            start_date = datetime.fromisoformat(updated_event['start']['date'])
            event_time = f"on {start_date.strftime('%B %d, %Y')}"
            
        return f"Calendar event '{updated_event['summary']}' {event_time} has been updated successfully."
        
    except Exception as e:
        print(f"Error updating calendar event: {e}")
        traceback.print_exc()
        return f"Sorry, I encountered an error while trying to update the calendar event: {str(e)}"

def cancel_calendar_event(event_id, notify_attendees=True, cancellation_message=None):
    """
    Cancel an existing calendar event
    """
    try:
        service = build("calendar", "v3", credentials=creds)
        
        # Get the event details first for a better response message
        event = service.events().get(calendarId='primary', eventId=event_id).execute()
        event_summary = event.get('summary', 'Unnamed event')
        
        # Add cancellation message if provided
        if cancellation_message and notify_attendees:
            # We can't directly add a cancellation message via the API
            # So we'll update the event description first
            original_description = event.get('description', '')
            event['description'] = f"CANCELLED: {cancellation_message}\n\n{original_description}"
            
            service.events().update(
                calendarId='primary',
                eventId=event_id,
                body=event,
                sendUpdates='all'
            ).execute()
        
        # Delete the event
        service.events().delete(
            calendarId='primary',
            eventId=event_id,
            sendUpdates='all' if notify_attendees else 'none'
        ).execute()
        
        notification_status = "Attendees have been notified" if notify_attendees else "Attendees were not notified"
        return f"Calendar event '{event_summary}' has been cancelled successfully. {notification_status}."
        
    except Exception as e:
        print(f"Error cancelling calendar event: {e}")
        traceback.print_exc()
        return f"Sorry, I encountered an error while trying to cancel the calendar event: {str(e)}"

def find_calendar_event(description, time_range_days=7):
    """Helper function to find calendar events matching a description"""
    try:
        service = build("calendar", "v3", credentials=creds)
        
        # Set time range for search
        now = datetime.now(timezone.utc)
        time_min = now.isoformat()
        time_max = (now + timedelta(days=time_range_days)).isoformat()
        
        # Get events in the time range
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        # Filter events by description
        description = description.lower()
        matching_events = []
        
        for event in events:
            # Check against summary, description and location
            event_summary = event.get('summary', '').lower()
            event_description = event.get('description', '').lower()
            event_location = event.get('location', '').lower()
            
            # Simple fuzzy matching
            if (description in event_summary or 
                description in event_description or 
                description in event_location):
                
                start = event['start'].get('dateTime', event['start'].get('date'))
                
                # Format the start time
                if 'T' in start:  # This is a datetime
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    start_str = start_dt.strftime('%I:%M %p on %B %d, %Y')
                else:  # This is a date
                    start_dt = datetime.fromisoformat(start)
                    start_str = start_dt.strftime('%B %d, %Y')
                
                matching_events.append({
                    'id': event['id'],
                    'summary': event.get('summary', 'Unnamed event'),
                    'start': start,
                    'start_formatted': start_str,
                    'location': event.get('location', 'No location'),
                    'attendees': [attendee.get('email') for attendee in event.get('attendees', [])]
                })
        
        return matching_events
        
    except Exception as e:
        print(f"Error finding calendar events: {e}")
        traceback.print_exc()
        return []

def create_calendar_event(summary: str, start_time: str, end_time: str, 
                        description: str = "", attendees: list = None) -> str:
    """Create a calendar event and send invites"""
    print("DEBUG: Starting create_calendar_event")
    try:
        service = get_calendar_service()
        if not service:
            return "Failed to initialize calendar service"

        # Use your local timezone
        local_timezone = "America/Los_Angeles"  # Adjust this to your timezone

        event_body = {
            "summary": summary,
            "description": description,
            "start": {
                "dateTime": start_time,
                "timeZone": local_timezone
            },
            "end": {
                "dateTime": end_time,
                "timeZone": local_timezone
            }
        }
        
        if attendees:
            event_body["attendees"] = [{"email": email} for email in attendees]
            event_body["sendUpdates"] = "all"

        print(f"DEBUG: Attempting to create event with body: {event_body}")
        
        event = service.events().insert(
            calendarId="primary",
            body=event_body,
            sendUpdates="all"
        ).execute()
        
        return f"Event created successfully: {event.get('htmlLink')}"
    except Exception as e:
        print(f"DEBUG: Error in create_calendar_event: {e}")
        return f"Error creating event: {str(e)}"

async def run_vad_calibration():
    """Run the VAD calibration script and reload settings"""
    global VAD_SETTINGS  # Global declaration at beginning
    
    try:
        # First, wait for any current audio to finish
        await audio_manager.wait_for_audio_completion()
        
        # Path to the calibration script
        calibration_script = os.path.join(os.path.dirname(__file__), "vad_calib5.py")
        
        # Print current settings
        print(f"{Fore.YELLOW}Starting VAD calibration...{Fore.WHITE}")
        print(f"Current VAD settings before calibration: {VAD_SETTINGS}")
        
        # Update display to show we're in calibration mode
        previous_state = display_manager.current_state
        previous_mood = display_manager.current_mood
        await display_manager.update_display('tools')
        
        # Run the calibration script
        process = await asyncio.create_subprocess_exec(
            "python3", calibration_script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        stdout_text = stdout.decode()
        
        if "CALIBRATION_COMPLETE" in stdout_text:
            # Import reload function and use it
            from vad_settings import reload_vad_settings
            VAD_SETTINGS = reload_vad_settings()
            
            # Print new settings
            print(f"{Fore.GREEN}Calibration successful!{Fore.WHITE}")
            print(f"Updated VAD settings after calibration: {VAD_SETTINGS}")
            
            # IMPORTANT: Update the global VAD settings in your Vosk transcriber code
            if "transcriber" in globals() and transcriber and hasattr(transcriber, "vad_settings"):
                transcriber.vad_settings = VAD_SETTINGS
                print("Updated transcriber VAD settings")
            
            # Restore previous display state
            await display_manager.update_display(previous_state, mood=previous_mood)
            
            # Generate voice confirmation
            await generate_voice("Voice detection calibration completed successfully. Your settings have been updated.")
            await audio_manager.wait_for_audio_completion()
            
            return True
        else:
            print(f"{Fore.RED}Calibration error: {stderr.decode()}{Fore.WHITE}")
            await display_manager.update_display(previous_state, mood=previous_mood)
            await generate_voice("There was an error during voice detection calibration. Please try again later.")
            await audio_manager.wait_for_audio_completion()
            return False
            
    except Exception as e:
        print(f"{Fore.RED}Failed to run calibration: {e}{Fore.WHITE}")
        traceback.print_exc()
        await generate_voice("Voice detection calibration failed to start properly. Please try again later.")
        await audio_manager.wait_for_audio_completion()
        return False

async def heartbeat(remote_transcriber):
    while True:
        try:
            if remote_transcriber.websocket:
                await remote_transcriber.websocket.ping()
            await asyncio.sleep(30)  # Check every 30 seconds
        except:
            remote_transcriber.websocket = None

def get_random_audio(category):
    """Get random audio file from specified category directory"""
    audio_path = Path(SOUND_PATHS[category])
    audio_files = list(audio_path.glob('*.mp3')) + list(audio_path.glob('*.wav'))
    if not audio_files:
        print(f"WARNING: No audio files found in {audio_path}")
        return None
    return str(random.choice(audio_files))

async def main():
    global remote_transcriber, display_manager, transcriber, token_tracker  
    tasks = []  
    
    try:
        # Add verification before token tracker initialization
        verify_token_tracker_setup()

        # Initialize token trackert
        token_tracker = TokenTracker(anthropic_client)

        # Add the debug print RIGHT HERE, after initialization
        print(f"TokenTracker methods available: {dir(token_tracker)}")

        # Verify initialization
        try:
            print(f"\n{Fore.CYAN}=== Token Tracker Initialization ===")
            print(f"Token Tracker Status: {'Initialized' if token_tracker else 'Failed'}")
            print(f"System Prompt Tokens: {token_tracker.session_tokens['input']['system_prompt']}")
            print(f"Initial Setup Complete: {'Yes' if token_tracker.client else 'No'}{Fore.WHITE}\n")
        except Exception as e:
            print(f"{Fore.RED}Token Tracker Verification Failed: {e}{Fore.WHITE}")

        # Initialize transcription based on configuration
        if TRANSCRIPTION_MODE == "remote":
            remote_transcriber = RemoteTranscriber()
            print(f"{Fore.YELLOW}Using remote transcription service{Fore.WHITE}")
        else:
            print(f"{Fore.YELLOW}Using local transcription{Fore.WHITE}")
            if TRANSCRIPTION_MODE == "local" and not transcriber:
                if TRANSCRIPTION_ENGINE == "vosk":
                    transcriber = VoskTranscriber(VOSK_MODEL_PATH)
                else:
                    transcriber = WhisperCppTranscriber(WHISPER_MODEL_PATH, VAD_SETTINGS)
        
        # Create tasks list with token tracking consideration
        tasks = [
            asyncio.create_task(display_manager.rotate_background()),
            asyncio.create_task(run_main_loop()),
            asyncio.create_task(check_upcoming_events())
        ]
        
        # Add heartbeat task only for remote transcription
        if TRANSCRIPTION_MODE == "remote":
            tasks.append(asyncio.create_task(heartbeat(remote_transcriber)))
        
        # Wait for all tasks with exception handling
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"Task error: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
    finally:
        # Cleanup section
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Clean up transcription resources
        if TRANSCRIPTION_MODE == "remote" and remote_transcriber:
            try:
                await remote_transcriber.cleanup()
            except Exception as e:
                print(f"Error cleaning up remote transcriber: {e}")
        elif transcriber:
            try:
                transcriber.cleanup()
            except Exception as e:
                print(f"Error cleaning up transcriber: {e}")
        
        try:
            display_manager.cleanup()
        except Exception as e:
            print(f"Error cleaning up display: {e}")
        
        try:
            await audio_manager.reset_audio_state()
        except Exception as e:
            print(f"Error cleaning up audio: {e}")

async def run_main_loop():
    """Main interaction loop with proper state management"""
    while True:
        try:
            # Handle initial state and wake word
            if display_manager.current_state in ['sleep', 'idle']:
                wake_detected = await wake_word()
                
                if not wake_detected:
                    await asyncio.sleep(0.1)
                    continue
                
                if display_manager.current_state == 'sleep':
                    await display_manager.update_display('wake')
                
                wake_audio = get_random_audio('wake')
                if wake_audio:
                    await audio_manager.play_audio(wake_audio)
                else:
                    # Fallback to ElevenLabs if no audio file found
                    await generate_voice("I'm here to help")
                
                await audio_manager.wait_for_audio_completion()
                await asyncio.sleep(0.3)  # Small buffer after wake word response
                await display_manager.update_display('listening')
            
            # Handle voice input
            transcript = await handle_voice_query()
            
            if transcript:
                print(f"\n{Style.BRIGHT}User:{Style.NORMAL} {transcript}")
                await display_manager.update_display('thinking')
                
                # Generate and process response
                res = await generate_response(transcript)
                print(f"\n{Style.BRIGHT}Laura:{Style.NORMAL}")
                
                # Handle mood-based display updates and print response
                mood_match = re.match(r'\[(.*?)\]', res)
                if mood_match:
                    mood, message = mood_match.groups()
                    await display_manager.update_display('speaking', mood=mood.lower())
                    res = message.strip()
                else:
                    await display_manager.update_display('speaking')
                
                # Print response before generating voice
                print(res)
                await generate_voice(res)
                
                # Handle follow-up conversation if response ends with question
                if isinstance(res, str) and "?" in res:
                    await audio_manager.wait_for_audio_completion()
                    await asyncio.sleep(0.3)  # Small buffer before listening
                    await display_manager.update_display('listening')
                    
                    # Handle follow-up conversation if response ends with question
                    if isinstance(res, str) and "?" in res:
                        await audio_manager.wait_for_audio_completion()
                        await asyncio.sleep(0.3)  # Small buffer before listening
                        await display_manager.update_display('listening')
    
                        while "?" in res:
                            follow_up = await conversation_mode()
                            if follow_up:
                                await display_manager.update_display('thinking')
                                res = await generate_response(follow_up)
                                print(f"\n{Style.BRIGHT}Response:{Style.NORMAL}")
            
                                mood_match = re.match(r'\[(.*?)\]', res)
                                if mood_match:
                                    mood, message = mood_match.groups()
                                    await display_manager.update_display('speaking', mood=mood.lower())
                                    res = message.strip()
                                else:
                                    await display_manager.update_display('speaking')
            
                                print(res)
                                await generate_voice(res)
            
                                if not "?" in res:
                                    await audio_manager.wait_for_audio_completion()
                                    await asyncio.sleep(0.3)  # Small buffer before idle
                                    await display_manager.update_display('idle')
                                    break
            
                                await audio_manager.wait_for_audio_completion()
                                await asyncio.sleep(0.3)  # Small buffer before listening again
                                await display_manager.update_display('listening')
                            else:
                                timeout_audio = get_random_audio("timeout")
                                if timeout_audio:
                                    await audio_manager.play_audio(timeout_audio)
                                else:
                                    # Fallback to ElevenLabs if no audio file found
                                    await generate_voice("No input detected. Feel free to ask for assistance when needed")
                                await audio_manager.wait_for_audio_completion()
                                await asyncio.sleep(0.2)  # Small buffer before idle
                                await display_manager.update_display('idle')
                                break
                else:
                    await audio_manager.wait_for_audio_completion()
                    await asyncio.sleep(0.2)  # Small buffer before idle
                    await display_manager.update_display('idle')
            else:
                timeout_audio = get_random_audio('timeout')
                if timeout_audio:
                    await audio_manager.play_audio(timeout_audio)
                else:
                    print("WARNING: No timeout audio files found")
                await audio_manager.wait_for_audio_completion()
                await asyncio.sleep(0.2)  # Small buffer before idle
                await display_manager.update_display('idle')

        except Exception as e:
            print(f"Error in main loop: {e}")
            traceback.print_exc()  # Added for better error debugging
            await display_manager.update_display('idle')

if __name__ == "__main__":
    try:
        display_manager = DisplayManager()
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting Virtual Assistant")
        if 'remote_transcriber' in globals() and remote_transcriber:
            asyncio.run(remote_transcriber.cleanup())
        if 'transcriber' in globals() and transcriber:  # Changed from whisper_transcriber
            transcriber.cleanup()
        if 'display_manager' in globals() and display_manager:
            display_manager.cleanup()
        if 'audio_manager' in globals() and audio_manager:
            asyncio.run(audio_manager.cleanup())