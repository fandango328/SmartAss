#!/usr/bin/env python3

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
from typing import Dict, Tuple
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
AUDIO_FILE = "speech.mp3" #get saved to/overwritten by elevenlabs after each completed voice generation and delivery

SCOPES = [  
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.labels",
    "https://www.googleapis.com/auth/gmail.settings.basic",
    "https://www.googleapis.com/auth/gmail.settings.sharing",
    "https://mail.google.com/",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
]

config = {
    "TTS_ENGINE": TTS_ENGINE,
    "ELEVENLABS_KEY": ELEVENLABS_KEY,
    "VOICE": VOICE,
    "ELEVENLABS_MODEL": ELEVENLABS_MODEL, 
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
    }
]
#this is to update the pygame window with the appropriate mood logic based on the user response
#if len(MOODS) > 1:
#    SYSTEM_PROMPT += "\nInclude one of these mood identifiers in your response: "
#    SYSTEM_PROMPT += " ".join(f"[{m}]" for m in MOODS)

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
            audio_length = len(audio_data) / 16000
        
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

# google setup
creds = None
if USE_GOOGLE:
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(open_browser=True)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

async def generate_response(query):
    global chat_log, last_interaction, last_interaction_check
    
    now = datetime.now()    
    last_interaction = now  # Update last interaction time
    last_interaction_check = now  # Update check time for wake word listener
    
    if (now - last_interaction).total_seconds() > CONVERSATION_END_SECONDS:    
        chat_log = []    
        await display_manager.update_display('sleep')
    
    chat_log.append({    
        "role": "user",
        "content": query
    })      
    
    try:
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            system=SYSTEM_PROMPT,
            messages=chat_log,
            max_tokens=1024,
            temperature=0.7,
            tools=TOOLS,
            tool_choice={"type": "auto"}
        )
                
        if hasattr(response, 'error'):    
            error_msg = f"Sorry, there was an error: {response.error}"
            print(f"API Error: {response.error}")
            await display_manager.update_display('speaking', mood='neutral')
            return error_msg
            
        if response.stop_reason == "tool_use":
            print("DEBUG: Tool use detected!")
            await display_manager.update_display('tools')
            tool_audio = get_random_audio('tool')
            if tool_audio:
                await audio_manager.play_audio(tool_audio)
            else:
                print("WARNING: No tool audio files found, skipping audio")
            
            chat_log.append({
                "role": "assistant",
                "content": [block.model_dump() for block in response.content]
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
                        elif tool_call.name == "calendar_query":
                            if tool_args["query_type"] == "next_event":
                                tool_response = get_next_event()
                            elif tool_args["query_type"] == "day_schedule":
                                tool_response = get_day_schedule()
                            else:
                                tool_response = "Unsupported query type"
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
                chat_log.append({
                    "role": "user",
                    "content": tool_results
                })
            
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
                
            if not hasattr(final_response, 'content') or not final_response.content:
                await display_manager.update_display('speaking', mood='neutral')
                return "Sorry, there was an error processing the tool response"
                
            for content_block in final_response.content:
                if content_block.type == "text":
                    text = content_block.text
        
                    # Only use one regex - the strict one that only matches valid moods
                    mood_match = re.match(r'^\[(neutral|happy|confused|disappointed|annoyed|surprised)\](.*)', 
                                         text, re.IGNORECASE)
        
                    if mood_match:
                        # Extract mood and message from valid mood match
                        mood, message = mood_match.groups()
                        mood = mood.lower()
                        message = message.strip()
            
                        print(f"DEBUG: Mood detected: '{mood}'")
                        await display_manager.update_display('speaking', mood=mood)
                        chat_log.append({"role": "assistant", "content": message})
                        return message
                    else:
                        # No valid mood found in brackets at the start
                        print("DEBUG: No valid mood detected, using neutral")
                        await display_manager.update_display('speaking', mood='neutral')
                        chat_log.append({"role": "assistant", "content": text})
                        return text
                
                await display_manager.update_display('speaking', mood='neutral')
                return "No text response found"
            
        else:
            for content_block in response.content:
                if content_block.type == "text":
                    text = content_block.text
                    mood_match = re.match(r'\[(.*?)\](.*)', text)
                    if mood_match:
                        mood, message = mood_match.groups()
                        mood = mood.lower()
                        message = message.strip()
                        
                        if mood not in MOODS:
                            print(f"Warning: Invalid mood '{mood}' received, defaulting to neutral")
                            mood = 'neutral'
                            
                        await display_manager.update_display('speaking', mood=mood)
                        chat_log.append({"role": "assistant", "content": message})
                        return message
                    else:
                        await display_manager.update_display('speaking', mood='neutral')
                        chat_log.append({"role": "assistant", "content": text})
                        return text
            
            await display_manager.update_display('speaking', mood='neutral')
            return "No text response found"

    except (APIError, APIConnectionError, BadRequestError, InternalServerError) as e:
        error_msg = "I apologize, but the service is temporarily overloaded. Please try your request again in a few moments." if "overloaded" in str(e).lower() else f"Sorry, there was a server error: {str(e)}"
        print(f"Anthropic API Error: {e}")
        await display_manager.update_display('speaking', mood='neutral')
        return error_msg
    except Exception as e:
        error_msg = f"Sorry, there was an unexpected error: {str(e)}"
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        await display_manager.update_display('speaking', mood='neutral')
        return error_msg

async def wake_word():
    """Wake word detection with dedicated notification windows"""
    global last_interaction_check, last_interaction
    try:
        snowboy = None
        wake_pa = None
        wake_audio_stream = None
        
        # Track time since last break
        last_break_time = time.time()
        BREAK_INTERVAL = 30  # Take break every 30 seconds
        BREAK_DURATION = 5   # 5 second break for notifications
        
        try:
            print(f"{Fore.YELLOW}Initializing wake word detector...{Fore.WHITE}")
            
            # Debug resource path
            resource_path = Path("resources/common.res")
            model_paths = [Path("GD_Laura.pmdl"), Path("Wake_up_Laura.pmdl"), Path("Laura.pmdl")]
            
            #print(f"Looking for resource file at: {resource_path.absolute()}")
            #print(f"Looking for model files at: {[str(p.absolute()) for p in model_paths]}")
            
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
            
            snowboy.SetSensitivity(b"0.5,0.5,0.5")
            
            wake_pa = pyaudio.PyAudio()
            wake_audio_stream = wake_pa.open(
                rate=16000,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=2048
            )
            
            #print("Snowboy detector created successfully")
            #print(f"{Fore.YELLOW}Listening for wake word{Fore.WHITE}")
            
            while True:
                current_time = time.time()
                
                # Check if it's time for a break
                if current_time - last_break_time >= BREAK_INTERVAL:
                    #print("DEBUG: Taking scheduled break for notifications")
                    last_break_time = current_time
                    
                    # Close audio stream during break
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
                # Get VAD settings from config
                silence_duration = VAD_SETTINGS["silence_duration"]
                energy_threshold = VAD_SETTINGS.get("energy_threshold", 0.060851)  
                continued_threshold = energy_threshold * VAD_SETTINGS.get("continued_threshold_ratio", 0.4)
                very_low_threshold = energy_threshold * 0.2  
                min_speech_duration = VAD_SETTINGS.get("min_speech_duration", 1.5)
                speech_buffer_time = VAD_SETTINGS.get("speech_buffer_time", 1.0)
    
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
                    if partial_text and (current_time - last_partial_time) > 1.5:  # Reduced from 0.5 to 1.0
                        last_partial_time = current_time
                        print(f"Partial: {partial_text}")
        
                    # VAD STATE MACHINE
                    if avg_energy > energy_threshold and not is_speaking:
                        # Speech just started
                        print(f"{Fore.BLUE}Voice detected{Fore.WHITE}")
                        voice_detected = True
                        is_speaking = True
                        speech_start_time = time.time()
                        silence_frames = 0
        
                    elif is_speaking and avg_energy <= energy_threshold:
                        # Potential end of speech - apply nuanced silence tracking
                        if avg_energy < very_low_threshold:
                            silence_frames += 1
                        else:
                            silence_frames = max(0, silence_frames - 2)
            
                        # Check if we've had enough continuous silence
                        speech_duration = time.time() - speech_start_time if speech_start_time else 0
            
                        if (silence_frames >= max_silence_frames and speech_duration > min_speech_duration):
                            print(f"{Fore.MAGENTA}End of speech detected{Fore.WHITE}")
                            await asyncio.sleep(speech_buffer_time)
                            break
        
                    elif is_speaking:
                        # Ongoing speech - reset silence counter
                        silence_frames = 0
        
                    # Reduced CPU usage
                    await asyncio.sleep(0.01)
    
                # Check if we actually detected voice
                if not voice_detected:
                    print("No voice detected")
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
                    
                    if is_speech and not voice_detected:
                        print(f"{Fore.BLUE}Voice detected{Fore.WHITE}")
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
            
            if not transcript:
                return None
                
            print(f"Transcription: {transcript}")
            return transcript
            
        else:  # Remote transcription
            audio_stream, _ = await audio_manager.start_listening()
            
            recording = []
            start_time = time.time()
            voice_detected = False
            speech_energy_threshold = VAD_SETTINGS["energy_threshold"]
            silence_threshold = VAD_SETTINGS["silence_duration"]
            last_voice_time = time.time()
            
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
                if energy > speech_energy_threshold:
                    print(f"{Fore.BLUE}Voice detected (energy: {energy:.4f}){Fore.WHITE}")
                    voice_detected = True
                    last_voice_time = time.time()
                    recording.extend(pcm)
                    break
                
            # If no voice detected in initial phase, return
            if not voice_detected:
                print("No voice detected")
                return None
            
            # Continuous recording phase
            print(f"{Fore.MAGENTA}Recording...{Fore.WHITE}")
            silence_frames = 0
            silence_frame_threshold = int(silence_threshold * audio_manager.sample_rate / audio_manager.frame_length)
            
            while True:
                pcm_bytes = audio_manager.read_audio_frame()
                if not pcm_bytes:
                    await asyncio.sleep(0.01)
                    continue
                
                # Convert bytes to int16 values
                pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
                
                # Add to recording
                recording.extend(pcm)
                
                # Calculate energy
                float_data = pcm.astype(np.float32) / 32768.0
                energy = np.sqrt(np.mean(float_data**2)) if len(float_data) > 0 else 0
                
                # Check if this frame has voice
                if energy > (speech_energy_threshold * 0.4):  # Definite speech
                    silence_frames = 0
                else:
                    # Only increment silence for VERY low energy
                    if energy < speech_energy_threshold * 0.2:  # Very low energy
                        silence_frames += 1
                    else:
                        # Actively decrease the counter for moderate energy
                        silence_frames = max(0, silence_frames - 2)  # Decrease twice as fast
                
                # End recording conditions
                max_recording_time = 60  # Maximum recording time in seconds
                current_length = len(recording) / audio_manager.sample_rate
                
                if silence_frames >= silence_frame_threshold:
                    print(f"{Fore.MAGENTA}Silence detected, ending recording{Fore.WHITE}")
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
    
        if not transcript or not isinstance(transcript, str) or transcript.lower().strip() in end_phrases:
            print("Invalid or end phrase detected, ignoring...")
            return None
            
        print(f"Received transcript: {transcript}")
        return transcript

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
            initial_timeout = 6.0  # Define timeout here at the beginning

            # VOSK-specific conversation handling
            if TRANSCRIPTION_ENGINE == "vosk":
                # Get VAD settings from config
                silence_duration = VAD_SETTINGS["silence_duration"]
                energy_threshold = VAD_SETTINGS.get("energy_threshold", 0.060851)  
                continued_threshold = energy_threshold * VAD_SETTINGS.get("continued_threshold_ratio", 0.4)
                very_low_threshold = energy_threshold * 0.2  
                min_speech_duration = VAD_SETTINGS.get("min_speech_duration", 1.5)
                speech_buffer_time = VAD_SETTINGS.get("speech_buffer_time", 1.0)
    
                # Calculate frames needed for silence duration
                max_silence_frames = int(silence_duration * 16000 / audio_manager.frame_length)
    
                # State tracking
                initial_timeout = 6.0  # For conversation mode initial timeout
                silence_frames = 0
                last_partial_time = time.time()
                frame_history = []
                is_speaking = False
                voice_detected = False
                speech_start_time = None
                start_time = time.time()
    
                while True:
                    # Check for initial timeout (conversation-specific timeout)
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
                    if partial_text and (current_time - last_partial_time) > 1.5:  # Reduced from 0.5 to 1.0
                        last_partial_time = current_time
                        print(f"Partial: {partial_text}")
        
                    # VAD STATE MACHINE
                    if avg_energy > energy_threshold and not is_speaking:
                        # Speech just started
                        print(f"{Fore.GREEN}Voice detected{Fore.WHITE}")
                        voice_detected = True
                        is_speaking = True
                        speech_start_time = time.time()
                        silence_frames = 0
        
                    elif is_speaking and avg_energy <= energy_threshold:
                        # Potential end of speech with nuanced silence tracking
                        if avg_energy < very_low_threshold:
                            silence_frames += 1
                        else:
                            silence_frames = max(0, silence_frames - 2)
            
                        # Check if we've had enough continuous silence
                        speech_duration = time.time() - speech_start_time if speech_start_time else 0
            
                        if (silence_frames >= max_silence_frames and speech_duration > min_speech_duration):
                            print(f"{Fore.MAGENTA}End of speech detected{Fore.WHITE}")
                            await asyncio.sleep(speech_buffer_time)
                            break
        
                    elif is_speaking:
                        # Ongoing speech - reset silence counter
                        silence_frames = 0
        
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
                initial_timeout = 6.0
                
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
            # Now start listening
            audio_stream, _ = await audio_manager.start_listening()
    
            print(f"\n{Fore.MAGENTA}Waiting for response...{Fore.WHITE}")
            start_time = time.time()
            recording = []
            voice_detected = False
            
            # Increased initial timeout for user to start speaking
            initial_timeout = 6.0  # More time to formulate a response
            speech_energy_threshold = VAD_SETTINGS["energy_threshold"]
            silence_threshold = VAD_SETTINGS["silence_duration"]
            silence_frames = 0
            silence_frame_threshold = int(silence_threshold * audio_manager.sample_rate / audio_manager.frame_length)
    
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
                    if energy > speech_energy_threshold:
                        print(f"{Fore.GREEN}Voice detected{Fore.WHITE}")
                        voice_detected = True
                        recording.extend(pcm)
                    continue
                
                # We're already recording, so add this frame
                recording.extend(pcm)
                
                # Check if this frame has speech
                if energy > (speech_energy_threshold * 0.4):  # Lower threshold for continued speech
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
        # More precise mood pattern matching
        mood_pattern = r'^\[(.*?)\](.*)'  # Only match at start of string
        mood_match = re.match(mood_pattern, chat, re.IGNORECASE)
        
        if mood_match:
            mood, message = mood_match.groups()
            mood = mood.lower().strip()
            if mood in MOODS:
                await display_manager.update_display('speaking', mood=mood)
            else:
                print(f"Warning: Invalid mood '{mood}' received, defaulting to neutral")
                await display_manager.update_display('speaking', mood='neutral')
            chat = message.strip()
        else:
            await display_manager.update_display('speaking', mood='neutral')

        # Debug print to verify full message
        print(f"Generating voice for text: {chat}")

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
    global remote_transcriber, display_manager, transcriber
    tasks = []  # Initialize tasks list outside the try block
    
    try:
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
        
        # Create tasks only for components that actually have these methods
        tasks = [
            asyncio.create_task(display_manager.rotate_background()),
            # Removed audio_manager.monitor_audio_state()
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
            # Ensure proper cleanup of all tasks
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
            
            # Use available methods from your existing AudioManager
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
                mood_match = re.match(r'\[(.*?)\](.*)', res)
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
            
                                mood_match = re.match(r'\[(.*?)\](.*)', res)
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