
#!/usr/bin/env python3

import os
import re
import time
import json
import base64
import random
import struct
import requests
import textwrap
import threading
import traceback
import websockets
import asyncio
import json

from PIL import Image
from mutagen.mp3 import MP3
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from anthropic import Anthropic, APIError, APIConnectionError, BadRequestError, InternalServerError

import pyaudio
import pvporcupine
import whisper
import numpy as np
import wave

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from display_manager import DisplayManager

from datetime import datetime
import requests
from typing import Dict, Tuple
from secret import GOOGLE_MAPS_API_KEY 

from colorama import Fore, Style
from elevenlabs.client import ElevenLabs
from config import TRANSCRIPTION_SERVER

from audio_manager import AudioManager  # Your existing audio manager
from pathlib import Path
from secret import OPENROUTER_API_KEY, PV_ACCESS_KEY, ELEVENLABS_KEY, ANTHROPIC_API_KEY
from config import (VOICE, MOODS, USE_GOOGLE, CONVERSATION_END_SECONDS, ELEVENLABS_MODEL,
                   VOICE_TIMEOUT, VOICE_START_TIMEOUT, SYSTEM_PROMPT, WAKE_WORDS, 
                   WAKE_SENTENCES, TOOL_SENTENCES, TIMEOUT_SENTENCES, CALENDAR_NOTIFICATION_INTERVALS, DEBUG_CALENDAR,
                   TTS_ENGINE, ANTHROPIC_MODEL,CALENDAR_NOTIFICATION_MINUTES, CALENDAR_NOTIFICATION_SENTENCES,)

#BASE_URL = "https://openrouter.ai/api/v1/chat/completions"  # This is correct as per docs
AUDIO_FILE = "speech.mp3"

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

if len(MOODS) > 1:
    SYSTEM_PROMPT += "\nInclude one of these mood identifiers in your response: "
    SYSTEM_PROMPT += " ".join(f"[{m}]" for m in MOODS)

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
                #print("Connection is stale, reconnecting...")
                await self.cleanup()
                await self.connect()
            elif not self.websocket and not await self.connect():
                #print("Failed to connect to transcription server")
                return None

            # Add debug information about audio data size
            #audio_size_mb = (len(audio_data) * audio_data.itemsize) / (1024 * 1024)
            #print(f"Sending audio data: {audio_size_mb:.2f} MB")

            # Send audio data with timeout
            try:
                await asyncio.wait_for(
                    self.websocket.send(json.dumps({
                        "audio": audio_data.tolist()
                    })),
                    timeout=self.send_timeout
                )
                #print("Audio data sent successfully")
            except asyncio.TimeoutError:
                #print(f"Timeout while sending audio data after {self.send_timeout} seconds")
                await self.cleanup()
                return None

            # Get transcription with timeout
            try:
                response = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=self.receive_timeout
                )
                #print("Response received successfully")
            except asyncio.TimeoutError:
                #print(f"Timeout while waiting for transcription after {self.receive_timeout} seconds")
                await self.cleanup()
                return None

            result = json.loads(response)
            if "error" in result:
                print(f"Server error: {result['error']}")
                return None

            transcript = result["transcript"]
    
            # If no transcript, return None
            if not transcript:
                return None
        
            # Normalize the transcript text (strip whitespace, lowercase)
            normalized_text = transcript.lower().strip()
    
            # Base phrases to catch
            base_phrases = [
                "thank you for watching",
                "thanks for watching",
                "thank you watching",
                "thanks watching",
            ]
    
            # Check for any variation of these phrases
            for phrase in base_phrases:
                # Remove all punctuation and check
                if phrase in normalized_text.replace('!', '').replace('.', '').replace(',', ''):
                    print(f"End phrase detected: {transcript}")
                    return None
            
            return transcript

        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed unexpectedly: {e}")
            self.websocket = None
            return await self.transcribe(audio_data)
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
whisper_transcriber = None
audio_manager = AudioManager(PV_ACCESS_KEY)
display_manager = None
chat_log = []
last_interaction = datetime.now()
tts_handler = TTSHandler(config)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
initial_startup = True
last_interaction_check = datetime.now()

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
            
        print("DEBUG: Response received:", response)
        
        if response.stop_reason == "tool_use":
            print("DEBUG: Tool use detected!")
            await display_manager.update_display('tools')
            t1 = threading.Thread(target=generate_voice, args=(random.choice(TOOL_SENTENCES),))
            t1.start()
            
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
    try:
        global initial_startup, last_interaction_check
        porcupine = pvporcupine.create(
            access_key=PV_ACCESS_KEY,
            keyword_paths=WAKE_WORDS.keys(),
            sensitivities=WAKE_WORDS.values()
        )
        
        wake_pa = pyaudio.PyAudio()
        wake_audio_stream = wake_pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length)
        
        print(Fore.YELLOW + "\nListening for wake word" + Fore.WHITE)
        
        while True:
            # Check conversation timeout
            now = datetime.now()
            if (now - last_interaction_check).total_seconds() > CONVERSATION_END_SECONDS:
                last_interaction_check = now
                if display_manager.current_state != 'sleep':
                    await display_manager.update_display('sleep')
            
            pcm = wake_audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            keyword_index = porcupine.process(pcm)

            if keyword_index >= 0:
                print(Fore.GREEN + "Wake word detected" + Fore.WHITE)
                last_interaction_check = datetime.now()
                return True
    finally:
        if 'wake_audio_stream' in locals():
            wake_audio_stream.stop_stream()
            wake_audio_stream.close()
        if 'wake_pa' in locals():
            wake_pa.terminate()
        if 'porcupine' in locals():
            porcupine.delete()

async def handle_voice_query():
    try:
        print(f"{Fore.BLUE}Listening...{Fore.WHITE}")
        audio_stream, cobra = await audio_manager.start_listening()
        
        recording = []
        start_time = time.time()
        voice_detected = False
        
        while (time.time() - start_time) < VOICE_START_TIMEOUT:
            pcm = audio_stream.read(cobra.frame_length)
            pcm = struct.unpack_from("h" * cobra.frame_length, pcm)
            recording.extend(pcm)
            
            if cobra.process(pcm) > 0.3:
                print(f"{Fore.BLUE}Voice detected{Fore.WHITE}")
                voice_detected = True
                break
        
        if not voice_detected:
            print("No voice detected")
            return None

        print(f"{Fore.MAGENTA}Recording...{Fore.WHITE}")
        last_voice_time = time.time()
        
        while True:
            pcm = audio_stream.read(cobra.frame_length)
            pcm = struct.unpack_from("h" * cobra.frame_length, pcm)
            recording.extend(pcm)
            
            if cobra.process(pcm) > 0.1:
                last_voice_time = time.time()
            elif time.time() - last_voice_time > VOICE_TIMEOUT:
                print(f"{Fore.MAGENTA}End of query detected{Fore.WHITE}")
                break

        if recording:
            audio_array = np.array(recording, dtype=np.float32) / 32768.0
            transcript = await remote_transcriber.transcribe(audio_array)
            
            end_phrases = [
                "thank you for watching",
                "thanks for watching",
                "thank you for watching!",
                "thanks for watching!",
                "thanks you for watching",
                "thanks you for watching!"
            ]
            
            if not transcript or transcript.lower().strip() in end_phrases:
                print("Invalid or end phrase detected, ignoring...")
                return None
                
            print(f"Received transcript: {transcript}")
            return transcript

        return None

    finally:
        await audio_manager.stop_listening()

async def conversation_mode():
    try:
        await audio_manager.wait_for_audio_completion()
        # Remove redundant listening update - let run_main_loop handle it
        audio_stream, cobra = await audio_manager.start_listening()

        print(f"\n{Fore.MAGENTA}Recording...{Fore.WHITE}")
        start_time = time.time()
        recording = []
        last_voice_time = time.time()
        voice_detected = False

        initial_timeout = 4.0

        while True:
            if not voice_detected and (time.time() - start_time) > initial_timeout:
                print(f"{Fore.RED}Timeout occurred in conversation mode{Fore.WHITE}")
                return None
                
            pcm = audio_stream.read(cobra.frame_length)
            pcm = struct.unpack_from("h" * cobra.frame_length, pcm)
            recording.extend(pcm)
            
            if cobra.process(pcm) > 0.3:
                voice_detected = True
                last_voice_time = time.time()
            elif voice_detected and (time.time() - last_voice_time > 3.0):
                print(f"{Fore.MAGENTA}End of query detected{Fore.WHITE}")
                break

        if recording and voice_detected:
            audio_array = np.array(recording, dtype=np.float32) / 32768.0
            transcript = await remote_transcriber.transcribe(audio_array)
            
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

def print_response(chat):
    wrapper = textwrap.TextWrapper(width=70)
    paragraphs = chat.split('\n')
    wrapped_chat = "\n".join([wrapper.fill(p) for p in paragraphs if p.strip()])
    for word in wrapped_chat:
        time.sleep(0.06)
        print(word, end="", flush=True)
    print()

async def generate_voice(chat):
    try:
        # Extract mood if present
        mood_pattern = f"\\[({'|'.join(MOODS)})\\]"
        selected_moods = re.findall(mood_pattern, chat, re.RegexFlag.IGNORECASE)
        if selected_moods:
            await display_manager.update_display('speaking')
        chat = re.sub(mood_pattern, "", chat, flags=re.RegexFlag.IGNORECASE).strip()

        audio = tts_handler.generate_audio(chat)

        with open(AUDIO_FILE, "wb") as f:
            f.write(audio)
    
        file = MP3(AUDIO_FILE)
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

def get_next_event() -> str:
    """Get the next upcoming event"""
    if DEBUG_CALENDAR:
        print("DEBUG: Starting get_next_event()")
    try:
        service = get_calendar_service()
        if not service:
            return "Failed to initialize calendar service"
            
        now = datetime.now(timezone.utc)  # Use timezone-aware datetime
        if DEBUG_CALENDAR:
            print(f"DEBUG: Fetching events after {now}")
        
        events_result = service.events().list(
            calendarId="primary",
            timeMin=now.isoformat(),  # Use ISO format of timezone-aware datetime
            maxResults=10,  # Increase this to look at more events
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        
        events = events_result.get("items", [])
        if DEBUG_CALENDAR:
            print(f"DEBUG: Found {len(events)} events")
        
        if not events:
            return "No upcoming events found."
            
        # Filter out past events
        future_events = []
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            event_time = datetime.fromisoformat(start.replace('Z', '+00:00'))
            if event_time > now:
                future_events.append(event)
        
        if not future_events:
            return "No upcoming events found."
            
        event = future_events[0]
        start = event["start"].get("dateTime", event["start"].get("date"))
        start_time = datetime.fromisoformat(start.replace('Z', '+00:00'))
        
        # Format the response based on how far away the event is
        time_diff = start_time - now
        minutes_until = int(time_diff.total_seconds() / 60)
        
        if minutes_until < 0:
            return f"Your current event '{event['summary']}' started {abs(minutes_until)} minutes ago"
        else:
            return f"Your next event is '{event['summary']}' at {start_time.strftime('%I:%M %p')}"
            
    except Exception as e:
        if DEBUG_CALENDAR:
            print(f"DEBUG: Error in get_next_event: {e}")
        return f"Error retrieving next event: {str(e)}"

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

def check_upcoming_events():
    """Background thread to check for upcoming events"""
    notified_events = {}  # Dictionary to track notifications per event and interval
    
    while True:
        try:
            service = get_calendar_service()
            now = datetime.now(timezone.utc)
            
            # Look ahead to the furthest notification interval
            max_minutes = max(CALENDAR_NOTIFICATION_INTERVALS)
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
                start = event['start'].get('dateTime', event['start'].get('date'))
                
                if 'T' in start:
                    start_time = datetime.fromisoformat(start.replace('Z', '+00:00'))
                else:
                    start_time = datetime.strptime(start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                
                minutes_until = int((start_time - now).total_seconds() / 60)
                
                # Check each notification interval
                for interval in CALENDAR_NOTIFICATION_INTERVALS:
                    notification_key = f"{event_id}_{interval}"
                    
                    # If we're within 1 minute of the notification interval and haven't notified yet
                    if (interval-1) <= minutes_until <= interval and notification_key not in notified_events:
                        audio_manager.wait_for_audio_completion()
                        
                        notification_text = random.choice(CALENDAR_NOTIFICATION_SENTENCES).format(
                            minutes=minutes_until,
                            event=event['summary']
                        )
                        
                        generate_voice(notification_text)
                        notified_events[notification_key] = True
                
                # Clean up old notifications
                if minutes_until <= 0:
                    for interval in CALENDAR_NOTIFICATION_INTERVALS:
                        notified_events.pop(f"{event_id}_{interval}", None)
            
        except Exception as e:
            if DEBUG_CALENDAR:
                print(f"Error checking upcoming events: {e}")
                traceback.print_exc()
        
        time.sleep(60)

def start_notification_checker():
    """Start the notification checker in a background thread"""
    try:
        notification_thread = threading.Thread(
            target=check_upcoming_events,
            daemon=True
        )
        notification_thread.start()
        print("Calendar notification checker started successfully")
    except Exception as e:
        print(f"Error starting notification checker: {e}")

async def heartbeat(remote_transcriber):
    while True:
        try:
            if remote_transcriber.websocket:
                await remote_transcriber.websocket.ping()
            await asyncio.sleep(30)  # Check every 30 seconds
        except:
            remote_transcriber.websocket = None

async def main():
    global remote_transcriber, display_manager
    try:
        remote_transcriber = RemoteTranscriber()
        
        # Create background tasks
        rotate_task = asyncio.create_task(display_manager.rotate_background())
        audio_task = asyncio.create_task(audio_manager.monitor_audio_state())
        heartbeat_task = asyncio.create_task(heartbeat(remote_transcriber))
        main_loop_task = asyncio.create_task(run_main_loop())
        
        # Print task info
        print(f"Rotation task created: {rotate_task}")
        
        # Gather tasks with exception handling
        try:
            await asyncio.gather(
                rotate_task,
                audio_task,
                heartbeat_task,
                main_loop_task,
                return_exceptions=True  # This will prevent one task failing from killing others
            )
        except Exception as e:
            print(f"Task error: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
    finally:
        # Ensure tasks are cleaned up
        for task in [rotate_task, audio_task, heartbeat_task, main_loop_task]:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

async def run_main_loop():
    while True:
        try:
            # Handle initial state and wake word
            if display_manager.current_state == 'sleep':
                await wake_word()
                await display_manager.update_display('wake')
                await generate_voice(random.choice(WAKE_SENTENCES))
                await audio_manager.wait_for_audio_completion()
                await display_manager.update_display('listening')
            elif display_manager.current_state == 'idle':
                await wake_word()
                await generate_voice(random.choice(WAKE_SENTENCES))
                await audio_manager.wait_for_audio_completion()
                await display_manager.update_display('listening')
            
            transcript = await handle_voice_query()
            
            if transcript:
                print(f"\n{Style.BRIGHT}User:{Style.NORMAL} {transcript}")
                await display_manager.update_display('thinking')
                
                res = await generate_response(transcript)
                print(f"\n{Style.BRIGHT}Laura:{Style.NORMAL}\n")        
                
                mood_match = re.match(r'\[(.*?)\](.*)', res)
                if mood_match:
                    mood, message = mood_match.groups()
                    await display_manager.update_display('speaking', mood=mood.lower())
                    res = message.strip()
                else:
                    await display_manager.update_display('speaking')
                
                t1 = threading.Thread(target=lambda: asyncio.run(generate_voice(res)))
                t2 = threading.Thread(target=print_response, args=(res,))
                t1.start()
                t2.start()
                t1.join()
                t2.join()
                
                if isinstance(res, str) and "?" in res:
                    await audio_manager.wait_for_audio_completion()
                    await display_manager.update_display('listening')
                    while "?" in res:
                        follow_up = await conversation_mode()
                        if follow_up:
                            await display_manager.update_display('thinking')
                            res = await generate_response(follow_up)
                            print(f"\n{Style.BRIGHT}Response:{Style.NORMAL}\n")        
                            
                            mood_match = re.match(r'\[(.*?)\](.*)', res)
                            if mood_match:
                                mood, message = mood_match.groups()
                                await display_manager.update_display('speaking', mood=mood.lower())
                                res = message.strip()
                            else:
                                await display_manager.update_display('speaking')
                                
                            t1 = threading.Thread(target=lambda: asyncio.run(generate_voice(res)))
                            t2 = threading.Thread(target=print_response, args=(res,))
                            t1.start()
                            t2.start()
                            t1.join()
                            t2.join()
                            
                            if not "?" in res:
                                await display_manager.update_display('idle')
                                break
                            
                            await audio_manager.wait_for_audio_completion()
                            await display_manager.update_display('listening')
                        else:
                            print(f"{Fore.RED}No response detected, playing timeout prompt{Fore.WHITE}")
                            await generate_voice(random.choice(TIMEOUT_SENTENCES))
                            await audio_manager.wait_for_audio_completion()
                            await display_manager.update_display('idle')
                            break
                else:
                    await display_manager.update_display('idle')
            else:
                print(f"{Fore.RED}No response detected, playing timeout prompt{Fore.WHITE}")
                await generate_voice(random.choice(TIMEOUT_SENTENCES))
                await audio_manager.wait_for_audio_completion()
                await display_manager.update_display('idle')

        except Exception as e:
            print(f"Error in main loop: {e}")
            traceback.print_exc()
            await display_manager.update_display('idle')

# REPLACE your existing __main__ check with:
if __name__ == "__main__":
    try:
        display_manager = DisplayManager()  # Single instance creation
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting Virtual Assistant")
        asyncio.run(remote_transcriber.cleanup())
        display_manager.cleanup()
        asyncio.run(audio_manager.cleanup())