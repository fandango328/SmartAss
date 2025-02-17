
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
import logging
import sys 

from PIL import Image
from mutagen.mp3 import MP3
from datetime import datetime
from email.message import EmailMessage

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
import pygame

from datetime import datetime
import requests
from typing import Dict, Tuple
from secret import GOOGLE_MAPS_API_KEY 

from colorama import Fore, Style
from elevenlabs.client import ElevenLabs
from config import TRANSCRIPTION_SERVER

from audiomanager import AudioManager
from secret import OPENROUTER_API_KEY, PV_ACCESS_KEY, ELEVENLABS_KEY
from config import (MODEL, VOICE, MOODS, USE_GOOGLE, CONVERSATION_END_SECONDS, 
                   VOICE_TIMEOUT, VOICE_START_TIMEOUT, SYSTEM_PROMPT, WAKE_WORDS, 
                   WAKE_SENTENCES, TOOL_SENTENCES, TIMEOUT_SENTENCES,
                   TTS_ENGINE, ALLTALK_HOST, ALLTALK_VOICE, ALLTALK_MODEL)

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
AUDIO_FILE = "speech.mp3"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.labels",
    "https://www.googleapis.com/auth/gmail.settings.basic",
    "https://www.googleapis.com/auth/gmail.settings.sharing",
    "https://mail.google.com/",
]

config = {
    "TTS_ENGINE": "alltalk",  # Changed from "alltalk" to "elevenlabs"
   # "ELEVENLABS_KEY": ELEVENLABS_KEY,  # Added this
    #"VOICE": VOICE,  # Added this
    # Remove AllTalk specific entries since we're not using them
    "ALLTALK_HOST": ALLTALK_HOST,
    "ALLTALK_VOICE": ALLTALK_VOICE,
    "ALLTALK_MODEL": ALLTALK_MODEL
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "draft_email",
            "description": "Draft a new email with a recipient and content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {
                        "type": "string",
                        "description": "Whom will receive the email."
                    },
                    "subject": {
                        "type": "string",
                        "description": "The topic of the email."
                    },
                    "content": {
                        "type": "string",
                        "description": "The content of the email.",
                    }
                },
                "required": ["subject", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_location",
            "description": "Get current location based on WiFi networks",
            "parameters": {
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
    },


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
                print("Connection is stale, reconnecting...")
                await self.cleanup()
                await self.connect()
            elif not self.websocket and not await self.connect():
                print("Failed to connect to transcription server")
                return None

            # Add debug information about audio data size
            audio_size_mb = (len(audio_data) * audio_data.itemsize) / (1024 * 1024)
            print(f"Sending audio data: {audio_size_mb:.2f} MB")

            # Send audio data with timeout
            try:
                await asyncio.wait_for(
                    self.websocket.send(json.dumps({
                        "audio": audio_data.tolist()
                    })),
                    timeout=self.send_timeout
                )
                print("Audio data sent successfully")
            except asyncio.TimeoutError:
                print(f"Timeout while sending audio data after {self.send_timeout} seconds")
                await self.cleanup()
                return None

            # Get transcription with timeout
            try:
                response = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=self.receive_timeout
                )
                print("Response received successfully")
            except asyncio.TimeoutError:
                print(f"Timeout while waiting for transcription after {self.receive_timeout} seconds")
                await self.cleanup()
                return None

            result = json.loads(response)

            if "error" in result:
                print(f"Server error: {result['error']}")
                return None

            return result["transcript"]

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
        print("Initializing TTSHandler...")  # Debug print
        self.config = config
        self.tts_engine = config["TTS_ENGINE"]
        self.eleven = None
        self.current_settings = None
        self.available_voices = None
        self.available_rvc_voices = None
        
        print(f"TTS Engine: {self.tts_engine}")  # Debug print
        print(f"Config: {self.config}")  # Debug print
        
        if "ELEVENLABS_KEY" in config:
            self.eleven = ElevenLabs(api_key=config["ELEVENLABS_KEY"])
        
        if self.tts_engine == "alltalk":
            required_keys = ["ALLTALK_HOST", "ALLTALK_VOICE", "ALLTALK_MODEL"]
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                print(f"Missing required keys: {missing_keys}")  # Debug print
                raise ValueError(f"Missing required AllTalk configuration keys: {missing_keys}")

    def initialize(self):
        """Initialize connection and fetch server capabilities"""
        print("Attempting to initialize TTS...")  # Debug print
        try:
            if self.check_server_ready():
                print("Server is ready")  # Debug print
                self.current_settings = self.get_current_settings()
                self.available_voices = self.get_available_voices()
                self.available_rvc_voices = self.get_available_rvc_voices()
                return True
            else:
                print("Server not ready")  # Debug print
                return False
        except Exception as e:
            print(f"Initialization error: {str(e)}")  # Debug print
            return False

    def check_server_ready(self):
        """Check if AllTalk server is ready to accept requests"""
        try:
            print(f"Checking server at: {self.config['ALLTALK_HOST']}/api/health")  # Debug print
            response = requests.get(f"{self.config['ALLTALK_HOST']}/api/health", timeout=5)
            print(f"Server response: {response.status_code}")  # Debug print
            return response.status_code == 200
        except Exception as e:
            print(f"Server check error: {str(e)}")  # Debug print
            return False

    def check_server_ready(self):
        try:
            response = requests.get(f"{self.config['ALLTALK_HOST']}/api/ready", timeout=5)
            return response.text == "Ready"
        except:
            return False

    def get_current_settings(self):
        try:
            response = requests.get(f"{self.config['ALLTALK_HOST']}/api/currentsettings")
            return response.json() if response.status_code == 200 else None
        except:
            return None

    def get_available_voices(self):
        try:
            response = requests.get(f"{self.config['ALLTALK_HOST']}/api/voices")
            data = response.json()
            return data.get('voices', [])
        except:
            return None

    def get_available_rvc_voices(self):
        try:
            response = requests.get(f"{self.config['ALLTALK_HOST']}/api/rvcvoices")
            data = response.json()
            return data.get('rvcvoices', [])
        except:
            return None

    def generate_audio(self, text):
        try:
            if self.tts_engine == "elevenlabs":
                return self._generate_elevenlabs(text)
            else:
                return self._generate_alltalk(text)
        except Exception as e:
            logging.error(f"TTS generation failed: {str(e)}")
            raise

    def _generate_elevenlabs(self, text):
        if not self.eleven:
            raise ValueError("ElevenLabs not configured")
        try:
            audio = b"".join(self.eleven.generate(
                text=text,
                voice=self.config["VOICE"],
                model="eleven_flash_v2_5"
            ))
            return audio
        except Exception as e:
            logging.error(f"ElevenLabs generation failed: {str(e)}")
            raise

    def _generate_alltalk(self, text):
        try:
            health_check = requests.get(
                f"{self.config['ALLTALK_HOST']}/api/health",
                timeout=5
            )
        
            if health_check.status_code != 200:
                raise Exception("AllTalk server is not healthy")

            # Make TTS request
            response = requests.post(
                f"{self.config['ALLTALK_HOST']}/api/tts",  # Changed from /api/tts-generate
                json={  # Changed from data to json
                    "text": text,
                    "voice": self.config["ALLTALK_VOICE"],
                    "model": self.config["ALLTALK_MODEL"]
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.content
            else:
                raise Exception(f"AllTalk API error: {response.status_code} - {response.text}")
            
        except Exception as e:
            logging.error(f"AllTalk generation failed: {str(e)}")
            if self.eleven and "VOICE" in self.config:
                logging.warning("Falling back to ElevenLabs...")
                return self._generate_elevenlabs(text)
            raise

        def reload_config(self):
            try:
                response = requests.get(f"{self.config['ALLTALK_HOST']}/api/reload_config")
                if response.status_code == 200:
                    return self.initialize()
                return False
            except:
                return False

    def switch_model(self, model_name):
        try:
            response = requests.post(
                f"{self.config['ALLTALK_HOST']}/api/reload",
                params={"tts_method": model_name}
            )
            return response.json() if response.status_code == 200 else None
        except:
            return None

    def set_deepspeed(self, enabled):
        try:
            response = requests.post(
                f"{self.config['ALLTALK_HOST']}/api/deepspeed",
                params={"new_deepspeed_value": str(enabled).lower()}
            )
            return response.json() if response.status_code == 200 else None
        except:
            return None

    def set_low_vram(self, enabled):
        try:
            response = requests.post(
                f"{self.config['ALLTALK_HOST']}/api/lowvramsetting",
                params={"new_low_vram_value": str(enabled).lower()}
            )
            return response.json() if response.status_code == 200 else None
        except:
            return None

    def stop_generation(self):
        try:
            response = requests.put(f"{self.config['ALLTALK_HOST']}/api/stop-generation")
            return response.json() if response.status_code == 200 else None
        except:
            return None

    def display_server_info(self):
        print("\n=== AllTalk Server Information ===")
        print(f"\nServer URL: {self.config['ALLTALK_HOST']}")
        
        print("\n--- Current Settings ---")
        print(self.current_settings)
        
        print("\n--- Available Voices ---")
        print(self.available_voices)
        
        print("\n--- Available RVC Voices ---")
        print(self.available_rvc_voices)
        
        if self.current_settings:
            print("\n--- Server Capabilities ---")
            capabilities = {
                "DeepSpeed Capable": self.current_settings.get('deepspeed_capable', False),
                "DeepSpeed Enabled": self.current_settings.get('deepspeed_enabled', False),
                "Low VRAM Capable": self.current_settings.get('lowvram_capable', False),
                "Low VRAM Enabled": self.current_settings.get('lowvram_enabled', False),
                "Generation Speed Capable": self.current_settings.get('generationspeed_capable', False),
                "Current Generation Speed": self.current_settings.get('generationspeed_set', 'N/A'),
                "Pitch Capable": self.current_settings.get('pitch_capable', False),
                "Current Pitch": self.current_settings.get('pitch_set', 'N/A'),
                "Temperature Capable": self.current_settings.get('temperature_capable', False),
                "Current Temperature": self.current_settings.get('temperature_set', 'N/A'),
                "Streaming Capable": self.current_settings.get('streaming_capable', False),
                "Multi-voice Capable": self.current_settings.get('multivoice_capable', False),
                "Multi-model Capable": self.current_settings.get('multimodel_capable', False),
                "Languages Capable": self.current_settings.get('languages_capable', False)
            }
            print(capabilities)

# Global variables
whisper_transcriber = None
audio_manager = AudioManager(PV_ACCESS_KEY)
chat_log = [SYSTEM_PROMPT]
last_interaction = datetime.now()
tts_handler = TTSHandler(config)

# pygame setup
try:
    img = Image.open(MOODS[0]+".png")
    resolution = (img.width, img.height)
except pygame.error as message:
    print("Cannot load image")
    raise SystemExit(message) 
screen = pygame.display.set_mode(resolution)
try:
    moods_img = {mood: pygame.image.load(mood+".png") for mood in MOODS}
except pygame.error as message:
    print("Cannot load image")
    raise SystemExit(message)
screen.blit(moods_img[MOODS[0]],(0,0))
pygame.display.flip()

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

def generate_response(query):
    global chat_log, last_interaction

    now = datetime.now()
    if (now - last_interaction).total_seconds() > CONVERSATION_END_SECONDS:
        chat_log = [SYSTEM_PROMPT]
    last_interaction = now
    
    chat_log.append({
        "role": "user",
        "content": query
    })      
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Aerith Assistant",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL,
        "messages": chat_log,
        "tools": TOOLS,
    }
    
    try:
        response = requests.post(BASE_URL, headers=headers, json=data)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        response_json = response.json()
        if 'error' in response_json:
            print(f"API Error: {response_json['error']}")
            return f"Sorry, there was an error: {response_json['error']}"
            
        if 'choices' not in response_json:
            print(f"Unexpected API response: {response_json}")
            return "Sorry, there was an unexpected error in the response"
            
        answer = response_json["choices"][0]["message"]
        chat_log.append(answer)

        if answer.get("tool_calls"):
            t1 = threading.Thread(target=generate_voice, args=(random.choice(TOOL_SENTENCES),))
            t1.start()
            for tool_call in answer["tool_calls"]:
                args = json.loads(tool_call["function"]["arguments"])
                match tool_call["function"]["name"]:
                    case "draft_email":
                        tool_response = draft_email(**args)
                    case "get_location":
                        tool_response = get_location(**args)
                    case _:
                        tool_response = "Unsupported tool called"
                
                chat_log.append({
                    "role": "tool",
                    "name": tool_call["function"]["name"],
                    "tool_call_id": tool_call["id"],
                    "content": tool_response,
                })
                print(f"\n{Fore.GREEN}Finished tool call: {tool_call['function']['name']}{Fore.WHITE}")
            response = requests.post(BASE_URL, headers=headers, json=data)
            response_json = response.json()
            if 'choices' in response_json:
                answer = response_json["choices"][0]["message"]
                chat_log.append(answer)
            else:
                return "Sorry, there was an error processing the tool response"
        
        return answer["content"]
        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return f"Sorry, there was an error communicating with the API: {str(e)}"
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return "Sorry, there was an error processing the response"
    except Exception as e:
        print(f"Unexpected error: {e}")
        return f"Sorry, an unexpected error occurred: {str(e)}"

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

def generate_voice(chat):
    global tts_handler
    try:
        mood_pattern = f"\\[({'|'.join(MOODS)})\\]"
        selected_moods = re.findall(mood_pattern, chat, re.RegexFlag.IGNORECASE)
        if selected_moods:
            screen.blit(moods_img[selected_moods[0]], (0,0))
            pygame.display.flip()
        chat = re.sub(mood_pattern, "", chat, flags=re.RegexFlag.IGNORECASE).strip()

        audio = tts_handler.generate_audio(chat)

        with open(AUDIO_FILE, "wb") as f:
            f.write(audio)
    
        file = MP3(AUDIO_FILE)
        t1 = threading.Thread(target=audio_manager.play_audio, args=(AUDIO_FILE,))
        t1.start()

    except Exception:
        print(traceback.print_exc())

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

def wake_word():
    try:
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
            pcm = wake_audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            keyword_index = porcupine.process(pcm)

            if keyword_index >= 0:
                print(Fore.GREEN + "Wake word detected" + Fore.WHITE)
                break
    finally:
        if 'wake_audio_stream' in locals():
            wake_audio_stream.stop_stream()
            wake_audio_stream.close()
        if 'wake_pa' in locals():
            wake_pa.terminate()
        if 'porcupine' in locals():
            porcupine.delete()

async def handle_voice_query():
    global remote_transcriber
    try:
        print(f"{Fore.BLUE}Listening...{Fore.WHITE}")
        audio_stream, cobra = audio_manager.start_listening()
        
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
            print("Timeout occurred")
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
            audio_array = np.array(recording, dtype=np.float32) / 32768.0  # Normalize int16 to float32
            print(f"Sending audio data: shape={audio_array.shape}, dtype={audio_array.dtype}")
            transcript = await remote_transcriber.transcribe(audio_array)
            print(f"Received transcript: {transcript}")
            if transcript:
                return transcript
        return None

    finally:
        audio_manager.stop_listening()

async def conversation_mode():
    global remote_transcriber
    try:
        audio_manager.wait_for_audio_completion()
        audio_stream, cobra = audio_manager.start_listening()

        start_time = time.time()
        recording = []
        last_voice_time = time.time()
        voice_detected = False

        print(f"\n{Fore.MAGENTA}Recording...{Fore.WHITE}")
        while True:
            if not voice_detected and (time.time() - start_time) > 5:
                print(f"{Fore.RED}Timeout occurred in conversation mode{Fore.WHITE}")
                generate_voice(random.choice(TIMEOUT_SENTENCES))
                audio_manager.wait_for_audio_completion()
                return None
                
            pcm = audio_stream.read(cobra.frame_length)
            pcm = struct.unpack_from("h" * cobra.frame_length, pcm)
            recording.extend(pcm)
            
            if cobra.process(pcm) > 0.1:
                voice_detected = True
                last_voice_time = time.time()
            elif voice_detected and (time.time() - last_voice_time > 4.0):
                print(f"{Fore.MAGENTA}End of query detected{Fore.WHITE}")
                break

        if recording and voice_detected:
            # Convert to float32 and normalize
            audio_array = np.array(recording, dtype=np.float32) / 32768.0
            transcript = await remote_transcriber.transcribe(audio_array)
            if transcript:
                print(f"\n{Style.BRIGHT}You said:{Style.NORMAL} {transcript}\n")
                return transcript
        return None

    finally:
        audio_manager.stop_listening()

async def heartbeat(remote_transcriber):
    while True:
        try:
            if remote_transcriber.websocket:
                await remote_transcriber.websocket.ping()
            await asyncio.sleep(30)  # Check every 30 seconds
        except:
            print("Lost connection to transcription server")
            remote_transcriber.websocket = None

async def main():
    global remote_transcriber
    try:
        remote_transcriber = RemoteTranscriber()
        
        print("Creating TTS handler...")  # Debug print
        try:
            if not tts_handler.initialize():
                print("Failed to initialize TTS system - server not ready")
                sys.exit(1)
        except Exception as e:
            print(f"Error during TTS initialization: {str(e)}")
            sys.exit(1)
            
        # Display server information
        tts_handler.display_server_info()
        
        # Configure optimal settings if available
        if tts_handler.current_settings:
            if tts_handler.current_settings.get('deepspeed_capable'):
                tts_handler.set_deepspeed(True)
            if tts_handler.current_settings.get('lowvram_capable'):
                tts_handler.set_low_vram(True)
        
        # Start heartbeat in the background
        heartbeat_task = asyncio.create_task(heartbeat(remote_transcriber))
        
        while True:
            try:
                wake_word()
                generate_voice(random.choice(WAKE_SENTENCES))
                transcript = await handle_voice_query()
                
                if transcript:
                    print(f"\n{Style.BRIGHT}You said:{Style.NORMAL} {transcript}")
                    res = generate_response(transcript)
                    print(f"\n{Style.BRIGHT}Response:{Style.NORMAL}\n")        
                    
                    t1 = threading.Thread(target=generate_voice, args=(res,))
                    t2 = threading.Thread(target=print_response, args=(res,))
                    t1.start()
                    t2.start()

                    while "?" in res:
                        t1.join()
                        t2.join()
    
                        follow_up = await conversation_mode()
                        if follow_up:
                            res = generate_response(follow_up)
                            print(f"\n{Style.BRIGHT}Response:{Style.NORMAL}\n")        
                            t1 = threading.Thread(target=generate_voice, args=(res,))
                            t2 = threading.Thread(target=print_response, args=(res,))
                            t1.start()
                            t2.start()
                        else:
                            break
                    
                    t1.join()
                    t2.join()
                else:
                    print(f"{Fore.RED}No response detected, playing timeout prompt{Fore.WHITE}")
                    generate_voice(random.choice(TIMEOUT_SENTENCES))
                    audio_manager.wait_for_audio_completion()

            except Exception:
                print(traceback.format_exc())

    finally:
        await remote_transcriber.cleanup()
        if 'heartbeat_task' in locals():
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting Virtual Assistant")
        asyncio.run(remote_transcriber.cleanup())
        audio_manager.cleanup()
