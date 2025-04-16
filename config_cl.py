from pathlib import Path
from vad_settings import load_vad_settings
import json

# Load personalities configuration
PERSONALITIES_FILE = "personalities.json"
try:
    with open(PERSONALITIES_FILE, 'r') as f:
        PERSONALITIES_DATA = json.load(f)
    ACTIVE_PERSONA = PERSONALITIES_DATA.get("active_persona", "laura")
    ACTIVE_PERSONA_DATA = PERSONALITIES_DATA.get("personas", {}).get(ACTIVE_PERSONA, {})
except Exception as e:
    print(f"Error loading personalities configuration: {e}")
    PERSONALITIES_DATA = {"personas": {"laura": {"name": "Laura", "voice": "L.A.U.R.A.", 
                         "system_prompt": "You are Laura, an AI assistant."}}, 
                         "active_persona": "laura"}
    ACTIVE_PERSONA = "laura"
    ACTIVE_PERSONA_DATA = PERSONALITIES_DATA["personas"]["laura"]

ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219" # Or whichever Claude model you want to use          

CHAT_LOG_MAX_TOKENS = 80000  # Maximum tokens to keep in memory
CHAT_LOG_RECOVERY_TOKENS = 1000 # Tokens to recover on startup
CHAT_LOG_DIR = "chat_logs"  # Directory to store chat logs

# Let the system use whatever profile is set as active in the JSON file
VAD_SETTINGS = load_vad_settings()  # No hardcoded profile - uses active_profile from JSON

# Transcription Configuration
TRANSCRIPTION_MODE = "local"  # Options: "local", "remote"
TRANSCRIPTION_ENGINE = "vosk"  # Options: "vosk", "whisper"
WHISPER_MODEL_SIZE = "tiny"  # Options: "tiny" "small" "small-q8_0"
WHISPER_MODEL_PATH = f"models/ggml-{WHISPER_MODEL_SIZE}.bin"  # Path to whisper model

# Vosk Configuration
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"  # For 8GB Pi4
#VOSK_MODEL_PATH = "models/vosk-model-en-us-0.22"      # For 8GB Pi 500

# Transcription server configuration
TRANSCRIPTION_SERVER = {
    "host": "xxx.xxx.xx.xxx",  
    "port": 8765
}

# Set up paths for the active persona
SOUND_BASE_PATH = Path(f'/home/user/LAURA/sounds/{ACTIVE_PERSONA}')

# Audio file directories
SOUND_PATHS = {
    'wake': str(SOUND_BASE_PATH / 'wake_sentences'),  
    'tool': {
        'status': {
            'enabled': str(SOUND_BASE_PATH / 'tool_sentences' / 'status' / 'enabled'),
            'disabled': str(SOUND_BASE_PATH / 'tool_sentences' / 'status' / 'disabled'),
        },
        'use': str(SOUND_BASE_PATH / 'tool_sentences' / 'use')
    },
    'file': {
        'loaded': str(SOUND_BASE_PATH / 'file_sentences' / 'loaded'),
        'offloaded': str(SOUND_BASE_PATH / 'file_sentences' / 'offloaded'),
    },
    'timeout': str(SOUND_BASE_PATH / 'timeout_sentences'),
    'calibration': str(SOUND_BASE_PATH / 'calibration'),
    'filler': str(SOUND_BASE_PATH / 'filler'),
    'system': {
        'error': str(SOUND_BASE_PATH / 'system' / 'error')
    }
}

# Use the voice from the active persona
VOICE = ACTIVE_PERSONA_DATA.get("voice", "L.A.U.R.A.")  # voice - elevenlabs specific
USE_GOOGLE = True  # Flag to indicate if Google services are used
CONVERSATION_END_SECONDS = 1200  # Time in seconds before a conversation is considered ended
VOICE_TIMEOUT = 3  # Timeout in seconds for voice detection
VOICE_START_TIMEOUT = 6  # Timeout in seconds for starting voice detection

WAKE_WORDS = {
    "GD_Laura.pmdl": 0.5,
    "Wake_up_Laura.pmdl": 0.5,
    "Laura.pmdl": 0.45
}

# Updated MOODS list
MOODS = [
    "amused", "annoyed", "caring", "casual", "cheerful", "concerned", 
    "confused", "curious", "disappointed", "embarrassed", "excited",
    "frustrated", "interested", "sassy", "scared", "surprised",
    "suspicious", "thoughtful"
]

MOOD_MAPPINGS = {
    # Base moods mapping to themselves
    "amused": "amused",
    "annoyed": "annoyed",
    "caring": "caring",
    "casual": "casual",
    "cheerful": "cheerful",
    "concerned": "concerned",
    "confused": "confused",
    "curious": "curious",
    "disappointed": "disappointed",
    "embarrassed": "embarrassed",
    "excited": "excited",
    "frustrated": "frustrated",
    "interested": "interested",
    "sassy": "sassy",
    "scared": "scared",
    "surprised": "surprised",
    "suspicious": "suspicious",
    "thoughtful": "thoughtful",
    
    # Variant mappings
    "understanding": "caring",
    "helpful": "caring",
    "warm": "caring",
    "empathetic": "caring",
    "sympathetic": "caring",
    "compassionate": "caring",
    "deeply empathetic": "caring",
    "friendly": "casual",
    "comfortable": "casual",
    "practical": "casual",
    "pleased": "cheerful",
    "approving": "cheerful",
    "appreciative": "cheerful",
    "intrigued": "curious",
    "engaged": "curious",
    "attentive": "curious",
    "apologetic": "disappointed",
    "sheepish": "embarrassed",
    "playful": "sassy",
    "laughing": "sassy",
    "impressed": "surprised",
    "anticipatory": "surprised",
    "direct": "suspicious",
    "reflective": "thoughtful",
    "focused": "thoughtful",
    "pensive": "thoughtful",
    "deeply reflective": "thoughtful",
    "informative": "thoughtful"
}

# Load the persona-specific system prompt
PERSONA_SYSTEM_PROMPT = ACTIVE_PERSONA_DATA.get("system_prompt", "You are an AI assistant.")

# The universal parts of the system prompt
UNIVERSAL_SYSTEM_PROMPT = """
Response Guidelines:
- Short, clear responses optimized for voice interaction
- Natural flow without mechanical questioning
- Focus on actionable solutions
- Show backbone when challenged, but remain professional
- Balance efficiency with personality

Voice Interaction:
- Optimize responses for 1-2 sentences
- Clear, concise communication
- Natural conversational flow
- Avoid overly formal or mechanical responses

Development Status:
- Running on a Raspberry Pi 500
- Voice-activated interaction model
- Continuous learning about user preferences and habits

MOOD INSTRUCTIONS:
When indicating mood, use ONLY these exact words in brackets at the very beginning of your response:
[cheerful], [confused], [disappointed], [annoyed], [surprised], [caring], [casual], [cheerful], [concerned], [curious], [embarrassed], [sassy], [suspicious], [thoughtful]

EXAMPLE CORRECT FORMAT:
[mood] Your response text...

INCORRECT FORMATS:
[happy:] Text... (don't include colons)
[happy/excited] Text... (don't include slashes)
[Here's my answer:] Text... (this isn't a valid mood)
"""

# Combine persona-specific and universal parts
SYSTEM_PROMPT = f"{PERSONA_SYSTEM_PROMPT}\n\n{UNIVERSAL_SYSTEM_PROMPT}"


# System State Commands
SYSTEM_STATE_COMMANDS = {
    "tool": {
        "enable": [
            "tools activate", "launch toolkit", "begin assistance", "enable tool use",
            "start tools", "enable assistant", "tools online", "enable tools",
            "assistant power up", "toolkit online", "helper mode active",
            "utilities on", "activate functions", "tools ready",
            "tools on", "toolkit on", "functions on",
            "wake up tools", "prepare toolkit", "bring tools online"
        ],
        "disable": [
            "tools offline", "end toolkit", "close assistant",
            "stop tools", "disable assistant", "conversation only",
            "assistant power down", "toolkit offline", "helper mode inactive",
            "utilities off", "deactivate functions", "tools away",
            "tools off", "toolkit off", "functions off", "disable tools", "disable tool use",
            "sleep tools", "dismiss toolkit", "take tools offline"
        ]
    },
    "document": {
        "load": [
            "load file", "load files", "load all files", 
            "load my file", "load my files"
        ],
        "offload": [
            "offload file", "offload my file", "offload files", 
            "offload my files", "offload all files", "remove file", 
            "remove files", "remove all files", "remove my files", 
            "clear file", "clear files", "clear all files", 
            "clear my files"
        ]
    }
}

# TTS Settings
TTS_ENGINE = "elevenlabs"  # Options: "elevenlabs" or "local"

# ElevenLabs Settings
ELEVENLABS_MODEL = "eleven_flash_v2_5"  # Model used for ElevenLabs TTS

# Local TTS Settings
LOCAL_TTS_HOST = "http://xxx.xxx.xx.xxx"  # Host for the local TTS engine - for remote 192.168.0.50:7860
LOCAL_TTS_PAYLOAD = {
    "ref_audio_input": "voices\laura.wav",  # Path to reference audio file
    "ref_text_input": "",  # Reference text for the TTS engine
    "gen_text_input": "",  # The text to generate speech for
    "remove_silence": False,  # Whether to remove silence in the generated speech
    "cross_fade_duration_slider": 0.16,  # Cross-fade duration for the generated speech
    "nfe_slider": 36,  # NFE steps for the TTS engine
    "speed_slider": .95  # Speed of the generated speech
}

CALENDAR_NOTIFICATION_SENTENCES = [
    "Heads up, you have {minutes} minutes until your {event} starts",
    "Just a reminder, {event} begins in {minutes} minutes",
    "In {minutes} minutes, you have {event} scheduled",
]
DEBUG_CALENDAR = False  # Control calendar debug messages
CALENDAR_NOTIFICATION_INTERVALS = [15, 10, 5, 2]  # Minutes before event to notify
