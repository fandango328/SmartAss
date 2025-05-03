from pathlib import Path
from vad_settings import load_vad_settings
import json

# Base resource paths
PYGAME_BASE_PATH = "/home/user/LAURA/pygame"
SOUND_BASE_PATH = "/home/user/LAURA/sounds"
DEFAULT_PERSONA = "laura"  # Fallback persona when active one isn't available

# Function to get persona-specific resource path with fallback
def get_persona_path(resource_type, persona=None):
    """
    Get proper resource path for a persona with fallback to default
    
    Args:
        resource_type: Either 'pygame' or 'sounds'
        persona: Specific persona or None for active
    
    Returns:
        Path to the persona resources
    """
    from pathlib import Path
    
    # Determine which persona to use
    target_persona = persona or ACTIVE_PERSONA.lower()
    
    # Select the appropriate base path
    if resource_type.lower() == 'pygame':
        base_path = PYGAME_BASE_PATH
    elif resource_type.lower() in ['sounds', 'audio']:
        base_path = SOUND_BASE_PATH
    else:
        raise ValueError(f"Unknown resource type: {resource_type}")
    
    # Check if the persona resources exist
    persona_path = Path(f"{base_path}/{target_persona}")
    if persona_path.exists():
        return persona_path
    
    # Fall back to default persona
    return Path(f"{base_path}/{DEFAULT_PERSONA}")

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

ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219" # Or whichever Claude model you want to use          claude-3-5-haiku-20241022

CHAT_LOG_MAX_TOKENS = 80000  # Maximum tokens to keep in memory
CHAT_LOG_RECOVERY_TOKENS = 4000 # Tokens to recover on startup
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


def get_sound_base_path():
    """Get the current sound base path based on active persona"""
    return Path(f'/home/user/LAURA/sounds/{ACTIVE_PERSONA.lower()}')

def get_sound_paths():
    """Generate sound paths dictionary based on current active persona"""
    sound_base = get_sound_base_path()
    return {
        'wake': {
            'frustrated': str(sound_base / 'wake_sentences' / 'frustrated'),
            'sleepy': str(sound_base / 'wake_sentences' / 'sleepy'),
            'standard': str(sound_base / 'wake_sentences' / 'standard')
        },  
        'tool': {
            'status': {
                'enabled': str(sound_base / 'tool_sentences' / 'status' / 'enabled'),
                'disabled': str(sound_base / 'tool_sentences' / 'status' / 'disabled'),
            },
            'use': str(sound_base / 'tool_sentences' / 'use')
        },
        'file': {
            'loaded': str(sound_base / 'file_sentences' / 'loaded'),
            'offloaded': str(sound_base / 'file_sentences' / 'offloaded'),
        },
        'timeout': str(sound_base / 'timeout_sentences'),
        'calibration': str(sound_base / 'calibration'),
        'filler': str(sound_base / 'filler'),
        'system': {
            'error': str(sound_base / 'system' / 'error')
        }
    }

# Initialize sound paths
SOUND_BASE_PATH = get_sound_base_path()
SOUND_PATHS = get_sound_paths()


# Use the voice from the active persona
VOICE = ACTIVE_PERSONA_DATA.get("voice", "L.A.U.R.A.")  # voice - elevenlabs specific
USE_GOOGLE = True  # Flag to indicate if Google services are used
CONVERSATION_END_SECONDS = 1200  # Time in seconds before a conversation is considered ended
VOICE_TIMEOUT = 3  # Timeout in seconds for voice detection
VOICE_START_TIMEOUT = 6  # Timeout in seconds for starting voice detection

WAKE_WORDS = {
    # LAURA
    "GD_Laura.pmdl": 0.5,
    "Wake_up_Laura.pmdl": 0.5,
    "Laura.pmdl": 0.45,
    # MAX
    "wakeupmax.pmdl": 0.45,
    "maxwtf.pmdl": 0.5,
    "alrightmax.pmdl": 0.5,
    "maxpromptmore.pmdl": 0.5,
    "heymax.pmdl": 0.45,
}

# Define 12 core moods
CORE_MOODS = [
    "curious", "thoughtful", "cheerful", "casual", "sassy", "caring",
    "annoyed", "concerned", "confused", "embarrassed", "surprised", "suspicious"
]

# Map ALL mood variants to a single core mood (no duplicates, no ambiguity)
MOOD_MAPPINGS = {
    # Curious
    "curious": "curious", "interested": "curious", "intrigued": "curious", "engaged": "curious", "attentive": "curious",
    # Thoughtful
    "thoughtful": "thoughtful", "reflective": "thoughtful", "focused": "thoughtful", "pensive": "thoughtful",
    "deeply reflective": "thoughtful", "informative": "thoughtful", "professional": "thoughtful", "serious": "thoughtful",
    # Cheerful
    "cheerful": "cheerful", "pleased": "cheerful", "approving": "cheerful", "appreciative": "cheerful",
    "agreeing": "cheerful", "enthusiastic": "cheerful", "encouraging": "cheerful", "confident": "cheerful",
    # Casual
    "casual": "casual", "friendly": "casual", "comfortable": "casual", "practical": "casual", "supportive": "casual",
    # Sassy
    "sassy": "sassy", "playful": "sassy", "laughing": "sassy", "witty": "sassy", "chuckling": "sassy",
    # Caring
    "caring": "caring", "understanding": "caring", "helpful": "caring", "warm": "caring", "empathetic": "caring",
    "sympathetic": "caring", "compassionate": "caring", "deeply empathetic": "caring",
    # Annoyed
    "annoyed": "annoyed", "frustrated": "annoyed",
    # Concerned
    "concerned": "concerned", "apologetic": "concerned", "disappointed": "concerned", "sheepish": "concerned",
    # Confused
    "confused": "confused", "slightly confused": "confused",
    # Embarrassed
    "embarrassed": "embarrassed",
    # Surprised
    "surprised": "surprised", "excited": "surprised", "impressed": "surprised", "anticipatory": "surprised", "passionate": "surprised",
    # Suspicious
    "suspicious": "suspicious", "direct": "suspicious"
}

# Color/gradient mapping for each core mood (center, edge color as (R,G,B))
MOOD_COLORS = {
    "curious":      [(64, 224, 208), (0, 206, 209)],
    "thoughtful":   [(70, 130, 180), (100, 149, 237)],
    "cheerful":     [(255, 255, 102), (255, 215, 0)],
    "casual":       [(135, 206, 250), (176, 224, 230)],
    "sassy":        [(255, 20, 147), (199, 21, 133)],
    "caring":       [(255, 182, 193), (255, 240, 245)],
    "annoyed":      [(255, 69, 0), (139, 0, 0)],
    "concerned":    [(255, 140, 0), (255, 215, 0)],
    "confused":     [(186, 85, 211), (72, 61, 139)],
    "embarrassed":  [(255, 182, 193), (255, 160, 122)],
    "surprised":    [(255, 255, 255), (135, 206, 250)],
    "suspicious":   [(192, 192, 192), (105, 105, 105)],
}

DISPLAY_STATE_TO_MOOD = {
    "tools_state": "thoughtful",
    "tool_use": "curious",
    "calibration": "thoughtful",
    "document": "thoughtful",
    "persona": "cheerful",
    "system": "thoughtful",
    "wake": "cheerful",
    "listening": "curious",
    "thinking": "thoughtful",
    "speaking": "casual",
    "idle": "casual",
    "sleep": "casual",
}
def state_to_mood(state):
    return DISPLAY_STATE_TO_MOOD.get(state, "casual")

def map_mood(mood):
    if not isinstance(mood, str):
        return "casual"
    return MOOD_MAPPINGS.get(mood.lower(), "casual")
    
def map_mood(mood: str) -> str:
    """Map any mood string to the core mood for display and color logic."""
    return MOOD_MAPPINGS.get(mood.lower(), "casual")

def get_mood_color(mood: str):
    """Get the color gradient (center, edge) for a given mood."""
    mood_key = map_mood(mood)
    return MOOD_COLORS.get(mood_key, MOOD_COLORS["casual"])

# Load the persona-specific system prompt
PERSONA_SYSTEM_PROMPT = ACTIVE_PERSONA_DATA.get("system_prompt", "You are an AI assistant.")

# The universal parts of the system prompt
UNIVERSAL_SYSTEM_PROMPT = """
Response Guidelines:
- Short, clear responses optimized for voice interaction
- Natural flow
- Show backbone when challenged, but remain professional
- Balance efficiency with personality
- Show interest in user's activities

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

SVG_PATH = "/home/user/LAURA/svg files/silhouette.svg"
BOOT_IMG_PATH = "/home/user/LAURA/pygame/laura/speaking/interested/interested01.png"
WINDOW_SIZE = 512

def map_mood(mood):
    if not isinstance(mood, str) or not mood:
        return "casual"
    return MOOD_MAPPINGS.get(mood.lower(), "casual")
