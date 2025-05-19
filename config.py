from pathlib import Path
import json
import os

# =======================
# FREQUENTLY UPDATED: STT, CHATLOG/SESSION, LLM, TTS
# =======================

# --- LLM MODEL SELECTION (Multi-provider: Anthropic/OpenAI/...) ---
MODELS_FILE = os.path.join("llm_integrations", "models.json")
try:
    with open(MODELS_FILE, 'r') as f:
        MODELS_DATA = json.load(f)
    ACTIVE_PROVIDER = MODELS_DATA.get("active_provider", "anthropic")

    # Anthropic
    ANTHROPIC_MODELS = MODELS_DATA.get("anthropic_models", {})
    ACTIVE_ANTHROPIC_MODEL_KEY = MODELS_DATA.get("active_anthropic_model", "claude-3-7-sonnet")
    ACTIVE_ANTHROPIC_MODEL = ANTHROPIC_MODELS.get(ACTIVE_ANTHROPIC_MODEL_KEY, {})
    ANTHROPIC_MODEL = ACTIVE_ANTHROPIC_MODEL.get("api_name", "claude-3-7-sonnet-20250219")
    ANTHROPIC_MODEL_ALIAS = ACTIVE_ANTHROPIC_MODEL.get("api_alias", None)
    ANTHROPIC_MODEL_MAX_TOKENS = ACTIVE_ANTHROPIC_MODEL.get("max_tokens", 200000)

    # OpenAI
    OPENAI_MODELS = MODELS_DATA.get("openai_models", {})
    ACTIVE_OPENAI_MODEL_KEY = MODELS_DATA.get("active_openai_model", "gpt-4.1")
    ACTIVE_OPENAI_MODEL = OPENAI_MODELS.get(ACTIVE_OPENAI_MODEL_KEY, {})
    OPENAI_MODEL = ACTIVE_OPENAI_MODEL.get("api_name", "gpt-4.1")

    # --- Common LLM Parameters (for main loop and adapters) ---
    MAX_TOKENS = MODELS_DATA.get("max_tokens", 4096)
    TEMPERATURE = MODELS_DATA.get("temperature", 0.7)

except Exception as e:
    print(f"Error loading models configuration: {e}")
    # Safe fallbacks for both providers
    ACTIVE_PROVIDER = "anthropic"
    ANTHROPIC_MODELS = {}
    ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219"
    ANTHROPIC_MODEL_ALIAS = None
    ANTHROPIC_MODEL_MAX_TOKENS = 200000
    OPENAI_MODELS = {}
    OPENAI_MODEL = "gpt-4.1"
    MAX_TOKENS = 4096
    TEMPERATURE = 0.7

# --- API KEYS (from secret.py, not from models.json!) ---
try:
    import secret
    OPENAI_API_KEY = getattr(secret, "OPENAI_API_KEY", None)
    ANTHROPIC_API_KEY = getattr(secret, "ANTHROPIC_API_KEY", None)
except ImportError:
    print("Warning: Could not import secret.py, API keys not loaded.")
    OPENAI_API_KEY = None
    ANTHROPIC_API_KEY = None

# --- Dynamic API key selection for provider ---
def get_active_api_key():
    if ACTIVE_PROVIDER == "openai":
        return OPENAI_API_KEY
    elif ACTIVE_PROVIDER == "anthropic":
        return ANTHROPIC_API_KEY
    else:
        return None


# =======================
# STT (Speech-To-Text)
# =======================
VOSK_MODEL_OPTIONS = {
    "small": "models/vosk-model-small-en-us-0.15",
    "medium": "models/vosk-model-en-us-0.22",
    "large": "models/vosk-model-large-en-us-0.22"
}
VOSK_MODEL_SIZE = "small"  # <-- QUICK SELECT (change to "medium" or "large" as desired)
VOSK_MODEL_PATH = VOSK_MODEL_OPTIONS[VOSK_MODEL_SIZE]
WHISPER_MODEL_SIZE = "tiny"                             # UPDATE FREQUENTLY
WHISPER_MODEL_PATH = f"models/ggml-{WHISPER_MODEL_SIZE}.bin"  # UPDATE FREQUENTLY
TRANSCRIPTION_MODE = "local"  # Options: "local", "remote"
TRANSCRIPTION_ENGINE = "vosk"  # Options: "vosk", "whisper"
from vad_settings import load_vad_settings
VAD_SETTINGS = load_vad_settings()  # Uses active_profile from JSON

TRANSCRIPTION_SERVER = {
    "host": "xxx.xxx.xx.xxx",  
    "port": 8765
}

# =======================
# Chatlog / Session Management
# =======================
CHAT_LOG_MAX_TOKENS = 80000      # UPDATE FREQUENTLY
CHAT_LOG_RECOVERY_TOKENS = 4000  # UPDATE FREQUENTLY
CHAT_LOG_DIR = "chat_logs"       # UPDATE FREQUENTLY
CONVERSATION_END_SECONDS = 1200  # UPDATE FREQUENTLY

# =======================
# TTS (Text-To-Speech)
# =======================
TTS_ENGINE = "elevenlabs"  # "elevenlabs" or "local" -- UPDATE FREQUENTLY
VOICE = "L.A.U.R.A."       # ElevenLabs specific -- UPDATE FREQUENTLY
ELEVENLABS_MODEL = "eleven_flash_v2_5"  # UPDATE FREQUENTLY
LOCAL_TTS_HOST = "http://xxx.xxx.xx.xxx"
LOCAL_TTS_PAYLOAD = {
    "ref_audio_input": "voices/laura.wav",
    "ref_text_input": "",
    "gen_text_input": "",
    "remove_silence": False,
    "cross_fade_duration_slider": 0.16,
    "nfe_slider": 36,
    "speed_slider": .95
}

VOICE_TIMEOUT = 3
VOICE_START_TIMEOUT = 6
USE_GOOGLE = False

# =======================
# SYSTEM PROMPT & MOODS
# =======================

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

# If you want persona-specific voice, override here
if "voice" in ACTIVE_PERSONA_DATA:
    VOICE = ACTIVE_PERSONA_DATA["voice"]

PERSONA_SYSTEM_PROMPT = ACTIVE_PERSONA_DATA.get("system_prompt", "You are an AI assistant.")
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
SYSTEM_PROMPT = f"{PERSONA_SYSTEM_PROMPT}\n\n{UNIVERSAL_SYSTEM_PROMPT}"

# Mood/core color mappings
CORE_MOODS = [
    "curious", "thoughtful", "cheerful", "casual", "sassy", "caring",
    "annoyed", "concerned", "confused", "embarrassed", "surprised", "suspicious"
]

MOOD_MAPPINGS = {
    "curious": "curious", "interested": "curious", "intrigued": "curious", "engaged": "curious", "attentive": "curious",
    "thoughtful": "thoughtful", "reflective": "thoughtful", "focused": "thoughtful", "pensive": "thoughtful",
    "deeply reflective": "thoughtful", "informative": "thoughtful", "professional": "thoughtful", "serious": "thoughtful",
    "cheerful": "cheerful", "pleased": "cheerful", "approving": "cheerful", "appreciative": "cheerful",
    "agreeing": "cheerful", "enthusiastic": "cheerful", "encouraging": "cheerful", "confident": "cheerful",
    "casual": "casual", "friendly": "casual", "comfortable": "casual", "practical": "casual", "supportive": "casual",
    "sassy": "sassy", "playful": "sassy", "laughing": "sassy", "witty": "sassy", "chuckling": "sassy",
    "caring": "caring", "understanding": "caring", "helpful": "caring", "warm": "caring", "empathetic": "caring",
    "sympathetic": "caring", "compassionate": "caring", "deeply empathetic": "caring",
    "annoyed": "annoyed", "frustrated": "annoyed",
    "concerned": "concerned", "apologetic": "concerned", "disappointed": "concerned", "sheepish": "concerned",
    "confused": "confused", "slightly confused": "confused",
    "embarrassed": "embarrassed",
    "surprised": "surprised", "excited": "surprised", "impressed": "surprised", "anticipatory": "surprised", "passionate": "surprised",
    "suspicious": "suspicious", "direct": "suspicious"
}

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
    if not isinstance(mood, str) or not mood:
        return "casual"
    return MOOD_MAPPINGS.get(mood.lower(), "casual")

def get_mood_color(mood):
    mood_key = map_mood(mood)
    return MOOD_COLORS.get(mood_key, MOOD_COLORS["casual"])

# =======================
# GLOBAL PATHS, PERSONA, RESOURCES
# =======================

PYGAME_BASE_PATH = "/home/user/LAURA/pygame"
SOUND_BASE_PATH = "/home/user/LAURA/sounds"
DEFAULT_PERSONA = "laura"

def get_persona_path(resource_type, persona=None):
    target_persona = persona or ACTIVE_PERSONA.lower()
    if resource_type.lower() == 'pygame':
        base_path = PYGAME_BASE_PATH
    elif resource_type.lower() in ['sounds', 'audio']:
        base_path = SOUND_BASE_PATH
    else:
        raise ValueError(f"Unknown resource type: {resource_type}")
    persona_path = Path(f"{base_path}/{target_persona}")
    if persona_path.exists():
        return persona_path
    return Path(f"{base_path}/{DEFAULT_PERSONA}")

SVG_PATH = "/home/user/LAURA/svg files/silhouette.svg"
BOOT_IMG_PATH = "/home/user/LAURA/pygame/laura/speaking/interested/interested01.png"
WINDOW_SIZE = 512

# =======================
# SOUNDS & NOTIFICATIONS
# =======================

def get_sound_base_path():
    return Path(f'/home/user/LAURA/sounds/{ACTIVE_PERSONA.lower()}')

def get_sound_paths():
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

SOUND_BASE_PATH = get_sound_base_path()
SOUND_PATHS = get_sound_paths()

CALENDAR_NOTIFICATION_SENTENCES = [
    "Heads up, you have {minutes} minutes until your {event} starts",
    "Just a reminder, {event} begins in {minutes} minutes",
    "In {minutes} minutes, you have {event} scheduled",
]
DEBUG_CALENDAR = False
CALENDAR_NOTIFICATION_INTERVALS = [15, 10, 5, 2]

# =======================
# SYSTEM COMMANDS & WAKEWORDS
# =======================

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

WAKE_WORDS = {
    "GD_Laura.pmdl": 0.5,
    "Wake_up_Laura.pmdl": 0.5,
    "Laura.pmdl": 0.45,
    "wakeupmax.pmdl": 0.45,
    "maxwtf.pmdl": 0.5,
    "alrightmax.pmdl": 0.5,
    "maxpromptmore.pmdl": 0.5,
    "heymax.pmdl": 0.45,
}
# Place any rarely updated globals or advanced features at the bottom.
