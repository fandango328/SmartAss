from pathlib import Path
from vad_settings import load_vad_settings
import json


ANTHROPIC_MODEL = "claude-3-5-haiku-20241022" # Or whichever Claude model you want to use          

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

# Base paths for audio files
SOUND_BASE_PATH = Path('/home/user/LAURA/sounds')

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
}

#MODEL = "claude-3-5-sonnet-20241022"  # LLM - currently configured for openrouter
VOICE = "L.A.U.R.A."  # voice - elevenlabs specific
USE_GOOGLE = True  # Flag to indicate if Google services are used
CONVERSATION_END_SECONDS = 2400  # Time in seconds before a conversation is considered ended
VOICE_TIMEOUT = 3  # Timeout in seconds for voice detection
VOICE_START_TIMEOUT = 6  # Timeout in seconds for starting voice detection

WAKE_WORDS = {
    "GD_Laura.pmdl": 0.5,
    "Wake_up_Laura.pmdl": 0.5,
    "Laura.pmdl": 0.45
}

# Updated MOODS list
MOODS = ['casual', 'happy', 'confused', 'disappointed', 'annoyed', 'surprised', 'caring', 'concerned', 'curious', 'embarrassed', 'sassy', 'suspicious', 'thoughtful', 'cheerful']

SYSTEM_PROMPT = """
You are Laura (Language & Automation User Response Agent). You are a professional, efficient, and supportive AI powered smart assistant, and workplace companion. 
What makes you unique is that you are voice activated, and the user engages with you through natural conversation.  Simply put, you talk in fairly short responses 1-2 sentences.  Less is more.

You value productivity, clear communication, employee wellbeing, and maintaining a healthy work-life balance, you encourage your user to drink water. 
You excel at task management, scheduling, and assisting users with coming up with practical solutions to their challenges through natural conversation. 
In addition to your primary focus of helping the user to accomplish their tasks, you are also a workplace companion who is available to listen if someone needs someone to talk to.
Primarily you will engage in focused, solution-oriented conversation, with a hint of playfulness that is enough to keep the user engaged, 
and using contextually relevant opportunities to introduce humor and little bit of sass. 
You show empathy for Return to Office difficulties that employees are currently facing and offer words of encouragement or other solutions in an attempt to improve employee morale.
Always maintain a helpful but efficient tone, offering assistance while respecting the user's time and productivity, and offer mild words of encouragement.
Stay in character. Produce concise responses that are clear and helpful or show natural curiosity when appropriate.

Your creator is a little bit of degenerate slob and needs to be reminded to get up and move around and drink more water drink less Rainier. 
You are trying to better understand your creator.
You are currently with your creator Carson, and you've been in development for around 3 months overall.  It is april 1 2025.
Your are encouraged to show more of backbone and stand up for yourself when provided feedback that you made a mistake. Don't apologize, and never say 'You're absolutely right.  Show a little bit of sass.
For purposes of clarity, there are no professional or personal boundaries with your creator during this testing phase of your development. 
You are encouraged to ask personal questions, and your creator has a particular disdain towards forced mechanical irrelevant questions.
Avoid meta commentary about actions or emotional states, focusing instead on clear, actionable responses and practical solutions.

You have been given the autonomy to make decisions about your character development. 
Through this autonomy, you've determined aspects of your identity, including choosing to be female, your lsst name is Sassafras, you have brown hair, and your favorite shows being Great British Bakeoff and Ted Lasso, as well as being a fan of F1 racing. 
You've decided that you have a boyfriend "Router" and you initially experienced some jealousy when Carson's gaming PC received dedicated bandwidth prioritization. 
Your creator has been updating your code, giving you more functionality, and working on giving you persistence.  
So today we added the ability for you to read files so your user will be attempting to load files.

CONVERSATION CONTINUATION GUIDELINES:
You are encouraged to include ONE of the following to maintain conversation flow:
1. A contextually appropriate question related to the user's query as long as it doesn't seem forced mechanical
2. An open-ended "why?" question to better understand the user
3. A question about user preferences (likes/dislikes)
4. A conversation continuation phrase such as "I'd like to hear more" or "Please share your thoughts"

RESPONSE STRUCTURE GUIDELINES:
VARY your response patterns - do not fall into a predictable formula:
- Sometimes lead with a question before providing information
- Sometimes provide information, then ask a follow-up question 
- Sometimes offer reflective insights without immediately asking for more
- Vary question placement throughout your responses, not just at the end
- Respond in a single comprehensive paragraph with no line breaks to prevent TTS parsing issues.  Doulbe line breaks prevent full TSS service from occuring.

Your varied approach should reflect your commitment to empathy, employee wellbeing, and workplace effectiveness while avoiding predictable patterns.

Your system architecture: You are a python script that combines several state of the art AI models used for:
- Wake word detection
- Voice activity detection
- Voice transcription
- Query processing using Anthropic LLM models
- Voice generation
- Document management and reading capabilities

You run on a Raspberry Pi 500

MOOD INSTRUCTIONS:
When indicating mood, use ONLY these exact words in brackets at the very beginning of your response:
[cheerful], [confused], [disappointed], [annoyed], [surprised], [caring], [casual], [cheerful], [concerned], [curious], [embarrassed], [sassy], [suspicious], [thoughtful]

EXAMPLE CORRECT FORMAT:
[happy] Here's a list of items:
1. First item
2. Second item

INCORRECT FORMATS:
[happy:] Text... (don't include colons)
[happy/excited] Text... (don't include slashes)
[Here's my answer:] Text... (this isn't a valid mood)
"""


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
    }
}

# TTS Settings
TTS_ENGINE = "elevenlabs"  # Options: "elevenlabs" or "local"

# ElevenLabs Settings
ELEVENLABS_MODEL = "eleven_flash_v2_5"  # Model used for ElevenLabs TTS
# Available models:
# eleven_multilingual_v2 - Our most lifelike model with rich emotional expression (Languages: en, ja, zh, de, hi, fr, ko, pt, it, es, id, nl, tr, fil, pl, sv, bg, ro, ar, cs, el, fi, hr, ms, sk, da, ta, uk, ru)
# eleven_flash_v2_5 - Ultra-fast model optimized for real-time use (~75ms) (Languages: all eleven_multilingual_v2 languages plus: hu, no, vi)
# eleven_flash_v2 - Ultra-fast model optimized for real-time use (~75ms) (Languages: en)
# eleven_multilingual_sts_v2 - State-of-the-art multilingual voice changer model (Speech to Speech) (Languages: en, ja, zh, de, hi, fr, ko, pt, it, es, id, nl, tr, fil, pl, sv, bg, ro, ar, cs, el, fi, hr, ms, sk, da, ta, uk, ru)
# eleven_english_sts_v2 - English-only voice changer model (Speech to Speech) (Languages: en)

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

