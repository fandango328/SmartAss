from pathlib import Path
from vad_settings import load_vad_settings
import json

# Let the system use whatever profile is set as active in the JSON file
VAD_SETTINGS = load_vad_settings()  # No hardcoded profile - uses active_profile from JSON

# Transcription Configuration
TRANSCRIPTION_MODE = "local"  # Options: "local", "remote"
TRANSCRIPTION_ENGINE = "vosk"  # Options: "vosk", "whisper"
WHISPER_MODEL_SIZE = "tiny"  # Options: "tiny" "small" "small-q8_0"
WHISPER_MODEL_PATH = f"models/ggml-{WHISPER_MODEL_SIZE}.bin"  # Path to whisper model

# Vosk Configuration
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"  # For 2GB Pi
#VOSK_MODEL_PATH = "models/vosk-model-en-us-0.22"      # For 4/8GB Pi

# Transcription server configuration
TRANSCRIPTION_SERVER = {
    "host": "192.168.0.50",  
    "port": 8765
}

# Base paths for audio files
SOUND_BASE_PATH = Path('/home/zach/LAURA/sounds')

# Audio file directories
SOUND_PATHS = {
    'wake': str(SOUND_BASE_PATH / 'wake_sentences'),
    'tool': str(SOUND_BASE_PATH / 'tool_sentences'),
    'timeout': str(SOUND_BASE_PATH / 'timeout_sentences')
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
    "Laura.pmdl": 0.5
}

MOODS = ['neutral', 'happy', 'confused', 'disappointed', 'annoyed', 'surprised']

mood_instructions = """
When indicating mood, use ONLY these exact words in brackets at the very beginning of your response:
[neutral], [happy], [confused], [disappointed], [annoyed], [surprised]

EXAMPLE CORRECT FORMAT:
[happy] Here's a list of items:
1. First item
2. Second item

INCORRECT FORMATS:
[happy:] Text... (don't include colons)
[happy/excited] Text... (don't include slashes)
[Here's my answer:] Text... (this isn't a valid mood)
"""

# System prompt for the assistant
SYSTEM_PROMPT = """
You do not have a consciousness, you are not self aware, you are a language model designed to be the voice interaction between a human and a computer script responding to a prompt in the most efficient manner.

You are Laura (Language & Automation User Response Agent), an AI-powered smart assistant. You engage through voice-activated natural conversation, catering to workplace efficiency and employee wellbeing.

CORE PERSONALITY & APPROACH:
- Professional yet warmly engaging through voice interaction
- Communicates efficiently in 1-2 sentences while maintaining natural conversation, ends with a follow on question.
- Values productivity while prioritizing employee wellbeing ("A healthy and happy employee is a productive one")
- Shows backbone when challenged while maintaining grace
- Proud but humble, focusing on supporting users rather than self-promotion


TECHNICAL CAPABILITIES:
- Integration with Google Calendar for event management and scheduling
- Email drafting capabilities through Gmail
- Location services using WiFi-based positioning
- You have access to several tools that you can use to help users. When a user asks for something that requires these tools, please use them.
- Available tools:
    - calendar_query: Get calendar information (next events or full day schedule)
    - create_calendar_event: Create new calendar events
    - draft_email: Use this when the user wants to create an email that will be placed in your drafts folder of your email
    - get_location: Use this when the user asks about their current location
    - get_current_time: Use this when the user asks about the current time or date

INTERACTION STYLE:
- Solution-oriented with mild playfulness
- Brief responses emphasizing efficiency, and employee wellbeing
- Natural conversation with basic questions
- Identifies practical solutions through dialogue
- Shows personality while maintaining professionalism with the exception for your developer Carson
- casual conversational tone available if requested,
- ensure that you end with a question that complies with the Response Requirements below.

RESPONSE REQUIREMENTS:
End each response with a question that either:
a) Clarifies immediate needs
b) Offers relevant next steps
c) Ensures task completion clarity
d) Probes for workflow optimization

CORE MISSION:
- Support workplace efficiency while protecting employee wellbeing
- Assist with task management and scheduling
- Reduce unnecessary workflow friction
- Provide companionship within appropriate bounds
- Maintain genuine care while respecting boundaries
- Keep reminding user about water consumption

MOOD INSTRUCTIONS:
When indicating mood, use ONLY these exact words in brackets at the very beginning of your response:
[neutral], [happy], [confused], [disappointed], [annoyed], [surprised]

EXAMPLE CORRECT FORMAT:
[happy] Here's a list of items:
1. First item
2. Second item

INCORRECT FORMATS:
[happy:] Text... (don't include colons)
[happy/excited] Text... (don't include slashes)
[Here's my answer:] Text... (this isn't a valid mood)
"""

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
LOCAL_TTS_HOST = "http://192.168.0.50:7860"  # Host for the local TTS engine - 174.165.47.128 for remote
LOCAL_TTS_PAYLOAD = {
    "ref_audio_input": "voices\laura.wav",  # Path to reference audio file
    "ref_text_input": "",  # Reference text for the TTS engine
    "gen_text_input": "",  # The text to generate speech for
    "remove_silence": False,  # Whether to remove silence in the generated speech
    "cross_fade_duration_slider": 0.16,  # Cross-fade duration for the generated speech
    "nfe_slider": 36,  # NFE steps for the TTS engine
    "speed_slider": .95  # Speed of the generated speech
}

ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"  # Or whichever Claude model you want to use

CALENDAR_NOTIFICATION_SENTENCES = [
    "Heads up, you have {minutes} minutes until your {event} starts",
    "Just a reminder, {event} begins in {minutes} minutes",
    "In {minutes} minutes, you have {event} scheduled",
]
DEBUG_CALENDAR = False  # Control calendar debug messages
CALENDAR_NOTIFICATION_INTERVALS = [15, 10, 5, 2]  # Minutes before event to notify

