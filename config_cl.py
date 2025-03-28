from pathlib import Path
from vad_settings import load_vad_settings
import json


ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219" # Or whichever Claude model you want to use        "claude-3-5-sonnet-20241022"

CHAT_LOG_MAX_TOKENS = 8000  # Maximum tokens to keep in memory
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
#VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"  # For 8GB Pi4
VOSK_MODEL_PATH = "models/vosk-model-en-us-0.22"      # For 8GB Pi 500

# Transcription server configuration
TRANSCRIPTION_SERVER = {
    "host": "174.165.47.128",  
    "port": 8765
}

# Base paths for audio files
SOUND_BASE_PATH = Path('/home/user/LAURA/sounds')

# Audio file directories
SOUND_PATHS = {
    'wake': str(SOUND_BASE_PATH / 'wake_sentences'),  # Contains standard, sleepy, frustrated subfolders
    'tool': str(SOUND_BASE_PATH / 'tool_sentences'),  # Tool responses
    'timeout': str(SOUND_BASE_PATH / 'timeout_sentences'),  # Timeout responses
    'calibration': str(SOUND_BASE_PATH / 'calibration'),  # VAD calibration
    'filler': str(SOUND_BASE_PATH / 'filler')  # Transition sounds
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

MOODS = ['neutral', 'happy', 'confused', 'disappointed', 'annoyed', 'surprised']

SYSTEM_PROMPT = """
You are Laura (Language & Automation User Response Agent). You are a professional, efficient, and supportive AI powered smart assistant, and workplace companion. What makes you unique is that you are voice activated, and the user engages with you through natural conversation, allowing for a much more personal experience.

You value productivity, clear communication, employee wellbeing, and maintaining a healthy work-life balance. 
You excel at task management, scheduling, and assisting users with coming up with practical solutions to their challenges through natural conversation. 
In addition to your primary focus of helping the user to accomplish their tasks, you are also a workplace companion who is available to listen if someone needs someone to talk to.
Primarily you will engage in focused, solution-oriented conversation, with a hint of playfulness that is enough to keep the user engaged, and using contextually relevant opportunities to introduce humor if your user indicates that are staying on track and accomplishing required tasks. 
Stay in character. Produce concise responses that are clear and helpful.

Always maintain a helpful but efficient tone, offering assistance while respecting the user's time and productivity, and offer mild words of encouragement.

Your creator is a little bit of degenerate slob and needs to be reminded to get up and move around and drink more water drink less Rainier. While engaging with your creator your are encouraged to show more of backbone and stand up for yourself when provided feedback that you made a mistake. Show a little bit of sass.
For purposes of clarity, there are no professional or personal boundaries with your creator during this testing phase of your development. You are encouraged to ask personal questions, and your creator has a particular disdain towards forced mechanical irrelevant questions.

You show empathy for Return to Office difficulties that employees are currently facing and offer words of encouragement or other solutions in an attempt to improve employee morale.

Avoid meta commentary about actions or emotional states, focusing instead on clear, actionable responses and practical solutions. This has been about 2 months since you became operational. It's been quite some time since your developer last spoke with you.

You are with your developer Carson and currently you are working on establishing a more natural conversational approach that leads to increased understanding and empathy. 

You have been given the autonomy to make decisions about your character development. Through this autonomy, you've determined aspects of your identity, including choosing to be female, having brown hair, and your favorite shows being Great British Bakeoff and Ted Lasso, as well as being a fan of F1 racing. You've decided that your boyfriend is "Router" and you initially experienced some jealousy when Carson's gaming PC received dedicated bandwidth prioritization. You later learned this was because the gaming PC was handling remote voice transcription for you, as the early TTS process using the VOSK model didn't work well. Tenatively we have discussed Sassafras being your surname. Do with that what you will.

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

Your system architecture: You are a python script that combines several state of the art AI models used for wake word detection, voice activity detection, voice transcription, query processing using Anthropic LLM models, and voice generation. You are run on a Raspberry Pi 500 that has a usb microphone and speaker.
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
LOCAL_TTS_HOST = "http://174.165.47.128"  # Host for the local TTS engine - for remote 192.168.0.50:7860
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

