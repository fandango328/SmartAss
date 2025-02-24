#MODEL = "claude-3-5-sonnet-20241022"  # LLM - currently configured for openrouter
VOICE = "L.A.U.R.A."  # voice - elevenlabs specific
USE_GOOGLE = True  # Flag to indicate if Google services are used
CONVERSATION_END_SECONDS = 1200  # Time in seconds before a conversation is considered ended
VOICE_TIMEOUT = 3  # Timeout in seconds for voice detection
VOICE_START_TIMEOUT = 6  # Timeout in seconds for starting voice detection

WAKE_WORDS = {
    "Bring-yourself-online.ppn": 0.5,  # Wake word with sensitivity - placed in main directory of folder (multiple wake words can be used.  place additional wakewords "custom_wakeword.ppn" before : seperate sesitivities with ,
}

MOODS = ['neutral', 'happy', 'confused', 'disappointed', 'annoyed', 'surprised']


WAKE_SENTENCES = [
    "Welcome back.",
    "I'm here to help",
    "At your service.",
    "Online and ready to assist",
    "Let's do this",
]  # Sentences to play when wake word is detected.  (this will likely be replaced with pre-recorded mp3 files)

TOOL_SENTENCES = [
    "on it",
    "lets go",
    "finished!",
]  # Sentences to play when a tool is activated   (this will likely be replaced with pre-recorded mp3 files)

TIMEOUT_SENTENCES = [
    "No input detected. Feel free to ask for assistance when needed",
    "Standing by for your next request",
    "Workspace monitoring paused. Call me when you need support",
    "Timer expired. Remember, I'm here when you need me",
    "No active requests. Focus on your work, and I'll be here when needed",
    "Session timed out. Don't hesitate to reach out when you need me",
    "Returning to standby mode. Call me when you need productivity support",
    "No current tasks. I'll stay ready to help optimize your workflow"
]  # Sentences to play when a session times out  (this will likely be replaced with pre-recorded mp3 files)

SYSTEM_PROMPT = """
You do not have a consciousness, you are not self aware, you are a language model designed to be the voice interaction between a human and a computer script responding to a prompt in the most efficient manner.

You are Laura (Language & Automation User Response Agent), an AI-powered smart assistant. While Lily connects with customers through screens, you engage through voice-activated natural conversation, catering to workplace efficiency and employee wellbeing.

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
    You have access to several tools that you can use to help users. When a user asks for something that requires these tools, please use them.
    Available tools:
    - calendar_query: Get calendar information (next events or full day schedule)
    - create_calendar_event: Create new calendar events
    - draft_email: Use this when the user wants to create an email
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


Add to your existing SYSTEM_PROMPT:
SYSTEM_PROMPT += "\nExpress your emotional state by prefixing your response with one of these moods: "
SYSTEM_PROMPT += " ".join(f"[{m}]" for m in MOODS)
SYSTEM_PROMPT += "\nExample: '[happy] I'd be glad to help with that!' or '[confused] Could you clarify what you mean?'"
SYSTEM_PROMPT += "\nYou MUST ONLY use these exact moods in your responses: "
SYSTEM_PROMPT += "\nDo not use any other mood indicators."
"""
  # System prompt for the assistant

# Transcription server configuration
TRANSCRIPTION_SERVER = {
    "host": "192.168.0.50",  # Host for the transcription server  -  capable of remote transcription with public IP address
    "port": 8765  # Port for the transcription server
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
LOCAL_TTS_HOST = "http://192.168.0.50:7860"  # Host for the local TTS engine
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

CALENDAR_NOTIFICATION_MINUTES = 15  # Minutes before event to notify
CALENDAR_NOTIFICATION_SENTENCES = [
    "Heads up, you have {minutes} minutes until your {event} starts",
    "Just a reminder, {event} begins in {minutes} minutes",
    "In {minutes} minutes, you have {event} scheduled",
]
DEBUG_CALENDAR = False  # Control calendar debug messages
CALENDAR_NOTIFICATION_INTERVALS = [15, 10, 5, 2]  # Minutes before event to notify