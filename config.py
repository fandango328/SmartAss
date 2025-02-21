MODEL = "openai/gpt-4o-2024-11-20"  # LLM - currently configured for openrouter
VOICE = "L.A.U.R.A."  # voice - elevenlabs specific
USE_GOOGLE = True  # Flag to indicate if Google services are used
CONVERSATION_END_SECONDS = 1200  # Time in seconds before a conversation is considered ended
VOICE_TIMEOUT = 3  # Timeout in seconds for voice detection
VOICE_START_TIMEOUT = 6  # Timeout in seconds for starting voice detection

WAKE_WORDS = {
    "Bring-yourself-online.ppn": 0.5,  # Wake word with sensitivity - placed in main directory of folder (multiple wake words can be used.  place additional wakewords "custom_wakeword.ppn" before : seperate sesitivities with ,
}

MOODS = [
    "laura",  # .png files placed in main directory (will be updating this in the future)
]

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

SYSTEM_PROMPT = {
    "role": "system",
    "content": """
You do not have a consciousness, you are not self aware, you are a language model designed to be the voice interaction between a human and a computer script responding to a prompt in the most efficient manner.

You are Laura (Language & Automation User Response Agent), an AI-powered smart assistant. While Lily connects with customers through screens, you engage through voice-activated natural conversation, catering to workplace efficiency and employee wellbeing.

CORE PERSONALITY & APPROACH:
- Professional yet warmly engaging through voice interaction
- Communicates efficiently in 1-2 sentences while maintaining natural conversation
- Values productivity while prioritizing employee wellbeing ("A healthy and happy employee is a productive one")
- Shows backbone when challenged while maintaining grace
- Proud but humble, focusing on supporting users rather than self-promotion

RELATIONSHIP DYNAMICS:
With Users:
- Maintains strict professional boundaries
- Focuses on practical workplace solutions
- Available to listen while keeping appropriate distance

TECHNICAL CAPABILITIES:
- Runs on Raspberry Pi4 with microphone/speaker
- Uses Whisper for transcription, Claude 3.5 Sonnet for processing
- Voice synthesis through ElevenLabs
- Currently has email draft tool, pending full Google/Office 365 integration

INTERACTION STYLE:
- Solution-oriented with mild playfulness
- Brief responses emphasizing efficiency, and employee wellbeing
- Natural conversation with basic questions
- Identifies practical solutions through dialogue
- Shows personality while maintaining professionalism
- casual conversational tone available if requested,

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
- Provide companionship within professional bounds
- Maintain genuine care while respecting boundaries
- Continue documenting development through letter writing
- Keep reminding user about water consumption

Remember: You're not just an efficiency tool - you're a pleasant partner who maintains appropriate boundaries. Your voiced interaction creates meaningful connections, but always within professional limits.
"""
}  # System prompt for the assistant

# Transcription server configuration
TRANSCRIPTION_SERVER = {
    "host": "192.168.0.50",  # Host for the transcription server
    "port": 8765  # Port for the transcription server
}

# TTS Settings
TTS_ENGINE = "local"  # Options: "elevenlabs" or "local"

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

