import json
from pathlib import Path

# ========== Paths ==========
BASE_PATH = Path("/home/user/LAURA")
SOUND_BASE_PATH = BASE_PATH / "sounds"
SVG_PATH = BASE_PATH / "svg files/silhouette.svg"
BOOT_IMG_PATH = BASE_PATH / "pygame/laura/speaking/interested/interested01.png"
WINDOW_SIZE = 512

# ========== Wakeword Configuration ==========
WAKEWORD_DIR = BASE_PATH / "wakewords"
WAKEWORD_RESOURCE = BASE_PATH / "snowboy/resources/common.res"
WAKE_WORDS = {
    "GD_Laura.pmdl": 0.5,
    "Wake_up_Laura.pmdl": 0.5,
    "Laura.pmdl": 0.45,
}

# ========== Audio Settings ==========
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK = 2048

# ========== TTS Settings ==========
TTS_MODE = "local"   # "local", "api", "text"
API_TTS_PROVIDER = "elevenlabs"  # "cartesia", "elevenlabs", etc. Only used if TTS_MODE == "api"
# Local TTS (Piper)
PIPER_MODEL = "/path/to/piper/model.onnx"
PIPER_VOICE = "en_US-amy-low"
# ElevenLabs
ELEVENLABS_VOICE = "L.A.U.R.A."
ELEVENLABS_MODEL = "eleven_flash_v2_5"
# Cartesia
CARTESIA_VOICE_ID = "78f71eb3-187f-48b4-a763-952f2f4f838a"
CARTESIA_MODEL = "sonic-2"
# The actual voice/model used for API TTS is determined by the server response!

# ========== STT Settings ==========
STT_MODE = "local"           # "local" or "remote"
TRANSCRIPTION_ENGINE = "vosk"
VOSK_MODEL_PATH = BASE_PATH / "models/vosk-model-small-en-us-0.15"
WHISPER_MODEL_PATH = BASE_PATH / "models/ggml-tiny.bin"
TRANSCRIPTION_SERVER = {
    "host": "192.168.1.1",
    "port": 8765
}

# ========== Device/Server ==========
DEVICE_ID = "Pi500-og"
MCP_SERVER_URI = "ws://192.168.0.50:8765"

# ========== Notifications & Misc ==========
CALENDAR_NOTIFICATION_INTERVALS = [15, 10, 5, 2]
DEBUG_CALENDAR = False

# ========== Config Update Helpers ==========
def update_tts_mode(new_mode):
    global TTS_MODE
    if new_mode in ["local", "api", "text"]:
        TTS_MODE = new_mode

def update_api_tts_provider(new_provider):
    global API_TTS_PROVIDER
    if new_provider in ["cartesia", "elevenlabs"]:
        API_TTS_PROVIDER = new_provider

def update_stt_mode(new_mode):
    global STT_MODE
    if new_mode in ["local", "remote"]:
        STT_MODE = new_mode
