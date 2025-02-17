#!/usr/bin/env python3

import os
import sys
import logging
import json
import datetime
from pathlib import Path
import requests
from flask import Flask, request, jsonify
import threading
import queue
import time

# Base paths
BASE_DIR = Path("I:/alltalk_tts")
VENV_DIR = BASE_DIR / "venv"
LOGS_DIR = BASE_DIR / "logs"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
CONFIG_FILE = BASE_DIR / "config.json"

# Create necessary directories
for directory in [LOGS_DIR, TRANSCRIPTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "alltalk_server.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AllTalk_Server")

# Default configuration
DEFAULT_CONFIG = {
    "host": "0.0.0.0",
    "port": 7853,  # Match your current port
    "alltalk_path": str(BASE_DIR),
    "max_queue_size": 10,
    "request_timeout": 30,
    "default_voice": "voices/Laura.wav",  # Updated path
    "default_model": "xtts",  # Simplified model name
    "save_transcripts": True,
    "models_path": str(BASE_DIR / "models"),
    "voices_path": str(BASE_DIR / "voices"),
    "xtts_config": {
        "model_path": "models/xtts/model.pth",
        "config_path": "models/xtts/config.json"
    }
}

def load_config():
    """Load configuration from file or create default"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    else:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG

config = load_config()

# Initialize Flask app
app = Flask(__name__)
request_queue = queue.Queue(maxsize=config['max_queue_size'])

class TTSRequest:
    def __init__(self, text, voice=None, model=None):
        self.text = text
        self.voice = voice or config['default_voice']
        self.model = model or config['default_model']
        self.timestamp = datetime.datetime.now()
        self.id = f"{self.timestamp.strftime('%Y%m%d_%H%M%S')}_{hash(text) % 10000:04d}"

    def save_transcript(self):
        """Save request details to transcript file"""
        if not config['save_transcripts']:
            return

        transcript_file = TRANSCRIPTS_DIR / f"{self.timestamp.strftime('%Y%m%d')}.jsonl"
        entry = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "text": self.text,
            "voice": self.voice,
            "model": self.model
        }
        
        with open(transcript_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')

# In process_tts_request function, update the endpoint
def process_tts_request(tts_request):
    """Process TTS request using AllTalk"""
    try:
        tts_request.save_transcript()
        
        logger.info(f"Processing TTS request {tts_request.id}")
        
        # Prepare request data
        request_data = {
            "text": tts_request.text,
            "voice": tts_request.voice,
            "model": "xtts",  # Force XTTS model
            "language": "en",
            "stream": False  # Disable streaming for now
        }
        
        # Make request to TTS engine
        response = requests.post(
            f"http://localhost:{config['port']}/api/tts",
            json=request_data,
            timeout=config['request_timeout']
        )
        
        if response.status_code == 200:
            return response.content
        else:
            logger.error(f"TTS processing failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error processing TTS request: {str(e)}")
        return None

# New endpoint for voice file upload:
@app.route('/api/upload_voice', methods=['POST'])
def upload_voice():
    """Handle voice file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not file.filename.endswith('.wav'):
            return jsonify({"error": "File must be WAV format"}), 400
            
        # Save file to voices directory
        voice_path = Path(config['voices_path']) / file.filename
        file.save(str(voice_path))
        
        return jsonify({"message": "Voice file uploaded successfully"}), 200
        
    except Exception as e:
        logger.error(f"Error handling voice upload: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/tts', methods=['POST'])
def tts_endpoint():
    """API endpoint for TTS requests"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing text parameter"}), 400

        # Process request to AllTalk
        response = requests.post(
            "http://localhost:7851/api/tts-generate",
            data={
                "text_input": data['text'],
                "character_voice_gen": data.get('voice', config['default_voice']),
                "text_filtering": "standard",
                "narrator_enabled": "false",
                "language": "en",
                "output_file_timestamp": "true"
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            # Get the audio file from the URL in the response
            audio_url = result.get('output_file_url')
            if audio_url:
                audio_response = requests.get(audio_url)
                return audio_response.content, 200, {'Content-Type': 'audio/mpeg'}
        
        return jsonify({"error": "TTS processing failed"}), 500

    except Exception as e:
        logger.error(f"Error handling request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "queue_size": request_queue.qsize()
    })

@app.route('/api/ready', methods=['GET'])
def ready_check():
    """Check if AllTalk is ready through this server"""
    try:
        # Check if AllTalk is actually ready
        response = requests.get("http://localhost:7851/api/ready")
        if response.text == "Ready":
            return "Ready"
        else:
            return "Not Ready", 503
    except Exception as e:
        logger.error(f"Error checking AllTalk readiness: {str(e)}")
        return "Not Ready", 503

def setup_venv():
    """Setup virtual environment and install requirements"""
    if not VENV_DIR.exists():
        logger.info("Setting up virtual environment...")
        os.system(f"python -m venv {VENV_DIR}")
        
        # Create requirements.txt if it doesn't exist
        requirements_file = BASE_DIR / "requirements.txt"
        if not requirements_file.exists():
            with open(requirements_file, 'w') as f:
                f.write("""
flask==2.0.1
requests==2.26.0
python-dotenv==0.19.0
# Add other required packages
                """.strip())
        
        # Install requirements
        if sys.platform == "win32":
            os.system(f"{VENV_DIR}/Scripts/pip install -r {requirements_file}")
        else:
            os.system(f"{VENV_DIR}/bin/pip install -r {requirements_file}")

def main():
    """Main function to run the server"""
    try:
        # Setup virtual environment if needed
        setup_venv()
        
        logger.info("Starting AllTalk TTS Server...")
        logger.info(f"Configuration loaded from {CONFIG_FILE}")
        
        # Run Flask app
        app.run(
            host=config['host'],
            port=config['port'],
            debug=False  # Set to True for development
        )
    
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()