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
    "host": "0.0.0.0",  # Listen on all interfaces
    "port": 7851,
    "alltalk_path": str(BASE_DIR),
    "max_queue_size": 10,
    "request_timeout": 30,
    "default_voice": "default",
    "default_model": "fstt-v2",
    "save_transcripts": True
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

def process_tts_request(tts_request):
    """Process TTS request using AllTalk"""
    try:
        # Save transcript first
        tts_request.save_transcript()
        
        # Here you would integrate with your AllTalk TTS system
        # This is a placeholder for the actual TTS processing
        # Replace with actual AllTalk API calls
        
        logger.info(f"Processing TTS request {tts_request.id}")
        # Example AllTalk integration (adjust based on actual API)
        response = requests.post(
            "http://localhost:7851/api/tts",
            json={
                "text": tts_request.text,
                "voice": tts_request.voice,
                "model": tts_request.model
            }
        )
        
        if response.status_code == 200:
            return response.content
        else:
            logger.error(f"TTS processing failed: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Error processing TTS request: {str(e)}")
        return None

@app.route('/api/tts', methods=['POST'])
def tts_endpoint():
    """API endpoint for TTS requests"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing text parameter"}), 400

        tts_request = TTSRequest(
            text=data['text'],
            voice=data.get('voice'),
            model=data.get('model')
        )

        logger.info(f"Received TTS request {tts_request.id}")
        
        # Process request
        audio_data = process_tts_request(tts_request)
        
        if audio_data:
            return audio_data, 200, {'Content-Type': 'audio/mpeg'}
        else:
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