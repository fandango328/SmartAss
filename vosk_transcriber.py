import os
import json
import queue
import numpy as np
from vosk import Model, KaldiRecognizer

class VoskTranscriber:
    def __init__(self, model_path, sample_rate=16000):
        """Initialize Vosk transcriber with the specified model"""
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found at: {model_path}")
            
        # Load the model
        print(f"Loading Vosk model from {model_path}")
        self.model = Model(model_path)
        self.sample_rate = sample_rate
        
        # Create recognizer
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.recognizer.SetWords(True)  # Enable word timings
        
        # Initialize state
        self.reset()
        print("Vosk transcriber initialized successfully")
        
    def reset(self):
        """Reset the transcriber state"""
        self.complete_text = ""
        self.partial_text = ""
        self.buffer = []
        self.is_final = False
        self.silence_frames = 0
        
        # Reset recognizer for a new session
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)
        
    def process_frame(self, frame_data):
        """
        Process a single audio frame and update transcription
        
        Args:
            frame_data: Audio frame as bytes or numpy array
            
        Returns:
            Tuple of (is_end_of_speech, is_speech_detected, partial_text)
        """
        # Convert numpy array to bytes if needed
        if isinstance(frame_data, np.ndarray):
            frame_bytes = frame_data.tobytes()
        elif isinstance(frame_data, list):
            frame_bytes = np.array(frame_data, dtype=np.int16).tobytes()
        else:
            frame_bytes = frame_data
            
        # Process the frame
        if self.recognizer.AcceptWaveform(frame_bytes):
            # We have a final result for a chunk of speech
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "").strip()
            
            if text:
                self.complete_text += " " + text
                self.complete_text = self.complete_text.strip()
                return False, True, self.complete_text
            return False, False, self.complete_text
        else:
            # We have a partial result
            partial = json.loads(self.recognizer.PartialResult())
            self.partial_text = partial.get("partial", "").strip()
            
            # Detect if speech is happening based on partial text changing
            is_speech = bool(self.partial_text)
            
            return False, is_speech, self.partial_text
            
    def get_partial_text(self):
        """Get the current partial transcription"""
        return self.partial_text
        
    def get_final_text(self):
        """Get the final complete transcription"""
        if not self.complete_text and self.partial_text:
            # If no final text but we have partial, use that
            return self.partial_text
        return self.complete_text
        
    def transcribe(self):
        """
        Get final transcription after processing is complete
        
        Returns:
            Transcribed text
        """
        # Finalize any remaining audio
        final_result = json.loads(self.recognizer.FinalResult())
        final_text = final_result.get("text", "").strip()
        
        if final_text:
            self.complete_text += " " + final_text
            self.complete_text = self.complete_text.strip()
            
        return self.complete_text
        
    def cleanup(self):
        """Clean up resources"""
        self.reset()