#!/usr/bin/env python3

import os
import wave
import struct
import numpy as np
import time
import tempfile
import subprocess
import traceback
from pathlib import Path

class WhisperCppTranscriber:
    def __init__(self, model_path, config=None):
        """Initialize the transcriber with model path and optional config"""
        self.model_path = model_path
    
        # Apply configuration with defaults - use the values from your measurements
        self.config = {
            "energy_threshold": 0.064310,
            "silence_duration": 3.0,
            "speech_buffer_time": 1.0,
            "max_recording_time": 60
        }
    
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Extract config values for easy access
        self.energy_threshold = self.config["energy_threshold"]
    
        # Set up audio parameters
        self.sample_rate = 16000  
        self.chunk_size = 2048  # CRITICAL FIX: Match the frame_length in AudioManager (2048)
        self.audio_buffer = []
        self.is_speaking = False
        self.speech_detected = False
        self.silence_frames = 0
        self.frame_history = []
        self.start_time = None
    
        # Calculate silence frame threshold
        # This is crucial: Defines how many frames of silence trigger end
        self.silence_frame_threshold = int(self.config["silence_duration"] * self.sample_rate / self.chunk_size)
    
        print(f"DEBUG: VAD settings - energy_threshold={self.energy_threshold:.6f}, "
              f"silence_duration={self.config['silence_duration']}s, "
              f"silence_frames={self.silence_frame_threshold}")
        
        # Initialize energy-based VAD
        self.energy_threshold = self.config["energy_threshold"]
        self.frame_history = []
        self.is_speaking = False
        self.speech_detected = False
        self.silence_frames = 0
        self.sample_rate = 16000
        self.frame_duration = 0.03  # 30ms frames
        self.frames_per_second = int(1 / self.frame_duration)
        self.silence_frame_threshold = int(self.config["silence_duration"] * self.sample_rate / self.chunk_size)
        
        # Get the absolute path to the model
        if not os.path.isabs(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.join(script_dir, model_path)
        else:
            self.model_path = model_path
            
        # Check that the model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
            
        # Try to import whisper_cpp_python to check if it's available
        try:
            import whisper_cpp_python
            self.whisper_module = whisper_cpp_python
            print("Found whisper_cpp_python module")
        except ImportError:
            self.whisper_module = None
            print("whisper_cpp_python module not found, will use subprocess approach")

        # Buffer for collecting audio
        self.audio_buffer = []
        self.start_time = None

        print("Whisper.cpp transcriber initialized successfully!")

        def update_vad_settings(self, new_settings):
            """Update VAD settings while the transcriber is running"""
            self.vad_settings = new_settings
            # Apply any other necessary updates to internal state
            print("Transcriber VAD settings updated")

        def process_frame(self, frame_data):
            """
            Process audio frame and detect speech

            Args:
                frame_data: Raw audio frame data (bytes or list/array of int16)
    
            Returns:
                Tuple of (speech_ended, is_speech)
            """
            # Record start time when processing begins
            if not self.start_time and self.is_speaking:
                self.start_time = time.time()
    
            # Convert to PCM data if needed
            if isinstance(frame_data, bytes):
                # Convert bytes to int16 values
                pcm_data = struct.unpack_from("h" * (len(frame_data) // 2), frame_data)
            else:
                pcm_data = frame_data
    
            # Convert to float32 for energy calculation
            float_data = np.array(pcm_data, dtype=np.float32) / 32768.0
    
        # Calculate energy (RMS)
        energy = np.sqrt(np.mean(float_data**2)) if len(float_data) > 0 else 0
    
        # Add to frame history for smoother detection
        self.frame_history.append(energy)
    
        # Keep limited history
        if len(self.frame_history) > 10:
            self.frame_history.pop(0)
        
        # Calculate average energy
        avg_energy = sum(self.frame_history) / len(self.frame_history)
    
        # Check for max recording time
        elapsed_time = 0
        if self.start_time:
            elapsed_time = time.time() - self.start_time
        
        max_time_exceeded = elapsed_time > self.config["max_recording_time"]
    
        # COMPLETELY REWRITTEN VAD LOGIC
    
        # Thresholds tuned to your measured values
        # Using hysteresis - different thresholds for start vs continue
        start_speech_threshold = self.energy_threshold
        start_speech_threshold = self.energy_threshold
        continue_speech_threshold = self.config.get("continued_threshold", self.energy_threshold * 0.4)
    
        # Detect if current frame has speech
        is_speech = avg_energy > (continue_speech_threshold if self.is_speaking else start_speech_threshold)
    
        # State machine for speech detection
        if is_speech and not self.is_speaking:
            # Speech just started
            self.is_speaking = True
            self.speech_detected = True
            self.silence_frames = 0
            self.audio_buffer.extend(pcm_data)
            return False, True
        
        elif self.is_speaking:
            # Already in speech mode - always add audio
            self.audio_buffer.extend(pcm_data)
        
            # Check if this frame has speech
            if avg_energy > continue_speech_threshold:
                # Active speech - reset silence counter completely
                self.silence_frames = 0
                return False, True
            else:
                # No speech in this frame - increment silence
                self.silence_frames += 1
            
                # Check if we've had enough continuous silence to end recording
                if (self.silence_frames > self.silence_frame_threshold and 
                    len(self.audio_buffer) > self.sample_rate * 1.5) or max_time_exceeded:
                
                    # Speech has ended - clean up and return
                    self.is_speaking = False
                
                    # Add buffer padding at the end
                    buffer_frames = int(self.config["speech_buffer_time"] * self.sample_rate / len(pcm_data))
                    for _ in range(buffer_frames):
                        if len(self.audio_buffer) < self.sample_rate * 45:  # Limit to 45 seconds
                            self.audio_buffer.extend([0] * len(pcm_data))
                
                    return True, False
            
                # Still accumulating silence but not enough to stop yet
                return False, False
    
        # Not speaking and no speech detected
        return False, False
        
    def transcribe(self):
        """
        Transcribe collected audio using the whisper.cpp executable
        with automatic chunking for longer recordings
        """
        if not self.audio_buffer or not self.speech_detected:
            self.reset()
            return ""
        
        try:
            # Convert to float array
            audio = np.array(self.audio_buffer, dtype=np.float32) / 32768.0
        
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Calculate audio length
            audio_length = len(self.audio_buffer) / self.sample_rate
            print(f"DEBUG: Transcribing {audio_length:.2f} seconds of audio")
            
            # CHUNKING LOGIC - crucial for longer audio
            # For short audio (under 15 seconds), process normally
            if audio_length < 15:
                return self._transcribe_chunk(audio)
                
            # For longer audio, break into overlapping chunks
            chunks = []
            chunk_size = 10 * self.sample_rate  # 10 second chunks
            overlap = 1 * self.sample_rate      # 1 second overlap
            
            print(f"Audio too long ({audio_length:.2f}s), breaking into chunks")
            
            # Process each chunk
            results = []
            for start in range(0, len(audio), chunk_size - overlap):
                end = min(start + chunk_size, len(audio))
                chunk = audio[start:end]
                
                # Only process chunks with meaningful length
                if len(chunk) > 2 * self.sample_rate:  # At least 2 seconds
                    chunk_text = self._transcribe_chunk(chunk)
                    if chunk_text and chunk_text != "I didn't catch that. Could you repeat?":
                        results.append(chunk_text)
                        print(f"Chunk {len(results)}: {chunk_text}")
            
            # Combine results
            if results:
                combined_text = " ".join(results)
                print(f"Combined transcription: {combined_text}")
                return combined_text
            else:
                return "I didn't catch that. Could you repeat?"
                
        except Exception as e:
            print(f"Error in transcription: {e}")
            traceback.print_exc()
            return ""
        finally:
            self.reset()
            import gc
            gc.collect()
        
    def _transcribe_chunk(self, audio_chunk):
        """Helper method to transcribe a single chunk of audio"""
        try:
            # Save to WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
        
            with wave.open(temp_filename, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                audio_int16 = (audio_chunk * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
        
            # Use the whisper-cli executable
            whisper_cmd = "/home/zach/whisper.cpp/build/bin/whisper-cli"
        
            # Optimized command for Raspberry Pi 4
            cmd = [
                whisper_cmd,
                "-m", self.model_path,
                "-f", temp_filename,
                "-l", "en",
                "--print-special", "false",
                "-t", "3",           # Use 3 threads
                "-p", "2",           # Use 2 processors
                "--max-len", "60",   # Allow longer output
                "-sth", "0.8",       # Higher silence threshold (faster)
                "--no-context"       # Don't use previous context
            ]
        
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15
            )
        
            if result.returncode == 0:
                text = result.stdout.strip()
            
                # Clean up whisper output markers
                import re
                text = re.sub(r'\[.*?\]', '', text)  # Remove [tags]
                text = re.sub(r'\[_.*?_\]', '', text)  # Remove [_BEG_], etc.
                text = re.sub(r'\[BLANK_AUDIO\]', '', text)  # Remove blank audio markers
                text = re.sub(r'^\s*\d+:\d+:\d+\.\d+\s*-->\s*\d+:\d+:\d+\.\d+\s*', '', text)  # Remove timestamps
            
                return text.strip()
            else:
                print(f"Error running whisper: {result.stderr}")
                return ""
            
        except subprocess.TimeoutExpired:
            print("Chunk transcription timed out")
            return ""
        except Exception as e:
            print(f"Error in chunk transcription: {e}")
            return ""
        finally:
            try:
                os.unlink(temp_filename)
            except:
                pass
            
    def reset(self):
        """Reset the transcriber for a new recording session"""
        self.audio_buffer = []
        self.is_speaking = False
        self.speech_detected = False
        self.silence_frames = 0
        self.frame_history = []
        self.start_time = None
        
    def cleanup(self):
        """Clean up resources"""
        self.reset()