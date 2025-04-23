#!/usr/bin/env python3

import os
import time
import asyncio
import pyaudio
from mutagen.mp3 import MP3
from asyncio import Event
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class AudioManagerState:
    """
    Centralized state tracking for audio system.
    
    Attributes:
        is_playing (bool): Indicates if any audio is currently playing
        is_speaking (bool): Indicates if TTS audio is currently playing
        is_listening (bool): Indicates if audio input is being captured
        playback_start_time (float): Timestamp when current audio started
        current_audio_file (str): Path to currently playing audio file
        expected_duration (float): Expected duration of current audio in seconds
    """
    is_playing: bool = False
    is_speaking: bool = False
    is_listening: bool = False
    playback_start_time: Optional[float] = None
    current_audio_file: Optional[str] = None
    expected_duration: Optional[float] = None

class AudioManager:
    """
    Manages audio input/output operations with proper state tracking and resource management.
    
    This class handles:
    - Audio input capture for speech recognition
    - Audio playback for TTS and notification sounds
    - Queue management for sequential audio playback
    - State tracking for coordination with other system components
    """
    
    def __init__(self, pv_access_key=None):
        """
        Initialize AudioManager with required resources and state tracking.
        
        Args:
            pv_access_key: Optional key for Picovoice integration (unused in current version)
        """
        # Initialize PyAudio for audio I/O
        self.pa = pyaudio.PyAudio()
        self.audio_stream = None
        self.current_process = None
        
        # Event and state management
        self.audio_complete = Event()
        self.audio_complete.set()
        self.audio_state_changed = asyncio.Queue()
        
        # Locks for thread safety
        self.activation_lock = asyncio.Lock()
        self.playback_lock = asyncio.Lock()
        self.state_lock = asyncio.Lock()
        
        # Centralized state tracking
        self.state = AudioManagerState()
        
        # Audio queue for sequential playback
        self.audio_queue = asyncio.Queue()
        self.queue_processor_task = None
        self.is_processing_queue = False
        
        # Audio frame configuration
        self.sample_rate = 16000  # Standard sample rate for speech recognition
        self.frame_length = 2048  # Buffer size for audio frames
        
        print(f"AudioManager initialized: frame_length={self.frame_length}, sample_rate={self.sample_rate}")

    async def get_state(self) -> Dict[str, Any]:
        """
        Get current audio system state in a thread-safe manner.
        
        Returns:
            Dict containing current state values for all tracked attributes
        """
        async with self.state_lock:
            return {
                "is_playing": self.state.is_playing,
                "is_speaking": self.state.is_speaking,
                "is_listening": self.state.is_listening,
                "playback_start_time": self.state.playback_start_time,
                "current_audio_file": self.state.current_audio_file,
                "expected_duration": self.state.expected_duration
            }

    @property
    def is_playing(self):
        """Property accessor for state.is_playing"""
        return self.state.is_playing
        
    @property
    def is_speaking(self):
        """Property accessor for state.is_speaking"""
        return self.state.is_speaking
    
    @property
    def is_listening(self):
        """Property accessor for state.is_listening"""
        return self.state.is_listening
    
    @property
    def playback_start_time(self):
        """Property accessor for state.playback_start_time"""
        return self.state.playback_start_time
    
    @property
    def current_audio_file(self):
        """Property accessor for state.current_audio_file"""
        return self.state.current_audio_file
    
    @property
    def expected_duration(self):
        """Property accessor for state.expected_duration"""
        return self.state.expected_duration
                
    async def initialize_input(self):
        """
        Initialize audio input stream for capturing speech.
        
        Raises:
            Exception: If audio device initialization fails
        """
        async with self.activation_lock:
            try:
                if self.audio_stream is None:
                    self.audio_stream = self.pa.open(
                        rate=self.sample_rate,
                        channels=1,
                        format=pyaudio.paInt16,
                        input=True,
                        frames_per_buffer=self.frame_length
                    )
                await self.audio_state_changed.put(('input_initialized', True))
            except Exception as e:
                await self.audio_state_changed.put(('error', str(e)))
                print(f"Error initializing input devices: {e}")
                await self.cleanup()
                raise

    async def start_listening(self):
        """
        Start audio input capture for speech recognition.
        
        Waits for any current speech playback to complete before starting input.
        
        Returns:
            Tuple of (audio_stream, None) - None is placeholder for future metadata
        """
        if self.state.is_speaking:
            await self.audio_complete.wait()
        
        await self.initialize_input()
        async with self.state_lock:
            self.state.is_listening = True
        return self.audio_stream, None

    async def stop_listening(self):
        """
        Stop audio input capture and cleanup resources.
        
        This method ensures proper cleanup of audio input resources
        even if errors occur during shutdown.
        """
        async with self.state_lock:
            self.state.is_listening = False
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except Exception as e:
                print(f"ERROR: Failed to close audio stream: {e}")
            finally:
                self.audio_stream = None

    async def queue_audio(self, audio_file: Optional[str] = None, generated_text: Optional[str] = None):
        """
        Add audio to playback queue for sequential processing.
        
        Args:
            audio_file: Path to MP3 file to play
            generated_text: Text for TTS generation (not currently implemented)
        """
        await self.audio_queue.put((audio_file, generated_text))
        
        # Start queue processor if not already running
        if not self.is_processing_queue:
            self.queue_processor_task = asyncio.create_task(self.process_audio_queue())

    async def process_audio_queue(self):
        """
        Process queued audio files sequentially.
        
        This method runs as a background task, processing audio files
        in the order they were queued until the queue is empty.
        """
        self.is_processing_queue = True
        try:
            while True:
                try:
                    audio_file, generated_text = await self.audio_queue.get()
                    
                    if audio_file:
                        await self.play_audio(audio_file)
                    elif generated_text:
                        # Future: Implement TTS generation
                        pass
                        
                    self.audio_queue.task_done()
                    
                    if self.audio_queue.empty():
                        break
                        
                except Exception as e:
                    print(f"Error processing audio queue item: {e}")
                    continue
                    
        finally:
            self.is_processing_queue = False

    async def play_audio(self, audio_file: str):
        """
        Play audio file with state tracking and resource management.
        
        Args:
            audio_file: Path to MP3 file to play
            
        This method handles:
        - State updates for playback tracking
        - Duration calculation for timing coordination
        - Process management for mpg123 playback
        - Cleanup after playback completion
        """
        async with self.playback_lock:
            async with self.state_lock:
                self.state.is_speaking = True
                self.state.is_playing = True
                self.state.playback_start_time = time.time()
                self.state.current_audio_file = audio_file
            self.audio_complete.clear()
            
            try:
                # Calculate audio duration for timing coordination
                try:
                    audio = MP3(audio_file)
                    async with self.state_lock:
                        self.state.expected_duration = audio.info.length
                except Exception as e:
                    print(f"Error calculating audio duration: {e}")
                    async with self.state_lock:
                        self.state.expected_duration = 2.0
                
                # Play audio using mpg123 (external process)
                self.current_process = await asyncio.create_subprocess_shell(
                    f'/usr/bin/mpg123 -q {audio_file}',
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                
                await self.current_process.wait()
                await asyncio.sleep(0.7)  # Buffer time for audio device release
                
            except Exception as e:
                print(f"Error in play_audio: {e}")
            finally:
                # Cleanup and state reset
                self.current_process = None
                async with self.state_lock:
                    self.state.is_speaking = False
                    self.state.is_playing = False
                    self.state.playback_start_time = None
                    self.state.current_audio_file = None
                    self.state.expected_duration = None
                self.audio_complete.set()
                await self.audio_state_changed.put(('audio_completed', True))

    async def stop_current_audio(self):
        """
        Stop currently playing audio and reset state.
        
        This method ensures proper cleanup of audio playback resources
        and state reset even if errors occur during shutdown.
        """
        if self.current_process and self.state.is_speaking:
            try:
                self.current_process.terminate()
                await self.current_process.wait()
            except Exception as e:
                print(f"Error stopping current audio: {e}")
            finally:
                self.current_process = None
                async with self.state_lock:
                    self.state.is_speaking = False
                    self.state.is_playing = False
                    self.state.playback_start_time = None
                    self.state.current_audio_file = None
                    self.state.expected_duration = None
                self.audio_complete.set()
                await self.audio_state_changed.put(('audio_completed', True))

    async def wait_for_audio_completion(self):
        """
        Wait for current audio playback to complete.
        
        This method blocks until the audio_complete event is set,
        indicating that playback has finished.
        """
        await self.audio_complete.wait()

    async def wait_for_queue_empty(self):
        """
        Wait for audio queue to be completely processed.
        
        This method blocks until all queued audio files have been
        played and the queue is empty.
        """
        if hasattr(self, 'audio_queue'):
            await self.audio_queue.join()

    async def reset_audio_state(self):
        """
        Reset all audio states to initial values.
        
        This method performs a complete reset of the audio system,
        stopping input capture and clearing all state flags.
        """
        await self.stop_listening()
        async with self.state_lock:
            self.state = AudioManagerState()
        self.audio_complete.set()

    async def cleanup(self):
        """
        Clean up all audio resources.
        
        This method ensures proper shutdown of all audio components
        and should be called before program termination.
        """
        await self.stop_listening()
        try:
            self.pa.terminate()
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")

    def read_audio_frame(self, frame_size: Optional[int] = None) -> Optional[bytes]:
        """
        Read a frame of audio data from the input stream.
        
        Args:
            frame_size: Optional custom frame size (defaults to self.frame_length)
            
        Returns:
            bytes containing audio data or None if error occurs
            
        This method is used by speech recognition systems to obtain
        raw audio data for processing.
        """
        if not self.audio_stream:
            return None
            
        if not frame_size:
            frame_size = self.frame_length
            
        try:
            return self.audio_stream.read(frame_size, exception_on_overflow=False)
        except Exception as e:
            print(f"Error reading audio frame: {e}")
            return None
