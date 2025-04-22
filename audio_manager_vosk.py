#!/usr/bin/env python3

import os
import time
import asyncio
import pyaudio
from mutagen.mp3 import MP3
from asyncio import Event
import numpy as np

class AudioManager:
    def __init__(self, pv_access_key=None):  # Keep param for compatibility, but don't use it
        self.pa = pyaudio.PyAudio()
        self.audio_stream = None
        self.is_speaking = False  # For tracking speech generation
        self.is_playing = False   # For tracking audio playback
        self.is_listening = False
        self.current_process = None  # Store current audio playback process
        self.audio_complete = Event()
        self.audio_complete.set()  # Initially set to True since no audio is playing
        self.audio_state_changed = asyncio.Queue()
        self.activation_lock = asyncio.Lock()
        self.wake_sentence_duration = 0
        self.playback_lock = asyncio.Lock()  # Add lock for playback management
        
        # Add a new event for notification signals
        self.notification_ready = Event()
        self.notification_ready.clear()  # Initially no notifications
        
        # Add notification text and path storage
        self.notification_text = ""
        self.notification_file = "notification.mp3"
        
        # Frame properties - CRITICAL: Use 2048 for better performance
        self.sample_rate = 16000
        self.frame_length = 2048  # Optimal chunk size for processing
        
        # Initialize audio queue system
        self.__init_audio_queue()
        print(f"AudioManager initialized: frame_length={self.frame_length}, sample_rate={self.sample_rate}")

    async def initialize_input(self):
        """Initialize audio input device"""
        async with self.activation_lock:
            try:
                if self.audio_stream is None:
                    #print("DEBUG: Opening audio stream")
                    self.audio_stream = self.pa.open(
                        rate=self.sample_rate,
                        channels=1,
                        format=pyaudio.paInt16,
                        input=True,
                        frames_per_buffer=self.frame_length
                    )
                    #print("DEBUG: Audio stream opened")
                await self.audio_state_changed.put(('input_initialized', True))
            except Exception as e:
                await self.audio_state_changed.put(('error', str(e)))
                print(f"Error initializing input devices: {e}")
                await self.cleanup()
                raise

    async def start_listening(self):
        """Start listening for audio input"""
        if self.is_speaking:
            await self.audio_complete.wait()  # Wait for any audio playback to complete
    
        await self.initialize_input()
        self.is_listening = True
    
        # Return just the audio stream now since we're not using Cobra
        return self.audio_stream, None

    async def stop_listening(self):
        """Stop listening for audio input"""
        # print("DEBUG: Stopping listening")  # Remove this
        self.is_listening = False
        if self.audio_stream:
            try:
                # print("DEBUG: Closing audio stream")  # Remove this
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                # print("DEBUG: Audio stream closed")  # Remove this
            except Exception as e:
                print(f"ERROR: Failed to close audio stream: {e}")
            finally:
                self.audio_stream = None

    async def play_audio(self, audio_file, interrupt_current=False):
        """
        Play audio file using mpg123
        Args:
            audio_file: Path to audio file
            interrupt_current: If True, stop current audio before playing
        """
        async with self.playback_lock:
            if interrupt_current and self.is_speaking:
                await self.stop_current_audio()
            
            self.is_speaking = True
            self.is_playing = True
            self.audio_complete.clear()
            
            try:
                # Calculate audio duration before playback
                try:
                    audio = MP3(audio_file)
                    self.wake_sentence_duration = audio.info.length
                except Exception as e:
                    print(f"Error calculating audio duration: {e}")
                    self.wake_sentence_duration = 2.0  # Default duration if can't determine
                
                # Start the playback process - removed hardcoded audio device
                self.current_process = await asyncio.create_subprocess_shell(
                    f'/usr/bin/mpg123 -q {audio_file}',
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                
                # Wait for process to complete
                await self.current_process.wait()
                
                # Add buffer time - critical for good UX
                buffer_time = 0.7  # Increased buffer time
                await asyncio.sleep(buffer_time)
                
            except Exception as e:
                print(f"Error in play_audio: {e}")
            finally:
                self.current_process = None
                self.is_speaking = False
                self.is_playing = False
                self.audio_complete.set()
                await self.audio_state_changed.put(('audio_completed', True))

    async def stop_current_audio(self):
        """Stop currently playing audio if any"""
        if self.current_process and self.is_speaking:
            try:
                self.current_process.terminate()
                await self.current_process.wait()
            except Exception as e:
                print(f"Error stopping current audio: {e}")
            finally:
                self.current_process = None
                self.is_speaking = False
                self.is_playing = False
                self.audio_complete.set()
                await self.audio_state_changed.put(('audio_completed', True))

    def set_notification(self, text):
        """Set notification text and trigger event"""
        self.notification_text = text
        self.notification_ready.set()
    
    def clear_notification(self):
        """Clear notification"""
        self.notification_text = ""
        self.notification_ready.clear()
    
    async def has_notification(self, timeout=0.1):
        """Check if notification is available with timeout"""
        try:
            # Check with a short timeout to avoid blocking
            await asyncio.wait_for(self.notification_ready.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def wait_for_audio_completion(self):
        """Wait for any audio playback to complete"""
        await self.audio_complete.wait()

    async def reset_audio_state(self):
        """Reset audio state between interactions"""
        await self.stop_listening()
        self.is_speaking = False
        self.is_playing = False
        self.is_listening = False
        self.audio_complete.set()

    async def cleanup(self):
        """Clean up resources"""
        await self.stop_listening()
        try:
            self.pa.terminate()
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")

    async def monitor_audio_state(self):
        """Background task to monitor audio state changes"""
        while True:
            try:
                state, value = await self.audio_state_changed.get()
                if state == 'audio_completed':
                    self.is_speaking = False
                    self.is_playing = False
                elif state == 'input_initialized':
                    self.is_listening = True
            except Exception as e:
                print(f"Error in audio state monitor: {e}")

    def read_audio_frame(self, frame_size=None):
        """Read a single audio frame"""
        if not self.audio_stream:
            return None
            
        if not frame_size:
            frame_size = self.frame_length
            
        try:
            return self.audio_stream.read(frame_size, exception_on_overflow=False)
        except Exception as e:
            print(f"Error reading audio frame: {e}")
            return None

    def __init_audio_queue(self):
        """Initialize audio queue system"""
        self.audio_queue = asyncio.Queue()
        self.queue_processor_task = None

    async def queue_audio(self, audio_file=None, generated_text=None, priority=False):
        """Add audio to play queue"""
        if not hasattr(self, 'audio_queue'):
            self.__init_audio_queue()
            
        if priority:
            # Create new priority queue
            temp_queue = asyncio.Queue()
            await temp_queue.put((audio_file, generated_text))
            # Transfer existing items
            while not self.audio_queue.empty():
                item = await self.audio_queue.get()
                await temp_queue.put(item)
            self.audio_queue = temp_queue
        else:
            await self.audio_queue.put((audio_file, generated_text))
            
        # Start queue processor if not running
        if not self.queue_processor_task or self.queue_processor_task.done():
            self.queue_processor_task = asyncio.create_task(self.process_audio_queue())

    async def process_audio_queue(self):
        """Process queued audio items"""
        while True:
            try:
                audio_file, generated_text = await self.audio_queue.get()
                
                # Use existing play_audio method
                if audio_file:
                    await self.play_audio(audio_file)
                elif generated_text:
                    # Handle generated text through existing pipeline
                    # This would connect to your existing TTS system
                    pass
                    
                self.audio_queue.task_done()
                
                # Exit if queue is empty
                if self.audio_queue.empty():
                    break
                    
            except Exception as e:
                print(f"Error processing audio queue: {e}")
                continue

    async def wait_for_queue_empty(self):
        """Wait for all queued audio to complete"""
        if hasattr(self, 'audio_queue'):
            await self.audio_queue.join()
