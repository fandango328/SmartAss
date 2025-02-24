import os
import time
import asyncio
import pyaudio
import pvcobra
from mutagen.mp3 import MP3
from asyncio import Event

class AudioManager:
    def __init__(self, pv_access_key):
        self.pv_access_key = pv_access_key
        self.pa = pyaudio.PyAudio()
        self.cobra = None
        self.audio_stream = None
        self.is_speaking = False
        self.is_listening = False
        self.audio_complete = Event()
        self.audio_complete.set()  # Initially set to True since no audio is playing
        self.audio_state_changed = asyncio.Queue()
        self.activation_lock = asyncio.Lock()
        
        # Add a new event for notification signals
        self.notification_ready = Event()
        self.notification_ready.clear()  # Initially no notifications
        
        # Add notification text and path storage
        self.notification_text = ""
        self.notification_file = "notification.mp3"

    async def initialize_input(self):
        async with self.activation_lock:
            try:
                if self.cobra is None:
                    self.cobra = pvcobra.create(access_key=self.pv_access_key)
                if self.audio_stream is None:
                    self.audio_stream = self.pa.open(
                        rate=self.cobra.sample_rate,
                        channels=1,
                        format=pyaudio.paInt16,
                        input=True,
                        frames_per_buffer=self.cobra.frame_length
                    )
                await self.audio_state_changed.put(('input_initialized', True))
            except Exception as e:
                await self.audio_state_changed.put(('error', str(e)))
                print(f"Error initializing input devices: {e}")
                await self.cleanup()
                raise

    async def start_listening(self):
        if self.is_speaking:
            await self.audio_complete.wait()  # Wait for any audio playback to complete
        
        await self.initialize_input()
        self.is_listening = True
        return self.audio_stream, self.cobra

    async def stop_listening(self):
        self.is_listening = False
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except Exception as e:
                print(f"Error closing audio stream: {e}")
            finally:
                self.audio_stream = None
        
        if self.cobra:
            try:
                self.cobra.delete()
            except Exception as e:
                print(f"Error deleting cobra: {e}")
            finally:
                self.cobra = None

    async def play_audio(self, audio_file):
        self.is_speaking = True
        self.audio_complete.clear()
        
        try:
            proc = await asyncio.create_subprocess_shell(
                f'/usr/bin/mpg123 -q -a plughw:2,0 {audio_file}',
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            
            await proc.wait()
            
        except Exception as e:
            print(f"Error in play_audio: {e}")
        finally:
            self.is_speaking = False
            self.audio_complete.set()
            await self.audio_state_changed.put(('audio_completed', True))

    # Add a method to trigger a notification
    def set_notification(self, text):
        self.notification_text = text
        self.notification_ready.set()
    
    # Add a method to clear notification state
    def clear_notification(self):
        self.notification_text = ""
        self.notification_ready.clear()
    
    # Add a method to check if notification is ready
    async def has_notification(self, timeout=0.1):
        try:
            # Check with a short timeout to avoid blocking
            await asyncio.wait_for(self.notification_ready.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def wait_for_audio_completion(self):
        await self.audio_complete.wait()

    async def reset_audio_state(self):
        """Reset audio state between interactions"""
        await self.stop_listening()
        self.is_speaking = False
        self.is_listening = False
        self.audio_complete.set()

    async def cleanup(self):
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
                elif state == 'input_initialized':
                    self.is_listening = True
            except Exception as e:
                print(f"Error in audio state monitor: {e}")