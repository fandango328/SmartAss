#!/usr/bin/env python3

import os
import time
import asyncio
import pyaudio
from mutagen.mp3 import MP3
from asyncio import Event
import numpy as np

class AudioManager:
    def __init__(self, pv_access_key=None):
        self.pa = pyaudio.PyAudio()
        self.audio_stream = None
        self.is_speaking = False
        self.is_playing = False
        self.is_listening = False
        self.current_process = None
        self.audio_complete = Event()
        self.audio_complete.set()
        self.audio_state_changed = asyncio.Queue()
        self.activation_lock = asyncio.Lock()
        self.wake_sentence_duration = 0
        self.playback_lock = asyncio.Lock()
        
        # Audio queue system
        self.audio_queue = asyncio.Queue()
        self.queue_processor_task = None
        self.is_processing_queue = False
        
        # Frame properties
        self.sample_rate = 16000
        self.frame_length = 2048
        
        print(f"AudioManager initialized: frame_length={self.frame_length}, sample_rate={self.sample_rate}")

    async def initialize_input(self):
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
        if self.is_speaking:
            await self.audio_complete.wait()
    
        await self.initialize_input()
        self.is_listening = True
        return self.audio_stream, None

    async def stop_listening(self):
        self.is_listening = False
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except Exception as e:
                print(f"ERROR: Failed to close audio stream: {e}")
            finally:
                self.audio_stream = None

    async def queue_audio(self, audio_file=None, generated_text=None, priority=False):
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
        if not self.is_processing_queue:
            self.queue_processor_task = asyncio.create_task(self.process_audio_queue())

    async def process_audio_queue(self):
        self.is_processing_queue = True
        try:
            while True:
                try:
                    audio_file, generated_text = await self.audio_queue.get()
                    
                    if audio_file:
                        await self.play_audio(audio_file)
                    elif generated_text:
                        # Handle TTS generation if needed
                        pass
                        
                    self.audio_queue.task_done()
                    
                    # Exit if queue is empty
                    if self.audio_queue.empty():
                        break
                        
                except Exception as e:
                    print(f"Error processing audio queue item: {e}")
                    continue
                    
        finally:
            self.is_processing_queue = False

    async def play_audio(self, audio_file, interrupt_current=False):
        async with self.playback_lock:
            if interrupt_current and self.is_speaking:
                await self.stop_current_audio()
            
            self.is_speaking = True
            self.is_playing = True
            self.audio_complete.clear()
            
            try:
                try:
                    audio = MP3(audio_file)
                    self.wake_sentence_duration = audio.info.length
                except Exception as e:
                    print(f"Error calculating audio duration: {e}")
                    self.wake_sentence_duration = 2.0
                
                self.current_process = await asyncio.create_subprocess_shell(
                    f'/usr/bin/mpg123 -q {audio_file}',
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                
                await self.current_process.wait()
                await asyncio.sleep(0.7)
                
            except Exception as e:
                print(f"Error in play_audio: {e}")
            finally:
                self.current_process = None
                self.is_speaking = False
                self.is_playing = False
                self.audio_complete.set()
                await self.audio_state_changed.put(('audio_completed', True))

    async def stop_current_audio(self):
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

    async def wait_for_audio_completion(self):
        await self.audio_complete.wait()

    async def wait_for_queue_empty(self):
        if hasattr(self, 'audio_queue'):
            await self.audio_queue.join()

    async def reset_audio_state(self):
        await self.stop_listening()
        self.is_speaking = False
        self.is_playing = False
        self.is_listening = False
        self.audio_complete.set()

    async def cleanup(self):
        await self.stop_listening()
        try:
            self.pa.terminate()
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")

    def read_audio_frame(self, frame_size=None):
        if not self.audio_stream:
            return None
            
        if not frame_size:
            frame_size = self.frame_length
            
        try:
            return self.audio_stream.read(frame_size, exception_on_overflow=False)
        except Exception as e:
            print(f"Error reading audio frame: {e}")
            return None
