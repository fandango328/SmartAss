import os
import time
import threading
import pyaudio
import pvcobra
from mutagen.mp3 import MP3


class AudioManager:
    def __init__(self, pv_access_key):
        self.pv_access_key = pv_access_key
        self.pa = pyaudio.PyAudio()
        self.cobra = None
        self.audio_stream = None
        self.is_speaking = False
        self.is_listening = False
        self.audio_complete = threading.Event()
        self.wake_sentence_duration = 0  # Initialize wake sentence duration

    def initialize_input(self):
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
        except Exception as e:
            print(f"Error initializing input devices: {e}")
            self.cleanup()
            raise

    def start_listening(self):
        self.initialize_input()
        self.is_listening = True
        return self.audio_stream, self.cobra

    def stop_listening(self):
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

    def play_audio(self, audio_file):
        self.is_speaking = True
        self.audio_complete.clear()

        def monitor_playback():
            try:
                audio = MP3(audio_file)
                duration = audio.info.length
                self.wake_sentence_duration = duration  # Set wake sentence duration
                # Increase the buffer time from 0.3 to something larger
                wait_time = max(0, duration + 0.5)  # Added extra buffer time
                time.sleep(wait_time)
            except Exception as e:
                print(f"Error in monitor_playback: {e}")
            finally:
                self.is_speaking = False
                self.audio_complete.set()

        playback_thread = threading.Thread(target=monitor_playback)
        playback_thread.start()
    
        os.system(f'/usr/bin/mpg123 -q -a plughw:2,0 {audio_file} >/dev/null 2>&1')
        playback_thread.join()

    def wait_for_audio_completion(self):
        return self.audio_complete.wait()

    def reset_audio_state(self):
        """Reset audio state between interactions"""
        self.stop_listening()
        self.is_speaking = False
        self.is_listening = False
        self.audio_complete.clear()

    def cleanup(self):
        self.stop_listening()
        try:
            self.pa.terminate()
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")

    def re_activate_microphone(self):
        """Re-activate the microphone after the wake sentence is completed"""
        if self.wake_sentence_duration > 0:
            reactivation_delay = self.wake_sentence_duration + 0.5  # Add buffer time
            time.sleep(reactivation_delay)
        self.start_listening()
