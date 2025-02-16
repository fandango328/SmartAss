import os
import time
import threading
import pyaudio
import pvcobra
from mutagen.mp3 import MP3

def list_microphones():
    p = pyaudio.PyAudio()
    print("\nAvailable Microphones:")
    print("---------------------")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels') > 0:  # Only show input devices
            print(f"Index: {i}")
            print(f"Name: {dev.get('name')}")
            print(f"Input Channels: {dev.get('maxInputChannels')}")
            print(f"Default Sample Rate: {dev.get('defaultSampleRate')}")
            print("---------------------")
    p.terminate()

class AudioManager:
    def __init__(self, pv_access_key):
        self.pv_access_key = pv_access_key
        self.pa = pyaudio.PyAudio()
        self.cobra = None
        self.audio_streams = {}  # Dictionary to store multiple audio streams
        self.is_speaking = False
        self.is_listening = False
        self.audio_complete = threading.Event()
        self.current_audio_duration = 0
        
        # List available input devices
        self.input_devices = self.get_input_devices()

    def get_input_devices(self):
        """List and return all available input devices"""
        input_devices = {}
        for i in range(self.pa.get_device_count()):
            device_info = self.pa.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:  # Only include input devices
                input_devices[i] = device_info
                print(f"Input Device {i}: {device_info['name']}")
        return input_devices

    def initialize_input(self, device_indices=None):
        """Initialize one or multiple input devices"""
        try:
            if self.cobra is None:
                self.cobra = pvcobra.create(access_key=self.pv_access_key)
            
            # If no specific devices are specified, use default device
            if device_indices is None:
                device_indices = [self.pa.get_default_input_device_info()['index']]
            elif not isinstance(device_indices, list):
                device_indices = [device_indices]

            for device_index in device_indices:
                if device_index not in self.audio_streams:
                    try:
                        stream = self.pa.open(
                            rate=self.cobra.sample_rate,
                            channels=1,
                            format=pyaudio.paInt16,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=self.cobra.frame_length
                        )
                        self.audio_streams[device_index] = stream
                        print(f"Initialized input device {device_index}")
                    except Exception as e:
                        print(f"Error initializing device {device_index}: {e}")

        except Exception as e:
            print(f"Error initializing input devices: {e}")
            self.cleanup()
            raise

    def start_listening(self, device_indices=None):
        """Start listening on specified devices"""
        self.initialize_input(device_indices)
        self.is_listening = True
        return self.audio_streams, self.cobra

    def stop_listening(self, device_indices=None):
        """Stop listening on specified devices"""
        self.is_listening = False
        
        # If no specific devices are specified, stop all streams
        if device_indices is None:
            device_indices = list(self.audio_streams.keys())
        elif not isinstance(device_indices, list):
            device_indices = [device_indices]

        for device_index in device_indices:
            if device_index in self.audio_streams:
                try:
                    self.audio_streams[device_index].stop_stream()
                    self.audio_streams[device_index].close()
                    del self.audio_streams[device_index]
                except Exception as e:
                    print(f"Error closing audio stream for device {device_index}: {e}")

        if not self.audio_streams and self.cobra:
            try:
                self.cobra.delete()
                self.cobra = None
            except Exception as e:
                print(f"Error deleting cobra: {e}")

    def play_audio(self, audio_file):
        self.is_speaking = True
        self.audio_complete.clear()
    
        def monitor_playback():
            try:
                audio = MP3(audio_file)
                self.current_audio_duration = audio.info.length
                duration = audio.info.length
                wait_time = max(0, duration + 0.5)
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

    def get_current_audio_duration(self):
        """Get the duration of the last played audio"""
        return self.current_audio_duration

    def wait_for_audio_completion(self):
        return self.audio_complete.wait()

    def read_audio(self, device_index):
        """Read audio from a specific device"""
        if device_index in self.audio_streams:
            return self.audio_streams[device_index].read(self.cobra.frame_length)
        return None

    def read_all_audio(self):
        """Read audio from all active devices"""
        audio_data = {}
        for device_index in self.audio_streams:
            audio_data[device_index] = self.read_audio(device_index)
        return audio_data
    
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