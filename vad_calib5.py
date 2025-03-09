#!/usr/bin/env python3
import pyaudio
import numpy as np
import time
import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vad_settings import save_vad_settings

# Match AudioManager settings exactly
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 2048

def measure_levels(stream, duration, description=""):
    levels = []
    peak_levels = []  # Track peaks specifically
    frame_history = []
    
    start_time = time.time()
    while time.time() - start_time < duration:
        data = stream.read(CHUNK, exception_on_overflow=False)
        float_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        energy = np.sqrt(np.mean(float_data**2))
        
        # Track both instantaneous and smoothed values
        frame_history.append(energy)
        if len(frame_history) > 5:  # Use 5-frame history
            frame_history.pop(0)
        
        # Calculate smoothed energy (max of last 2 frames)
        smoothed_energy = max(frame_history[-2:]) if len(frame_history) >= 2 else energy
        
        levels.append(smoothed_energy)
        if energy > 0.03:  # Only track significant peaks
            peak_levels.append(energy)
            
        print(f"{description} Energy: {smoothed_energy:.6f} (Peak: {energy:.6f})")
        
    return levels, peak_levels

def main():
    print("=== LAURA VAD CALIBRATION ===")
    print("This calibration uses improved peak detection\n")
    
    p = pyaudio.PyAudio()
    stream = p.open(
        rate=RATE,
        channels=CHANNELS,
        format=FORMAT,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    try:
        # Measure background noise
        print("Please remain silent for 5 seconds...")
        time.sleep(1)
        silence_levels, silence_peaks = measure_levels(stream, 5, "Silence")
        
        # Measure speech
        print("\nPlease speak normally for 5 seconds...")
        time.sleep(1)
        speech_levels, speech_peaks = measure_levels(stream, 5, "Speech")
        
        # Calculate thresholds
        silence_mean = np.mean(silence_levels)
        silence_max = np.percentile(silence_levels, 95)  # 95th percentile for stable max
        
        speech_mean = np.mean(speech_peaks) if speech_peaks else np.mean(speech_levels)
        speech_min = np.percentile(speech_levels, 25)  # 25th percentile for stable min
        
        # Set thresholds based on actual measurements
        energy_threshold = max(silence_max * 1.5, 0.040)  # At least 50% above noise
        continued_ratio = 0.7  # More permissive ratio
        
        # Create new profile
        new_profile = {
            "energy_threshold": round(energy_threshold, 6),
            "continued_threshold_ratio": round(continued_ratio, 2),
            "continued_threshold": round(energy_threshold * continued_ratio, 6),
            "silence_duration": 3.0,
            "speech_buffer_time": 1.0,
            "min_speech_duration": 0.4,
            "max_recording_time": 30,
            "chunk_size": CHUNK
        }
        
        # Save settings
        save_vad_settings(new_profile, "current")
        
        print("\n=== CALIBRATION RESULTS ===")
        print(f"Background Noise:")
        print(f"  Mean: {silence_mean:.6f}")
        print(f"  Max: {silence_max:.6f}")
        print(f"\nSpeech Levels:")
        print(f"  Mean: {speech_mean:.6f}")
        print(f"  Min: {speech_min:.6f}")
        print(f"  Peak Max: {max(speech_peaks):.6f}" if speech_peaks else "  No peaks detected")
        
        print(f"\nCalibrated Settings:")
        print(f"  Energy Threshold: {new_profile['energy_threshold']:.6f}")
        print(f"  Continued Ratio: {new_profile['continued_threshold_ratio']:.2f}")
        print(f"  Continued Threshold: {new_profile['continued_threshold']:.6f}")
        
        print("\nCalibration complete! Settings saved and active.")
        print("CALIBRATION_COMPLETE")
        
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
