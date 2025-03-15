#!/usr/bin/env python3
import pyaudio
import numpy as np
import time
import sys
import os
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vad_settings import save_vad_settings, load_vad_settings

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 2048

def convert_to_native_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    return obj

def verify_settings_file(settings_path):
    """Verify that the settings file exists and is valid JSON"""
    try:
        if os.path.exists(settings_path):
            with open(settings_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                required_keys = ['profiles']
                if not all(key in data for key in required_keys):
                    logger.error("Settings file missing required keys")
                    return False
                return True
        return False
    except json.JSONDecodeError:
        logger.error("Invalid JSON in settings file")
        return False
    except Exception as e:
        logger.error(f"Error verifying settings file: {e}")
        return False

def save_settings_directly(settings_data, settings_path):
    """Save settings directly to file with proper structure"""
    try:
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        native_settings = convert_to_native_types(settings_data)
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(native_settings, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        return True
    except Exception as e:
        logger.error(f"Error in direct save: {e}")
        return False

def measure_levels(stream, duration, description=""):
    """Measure audio levels over specified duration"""
    levels = []
    peak_levels = []
    frame_history = []
    
    start_time = time.time()
    while time.time() - start_time < duration:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            float_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            energy = np.sqrt(np.mean(float_data**2))
            
            frame_history.append(energy)
            if len(frame_history) > 10:
                frame_history.pop(0)
            
            smoothed_energy = np.median(frame_history)
            levels.append(smoothed_energy)
            if energy > 0.01:
                peak_levels.append(energy)
                
            logger.debug(f"{description} Energy: {smoothed_energy:.6f} (Peak: {energy:.6f})")
            
        except Exception as e:
            logger.error(f"Error reading audio: {e}")
            continue
            
    return levels, peak_levels

def calculate_threshold(silence_levels, speech_levels, speech_peaks):
    """Calculate appropriate threshold based on measurements"""
    silence_mean = np.mean(silence_levels)
    silence_std = np.std(silence_levels)
    silence_max = silence_mean + (2 * silence_std)
    
    speech_mean = np.mean(speech_peaks) if speech_peaks else np.mean(speech_levels)
    speech_min = np.percentile(speech_levels, 20)
    
    base_threshold = max(
        silence_max * 1.2,
        min(0.09, speech_min * 0.8)
    )
    
    energy_threshold = max(0.06, min(0.09, base_threshold))
    return float(energy_threshold)

def main():
    logger.info("=== LAURA VAD CALIBRATION ===")
    logger.info("Starting calibration with improved peak detection and threshold calculation\n")
    
    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            rate=RATE,
            channels=CHANNELS,
            format=FORMAT,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print("\nPlease remain silent for 5 seconds...")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        silence_levels, silence_peaks = measure_levels(stream, 5, "Silence")
        
        print("\nPlease speak normally for 5 seconds...")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        speech_levels, speech_peaks = measure_levels(stream, 5, "Speech")
        
        logger.info("\nDiagnostic Information:")
        logger.info(f"Silence levels min/max: {float(min(silence_levels)):.6f}/{float(max(silence_levels)):.6f}")
        logger.info(f"Speech levels min/max: {float(min(speech_levels)):.6f}/{float(max(speech_levels)):.6f}")
        logger.info(f"Number of peaks detected: {len(speech_peaks)}")
        
        energy_threshold = calculate_threshold(silence_levels, speech_levels, speech_peaks)
        continued_ratio = 0.7
        
        new_profile = {
            "energy_threshold": float(round(energy_threshold, 6)),
            "continued_threshold_ratio": float(round(continued_ratio, 2)),
            "continued_threshold": float(round(energy_threshold * continued_ratio, 6)),
            "silence_duration": 3.0,
            "speech_buffer_time": 1.0,
            "min_speech_duration": 0.4,
            "max_recording_time": 30,
            "chunk_size": int(CHUNK)
        }
        
        settings_path = os.path.join(os.path.dirname(__file__), "VAD_settings.json")
        logger.debug(f"Attempting to save to: {settings_path}")
        
        try:
            settings_data = {
                "profiles": {
                    "current": new_profile
                },
                "active_profile": "current"
            }
            
            if save_settings_directly(settings_data, settings_path):
                logger.info("Settings saved directly successfully")
            else:
                converted_profile = convert_to_native_types(new_profile)
                success = save_vad_settings(converted_profile, "current")
                if not success:
                    raise Exception("Both save attempts failed")
            
            if verify_settings_file(settings_path):
                logger.info("Settings verified successfully")
                with open(settings_path, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    if saved_data.get("profiles", {}).get("current") != new_profile:
                        raise Exception("Saved settings don't match original")
            else:
                raise Exception("Settings verification failed")
            
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")
            backup_path = f"{settings_path}.backup.{int(time.time())}"
            try:
                save_settings_directly({
                    "profiles": {"current": new_profile},
                    "active_profile": "current"
                }, backup_path)
                logger.info(f"Backup saved to: {backup_path}")
            except Exception as backup_error:
                logger.error(f"Failed to create backup: {str(backup_error)}")
            raise
        
        print("\n=== CALIBRATION RESULTS ===")
        print(f"Background Noise:")
        print(f"  Mean: {float(np.mean(silence_levels)):.6f}")
        print(f"  Max: {float(max(silence_levels)):.6f}")
        print(f"  StdDev: {float(np.std(silence_levels)):.6f}")
        
        print(f"\nSpeech Levels:")
        print(f"  Mean: {float(np.mean(speech_levels)):.6f}")
        print(f"  Min: {float(min(speech_levels)):.6f}")
        print(f"  Max: {float(max(speech_levels)):.6f}")
        if speech_peaks:
            print(f"  Peak Mean: {float(np.mean(speech_peaks)):.6f}")
            print(f"  Peak Max: {float(max(speech_peaks)):.6f}")
        else:
            print("  No peaks detected")
        
        print(f"\nCalibrated Settings:")
        print(f"  Energy Threshold: {new_profile['energy_threshold']:.6f}")
        print(f"  Continued Ratio: {new_profile['continued_threshold_ratio']:.2f}")
        print(f"  Continued Threshold: {new_profile['continued_threshold']:.6f}")
        
        print("\nCalibration complete! Settings saved and active.")
        print("CALIBRATION_COMPLETE")
        
    except Exception as e:
        logger.error(f"Error during calibration: {str(e)}")
        raise
        
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
