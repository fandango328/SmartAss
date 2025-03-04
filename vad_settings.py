import os
import json

def load_vad_settings(active_profile=None):
    """
    Load VAD settings from JSON file
    
    Args:
        active_profile: Override the active profile in the JSON file
    
    Returns:
        Dictionary of VAD settings
    """
    VAD_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "VAD_settings.json")
    
    try:
        if os.path.exists(VAD_SETTINGS_PATH):
            with open(VAD_SETTINGS_PATH, "r") as f:
                vad_profiles = json.load(f)
                
                # Use provided profile or fall back to the one in the file
                profile_name = active_profile or vad_profiles.get("active_profile", "home_quiet")
                
                if profile_name not in vad_profiles["profiles"]:
                    print(f"Warning: Profile '{profile_name}' not found, using 'home_quiet'")
                    profile_name = "home_quiet"
                
                settings = vad_profiles["profiles"][profile_name]
                
                # Add the continued_threshold for process_frame
                settings["continued_threshold"] = settings["energy_threshold"] * settings.get("continued_threshold_ratio", 0.4)
                
                print(f"Using VAD profile: {profile_name}")
                return settings
        else:
            # Default settings if file doesn't exist
            return {
                "energy_threshold": 0.059616,
                "continued_threshold": 0.023846,
                "silence_duration": 3.0,
                "speech_buffer_time": 1.0,
                "max_recording_time": 60,
                "chunk_size": 2048
            }
    except Exception as e:
        print(f"Error loading VAD settings: {e}")
        # Fallback settings
        return {
            "energy_threshold": 0.059616,
            "continued_threshold": 0.023846,
            "silence_duration": 3.0,
            "speech_buffer_time": 1.0,
            "max_recording_time": 60,
            "chunk_size": 2048
        }

def get_available_profiles():
    """Return list of available VAD profiles"""
    VAD_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "VAD_settings.json")
    
    try:
        if os.path.exists(VAD_SETTINGS_PATH):
            with open(VAD_SETTINGS_PATH, "r") as f:
                vad_profiles = json.load(f)
                return list(vad_profiles["profiles"].keys())
        return []
    except Exception as e:
        print(f"Error getting profiles: {e}")
        return []