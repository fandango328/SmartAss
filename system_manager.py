import asyncio
import os
import random
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from enum import Enum
from token_manager import TokenManager

# Configuration imports
from config_cl import (
    VAD_SETTINGS,
    SOUND_PATHS
)

class SystemManager:
    """
    Handles all system commands (loading files, enabling tools, etc.)
    Works with other managers to execute commands and provide audio feedback
    """
    def __init__(self, 
                 display_manager, 
                 audio_manager, 
                 document_manager, 
                 token_tracker):
        # Store references to other managers we'll need
        self.display_manager = display_manager
        self.audio_manager = audio_manager
        self.document_manager = document_manager
        self.token_tracker = token_tracker
        
        self.command_patterns = {
            "document": {
                "load": [
                    "load file", "load files", "load all files", 
                    "load my file", "load my files"
                ],
                "offload": [
                    "offload file", "offload files", "clear files",
                    "remove files", "clear all files"
                ]
            },
            "tool": {
                "enable": [
                    "tools activate", "enable tools", "start tools",
                    "tools online", "enable tool use"
                ],
                "disable": [
                    "tools offline", "disable tools", "stop tools",
                    "tools off", "disable tool use"
                ]
            },
            "calibration": {
                "calibrate": [
                    "calibrate voice", "calibrate detection",
                    "voice calibration", "detection calibration"
                ]
            }
        }
        
        # Debug flag for command detection
        self.debug_detection = False  # Set to True to see matching details

    def has_command_components(self, transcript: str, required_words: list, max_word_gap: int = 2) -> bool:
        """
        Check if command words appear within 2 words of each other in the transcript
        Example: "load of my file" would match ["load", "file"] but 
                "load something else then file" would not
        """
        words = transcript.lower().split()
        positions = []
        
        if self.debug_detection:
            print(f"Checking command components: {required_words}")
            print(f"Transcript words: {words}")
        
        for req_word in required_words:
            found = False
            for i, word in enumerate(words):
                if req_word in word:
                    positions.append(i)
                    found = True
                    break
            if not found:
                return False
                            
        # Check if words are within 2 words of each other
        positions.sort()
        
        if self.debug_detection:
            print(f"Word positions: {positions}")
            
        return all(positions[i+1] - positions[i] <= 2 
                  for i in range(len(positions)-1))

    def detect_command(self, transcript: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if the user's words match any of our command patterns
        Returns: 
        - is this a command? (True/False)
        - what type of command? (document/tool/calibration)
        - what action to take? (load/offload/enable/disable)
        """
        transcript_lower = transcript.lower().strip()
        
        for command_type, actions in self.command_patterns.items():
            for action, patterns in actions.items():
                # First try exact matches
                if any(pattern == transcript_lower for pattern in patterns):
                    return True, command_type, action
                    
                # Then try component matching for more flexible detection
                for pattern in patterns:
                    components = pattern.split()
                    if self.has_command_components(transcript_lower, components):
                        return True, command_type, action
        
        return False, None, None

    async def handle_command(self, command_type: str, action: str) -> bool:
        """
        Execute a command and provide audio feedback
        Returns True if command succeeded, False if it failed
        """
        try:
            await self.display_manager.update_display('tools')
            audio_file = None
            
            if command_type == "document":
                if action == "load":
                    load_success = await self.document_manager.load_all_files(clear_existing=False)
                    folder_path = SOUND_PATHS['file']['loaded' if load_success else 'error']
                    mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
                    if mp3_files:
                        audio_file = os.path.join(folder_path, random.choice(mp3_files))
                    if not load_success:
                        return False
                else:  # offload
                    await self.document_manager.offload_all_files()
                    folder_path = SOUND_PATHS['file']['offloaded']
                    mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
                    if mp3_files:
                        audio_file = os.path.join(folder_path, random.choice(mp3_files))
                    
            elif command_type == "tool":
                if action == "enable":
                    result = self.token_tracker.enable_tools()
                    if result["state"] == "enabled":
                        folder_path = SOUND_PATHS['tool']['status']['enabled']
                        mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
                        if mp3_files:
                            audio_file = os.path.join(folder_path, random.choice(mp3_files))
                else:  # disable
                    result = self.token_tracker.disable_tools()
                    if result["state"] == "disabled":
                        folder_path = SOUND_PATHS['tool']['status']['disabled']
                        mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
                        if mp3_files:
                            audio_file = os.path.join(folder_path, random.choice(mp3_files))
                    
            elif command_type == "calibration":
                success = await self._run_calibration()
                if success:
                    audio_file = os.path.join(SOUND_PATHS['calibration'],
                                            'voicecalibrationcomplete.mp3')

            if audio_file and os.path.exists(audio_file):
                await self.audio_manager.play_audio(audio_file)
            await self.audio_manager.wait_for_audio_completion()
            
            await self.display_manager.update_display('listening')
            
            return True
            
        except Exception as e:
            error_msg = f"Command error: {str(e)}"
            print(error_msg)
            print(f"Traceback: {traceback.format_exc()}")
            
            error_audio = os.path.join(SOUND_PATHS['system']['error'], 'error.mp3')
            if os.path.exists(error_audio):
                await self.audio_manager.play_audio(error_audio)
            
            return False

    async def _run_calibration(self) -> bool:
        """
        Run the voice calibration process
        Returns True if calibration succeeded, False if it failed
        """
        try:
            calib_script = Path("vad_calib.py").absolute()
            
            process = await asyncio.create_subprocess_exec(
                "python3", str(calib_script),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            success = "CALIBRATION_COMPLETE" in stdout.decode()
            
            if success:
                print("Voice calibration completed successfully")
            else:
                print("Voice calibration failed")
                if stderr:
                    print(f"Calibration error: {stderr.decode()}")
                    
            return success
            
        except Exception as e:
            print(f"Calibration error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return False
