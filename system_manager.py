import asyncio
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from enum import Enum

# Configuration imports
from config_cl import (
    VAD_SETTINGS,
    SOUND_PATHS
)

# Token state management
from token_manager import TokenState, TokenManager

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
        
        # Define our command patterns - what words trigger what commands
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

    def detect_command(self, transcript: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if the user's words match any of our command patterns
        Returns: 
        - is this a command? (True/False)
        - what type of command? (document/tool/calibration)
        - what action to take? (load/offload/enable/disable)
        """
        transcript_lower = transcript.lower()
        
        for command_type, actions in self.command_patterns.items():
            for action, patterns in actions.items():
                if any(pattern in transcript_lower for pattern in patterns):
                    return True, command_type, action
        
        return False, None, None

    async def handle_command(self, command_type: str, action: str) -> bool:
        """
        Execute a command and provide audio feedback
        Returns True if command succeeded, False if it failed
        """
        try:
            await self.display_manager.update_display('tools')
            
            if command_type == "document":
                if action == "load":
                    load_success = await self.document_manager.load_all_files(clear_existing=False)
                    if load_success:
                        audio_file = self.audio_manager.get_random_audio('file', 'loaded')
                    else:
                        audio_file = self.audio_manager.get_random_audio('file', 'error')
                        return False
                else:  # offload
                    await self.document_manager.offload_all_files()
                    audio_file = self.audio_manager.get_random_audio('file', 'offloaded')
                    
            elif command_type == "tool":
                if action == "enable":
                    self.token_tracker.enable_tools()
                    if self.token_tracker.current_state == TokenState.TOOLS_ENABLED:
                        audio_file = self.audio_manager.get_random_audio('tool', 'status/enabled')
                else:  # disable
                    self.token_tracker.disable_tools()
                    if self.token_tracker.current_state == TokenState.TOOLS_DISABLED:
                        audio_file = self.audio_manager.get_random_audio('tool', 'status/disabled')
                    
            elif command_type == "calibration":
                success = await self._run_calibration()
                audio_file = self.audio_manager.get_random_audio(
                    'calibration', 
                    'complete' if success else 'failed'
                )
            
            if audio_file:
                await self.audio_manager.play_audio(audio_file)
            await self.audio_manager.wait_for_audio_completion()
            
            await self.display_manager.update_display('listening')
            
            return True
            
        except Exception as e:
            error_msg = f"Command error: {str(e)}"
            print(error_msg)
            print(f"Traceback: {traceback.format_exc()}")
            
            error_audio = self.audio_manager.get_random_audio('system', 'error')
            if error_audio:
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
