from enum import Enum
from typing import Optional, Tuple, Dict, Any
import asyncio
from pathlib import Path

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
        
        # Check each command type and its patterns
        for command_type, actions in self.command_patterns.items():
            for action, patterns in actions.items():
                if any(pattern in transcript_lower for pattern in patterns):
                    return True, command_type, action
        
        # If we get here, no command was found
        return False, None, None

    async def handle_command(self, command_type: str, action: str) -> bool:
        """
        Execute a command and provide audio feedback
        Returns True if command succeeded, False if it failed
        """
        try:
            # Show tools display
            await self.display_manager.update_display('tools')
            
            # Execute command based on type
            if command_type == "document":
                if action == "load":
                    await self.document_manager.load_all_files(clear_existing=False)
                    audio_file = self.audio_manager.get_random_audio('file', 'loaded')
                else:  # offload
                    await self.document_manager.offload_all_files()
                    audio_file = self.audio_manager.get_random_audio('file', 'offloaded')
                    
            elif command_type == "tool":
                if action == "enable":
                    self.token_tracker.enable_tools()
                    audio_file = self.audio_manager.get_random_audio('tool', 'status/enabled')
                else:  # disable
                    self.token_tracker.disable_tools()
                    audio_file = self.audio_manager.get_random_audio('tool', 'status/disabled')
                    
            elif command_type == "calibration":
                success = await self._run_calibration()
                audio_file = self.audio_manager.get_random_audio(
                    'calibration', 
                    'complete' if success else 'failed'
                )
            
            # Play audio feedback and wait for completion
            if audio_file:
                await self.audio_manager.play_audio(audio_file)
            await self.audio_manager.wait_for_audio_completion()
            
            # Update display to listening state
            await self.display_manager.update_display('listening')
            
            return True
            
        except Exception as e:
            print(f"Command error: {e}")
            return False

    async def _run_calibration(self) -> bool:
        """
        Run the voice calibration process
        Returns True if calibration succeeded, False if it failed
        """
        try:
            # Run the calibration script
            process = await asyncio.create_subprocess_exec(
                "python3", "vad_calib.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for it to finish and check result
            stdout, stderr = await process.communicate()
            return "CALIBRATION_COMPLETE" in stdout.decode()
            
        except Exception as e:
            print(f"Calibration error: {e}")
            return False
