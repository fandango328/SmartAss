import asyncio
import os
import random
import json
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

RECURRENCE_TERMS = {
    "recurring": "recurring",    # standard spelling
    "reoccuring": "recurring",   # common misspelling
    "reoccurring": "recurring",  # another common variant
    "repeating": "recurring",    # alternative term
    "regular": "recurring",      # natural language variant
    "scheduled": "recurring"     # another natural term
}

class SystemManager:
    """
    Handles all system commands (loading files, enabling tools, etc.)
    Works with other managers to execute commands and provide audio feedback
    """

    def __init__(self, 
                 display_manager, 
                 audio_manager, 
                 document_manager,
                 notification_manager,
                 token_tracker):
        # Store references to other managers we'll need
        self.display_manager = display_manager
        self.audio_manager = audio_manager
        self.document_manager = document_manager
        self.notification_manager = notification_manager
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
            },
            "reminder": {
                "clear": [
                    "clear reminder", "dismiss reminder", "acknowledge reminder",
                    "clear notification", "dismiss notification", "I've finished my task", "I've taken my medicine" 
                ],
                "add": [
                    f"add {term} reminder" for term in RECURRENCE_TERMS.keys()
                ] + [
                    f"create {term} reminder" for term in RECURRENCE_TERMS.keys()
                ] + [
                    f"set {term} reminder" for term in RECURRENCE_TERMS.keys()
                ] + [
                    f"schedule {term} reminder" for term in RECURRENCE_TERMS.keys()
                ],
                "update": [
                    f"update {term} reminder" for term in RECURRENCE_TERMS.keys()
                ] + [
                    f"change {term} reminder" for term in RECURRENCE_TERMS.keys()
                ] + [
                    f"modify {term} reminder" for term in RECURRENCE_TERMS.keys()
                ],
                "list": [
                    f"list {term} reminders" for term in RECURRENCE_TERMS.keys()
                ] + [
                    f"show {term} reminders" for term in RECURRENCE_TERMS.keys()
                ] + [
                    f"active {term} reminders" for term in RECURRENCE_TERMS.keys()
                ]
            },
            "persona": {
                "switch": [
                    "switch persona", "change persona", "talk to", "switch to", "change to",
                    "load personality", "load character", "load assistant", "become", 
                    "switch personality", "change personality", "change character",
                    "switch character", "switch voice", "change voice", "use persona",
                    "activate persona", "activate personality", "activate character"
                ]
            }
        }
        
        # Debug flag for command detection
        self.debug_detection = False  # Set to True to see matching details

    def _normalize_command_input(self, transcript: str) -> str:
        """
        Normalize command input to handle natural language variations
        """
        # Common phrase mappings
        phrase_mappings = {
            "set up": "create",
            "make": "create",
            "setup": "create",
            "remove": "clear",
            "delete": "clear",
            "cancel": "clear",
            "get rid of": "clear",
            "turn off": "clear",
            "modify": "update",
            "change": "update",
            "fix": "update",
            "show me": "show",
            "tell me": "show",
            "what are": "show",
            "what's": "show",
            "daily": "recurring",
            "weekly": "recurring",
            "monthly": "recurring"
        }
        
        normalized = transcript.lower()
        for phrase, replacement in phrase_mappings.items():
            normalized = normalized.replace(phrase, replacement)
            
        return normalized

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

    def detect_command(self, transcript: str) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Check if the user's words match any of our command patterns
        Returns: 
        - is this a command? (True/False)
        - what type of command? (document/tool/calibration/reminder)
        - what action to take? (load/offload/enable/disable/clear/add/etc)
        - arguments for the command (if any)
        """
        normalized_transcript = self._normalize_command_input(transcript)
        
        for command_type, actions in self.command_patterns.items():
            for action, patterns in actions.items():
                # First try exact matches
                if any(pattern == normalized_transcript for pattern in patterns):
                    return True, command_type, action, None
                    
                # Then try component matching for more flexible detection
                for pattern in patterns:
                    components = pattern.split()
                    if self.has_command_components(normalized_transcript, components):
                        # Extract arguments (anything after the command pattern)
                        pattern_pos = normalized_transcript.find(pattern)
                        if pattern_pos >= 0:
                            cmd_end = pattern_pos + len(pattern)
                            arguments = transcript[cmd_end:].strip()  # Use original transcript for arguments
                            return True, command_type, action, arguments if arguments else None
        
        return False, None, None, None

    def _parse_schedule_days(self, day_input: str) -> list:
        """
        Parse day input into standardized schedule format
        """
        day_mapping = {
            'mon': 'Monday',
            'tue': 'Tuesday',
            'wed': 'Wednesday',
            'thu': 'Thursday',
            'fri': 'Friday',
            'sat': 'Saturday',
            'sun': 'Sunday'
        }
        
        if day_input.lower() in ['all', 'daily']:
            return list(day_mapping.values())
        elif day_input.lower() == 'weekdays':
            return list(day_mapping.values())[:5]
        elif day_input.lower() == 'weekends':
            return list(day_mapping.values())[5:]
        else:
            # Handle comma-separated days
            days = [d.strip().lower()[:3] for d in day_input.split(',')]
            return [day_mapping[d] for d in days if d in day_mapping]

    async def handle_command(self, command_type: str, action: str, arguments: str = None) -> bool:
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
                    print(f"Tool enable result: {result}")  # Debug output
                    if result["state"] == "enabled":
                        print("Tools successfully enabled")
                        folder_path = SOUND_PATHS['tool']['status']['enabled']
                        mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
                        if mp3_files:
                            audio_file = os.path.join(folder_path, random.choice(mp3_files))
                            if audio_file and os.path.exists(audio_file):
                                await self.audio_manager.play_audio(audio_file)
                                await self.audio_manager.wait_for_audio_completion()
                        await self.display_manager.update_display('listening')
                        return True
                    else:
                        print("Failed to enable tools")
                        return False
                else:  # disable
                    result = self.token_tracker.disable_tools()
                    print(f"Tool disable result: {result}")  # Debug output
                    if result["state"] == "disabled":
                        print("Tools successfully disabled")
                        folder_path = SOUND_PATHS['tool']['status']['disabled']
                        mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
                        if mp3_files:
                            audio_file = os.path.join(folder_path, random.choice(mp3_files))
                            if audio_file and os.path.exists(audio_file):
                                await self.audio_manager.play_audio(audio_file)
                                await self.audio_manager.wait_for_audio_completion()
                        await self.display_manager.update_display('listening')
                        return True
                    else:
                        print("Failed to disable tools")
                        return False
                    
            elif command_type == "calibration":
                success = await self._run_calibration()
                if success:
                    audio_file = os.path.join(SOUND_PATHS['calibration'],
                                            'voicecalibrationcomplete.mp3')

            elif command_type == "reminder":
                return await self._
