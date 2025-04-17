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

PERSONA_SWITCH_INTENTS = [
    "talk to",
    "switch to",
    "change to",
    "i want to talk to",
    "let me talk to"
]

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
        - what type of command? (document/tool/calibration/reminder/persona)
        - what action to take? (load/offload/enable/disable/switch)
        - arguments for the command (if any)
        """
        normalized_transcript = self._normalize_command_input(transcript)
        
        # First check for persona commands as they need special handling
        try:
            with open("personalities.json", 'r') as f:
                personas_data = json.load(f)
                available_personas = personas_data.get("personas", {}).keys()
                
                # Check each persona switch intent
                for intent in PERSONA_SWITCH_INTENTS:
                    if intent in normalized_transcript:
                        # Check if any persona name appears after the intent
                        for persona in available_personas:
                            if persona.lower() in normalized_transcript:
                                return True, "persona", "switch", persona.lower()
        except Exception as e:
            print(f"Error checking personas: {e}")
        
        # Handle all other commands
        for command_type, actions in self.command_patterns.items():
            if command_type != "persona":  # Skip persona commands as already handled
                for action, patterns in actions.items():
                    for pattern in patterns:
                        if pattern in normalized_transcript:
                            # Extract arguments after the pattern if any
                            start_idx = normalized_transcript.find(pattern) + len(pattern)
                            args = normalized_transcript[start_idx:].strip()
                            return True, command_type, action, args if args else None
        
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

    async def handle_command(self, command_type: str, action: str, arguments: str = None) -> Tuple[bool, Optional[str], Optional[str]]:
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
                    try:
                        result = self.token_tracker.enable_tools()
                        print(f"Tool enable result: {result}")  # Debug output
                        
                        if isinstance(result, dict) and result.get("state") == "enabled":
                            print("Tools successfully enabled")
                            # Play success audio
                            folder_path = SOUND_PATHS['tool']['status']['enabled']
                            if os.path.exists(folder_path):
                                mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
                                if mp3_files:
                                    audio_file = os.path.join(folder_path, random.choice(mp3_files))
                                    if os.path.exists(audio_file):
                                        await self.audio_manager.play_audio(audio_file)
                                        await self.audio_manager.wait_for_audio_completion()
                            
                            await self.display_manager.update_display('listening')
                            return True, None, None  # Return tuple of 3 values as expected
                        else:
                            print("Failed to enable tools")
                            return False, None, None  # Return tuple of 3 values as expected
                            
                    except Exception as e:
                        print(f"Error enabling tools: {e}")
                        return False, None, None  # Return tuple of 3 values as expected

            elif command_type == "calibration":
                success = await self._run_calibration()
                if success:
                    audio_file = os.path.join(SOUND_PATHS['calibration'],
                                            'voicecalibrationcomplete.mp3')

            elif command_type == "reminder":
                return await self._handle_reminder_command(action, arguments)

            elif command_type == "persona":
                return await self._handle_persona_command(action, arguments)

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

    async def _handle_reminder_command(self, action: str, arguments: str = None) -> bool:
        """
        Handle reminder-related commands
        Returns True if command succeeded, False if it failed
        """
        try:
            await self.display_manager.update_display('tools')
            success = False
            audio_folder = None

            if action == "clear":
                if arguments:
                    success = await self.notification_manager.clear_notification(arguments)
                    audio_folder = SOUND_PATHS['reminder']['cleared'] if success else SOUND_PATHS['reminder']['error']
                else:
                    # No reminder ID provided, show list of active reminders
                    active_reminders = await self.notification_manager.get_active_reminders()
                    if active_reminders:
                        print("Active reminders:")
                        for rid, details in active_reminders.items():
                            print(f"- {rid}: {details['type']} ({details['created']})")
                        success = True
                        audio_folder = SOUND_PATHS['reminder']['list']
                    else:
                        audio_folder = SOUND_PATHS['reminder']['none']

            elif action == "add":
                if arguments:
                    import shlex
                    try:
                        args = shlex.split(arguments)
                    except ValueError:
                        args = arguments.split()

                    if len(args) >= 2:
                        reminder_type = args[0]
                        time = args[1]
                        day_input = ' '.join(args[2:]) if len(args) > 2 else "all"

                        try:
                            schedule_days = self._parse_schedule_days(day_input)
                            await self.notification_manager.add_recurring_reminder(
                                reminder_type=reminder_type,
                                schedule={
                                    "time": time,
                                    "days": schedule_days,
                                    "recurring": True
                                }
                            )
                            print(f"Added recurring reminder '{reminder_type}' for {time} on {', '.join(schedule_days)}")
                            success = True
                            audio_folder = SOUND_PATHS['reminder']['added']
                        except ValueError as e:
                            print(f"Error adding reminder: {e}")
                            audio_folder = SOUND_PATHS['reminder']['error']
                    else:
                        audio_folder = SOUND_PATHS['reminder']['error']
                else:
                    audio_folder = SOUND_PATHS['reminder']['error']

            # Play appropriate audio feedback
            if audio_folder and os.path.exists(audio_folder):
                mp3_files = [f for f in os.listdir(audio_folder) if f.endswith('.mp3')]
                if mp3_files:
                    audio_file = os.path.join(audio_folder, random.choice(mp3_files))
                    await self.audio_manager.play_audio(audio_file)
                    await self.audio_manager.wait_for_audio_completion()

            await self.display_manager.update_display('listening')
            return success

        except Exception as e:
            print(f"Reminder command error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            error_audio = os.path.join(SOUND_PATHS['system']['error'], 'error.mp3')
            if os.path.exists(error_audio):
                await self.audio_manager.play_audio(error_audio)
            return False

    async def _handle_persona_command(self, action: str, arguments: str = None) -> bool:
        """
        Handle persona-related commands using pre-recorded audio files
        Returns True if command succeeded, False if it failed
        """
        # Initialize personas_data outside try block so it's accessible in catch-all error handler
        personas_data = None  
        
        try:
            # Step 1: Update display to 'tools' state to show we're processing a command
            try:
                await self.display_manager.update_display('tools')
            except Exception as e:
                print(f"Warning: Failed to update display to tools state: {e}")
            
            success = False
            
            # Step 2: Load or create personalities configuration
            persona_path = "personalities.json"
            try:
                with open(persona_path, 'r') as f:
                    personas_data = json.load(f)
            except FileNotFoundError:
                # Create default config if file doesn't exist
                personas_data = {
                    "personas": {
                        "laura": {
                            "voice": "L.A.U.R.A.",
                            "system_prompt": "You are Laura (Language & Automation User Response Agent), a professional and supportive AI-powered smart assistant."
                        }
                    },
                    "active_persona": "laura"
                }
                with open(persona_path, 'w') as f:
                    json.dump(personas_data, f, indent=2)
            
            # Step 3: Get available personas from config
            available_personas = personas_data.get("personas", {})
            
            # Step 4: Process the switch command
            if action == "switch":
                # Normalize input for case-insensitive matching
                normalized_input = arguments.strip().lower() if arguments else ""
                
                # Step 5: Find matching persona in available personas
                target_persona = None
                for key in available_personas:
                    if key.lower() in normalized_input:
                        target_persona = key
                        break
                
                # Step 6: If matching persona found, process the switch
                if target_persona:
                    # Step 7: Check for persona's audio files
                    audio_path = Path(f'/home/user/LAURA/sounds/{target_persona}/persona_sentences')
                    
                    if audio_path.exists():
                        # Get list of MP3 files in persona directory
                        audio_files = list(audio_path.glob('*.mp3'))
                        
                        if audio_files:
                            try:
                                # Step 8: Play switch announcement audio
                                chosen_audio = str(random.choice(audio_files))
                                await self.audio_manager.play_audio(chosen_audio)
                                await self.audio_manager.wait_for_audio_completion()
                                
                                # Step 9: Update active persona in config file and reload configuration
                                personas_data["active_persona"] = target_persona
                                with open(persona_path, 'w') as f:
                                    json.dump(personas_data, f, indent=2)
                                
                                # Update configuration in memory and reload
                                try:
                                    from config_cl import PERSONALITIES_DATA, ACTIVE_PERSONA, ACTIVE_PERSONA_DATA
                                    globals()['PERSONALITIES_DATA'] = personas_data
                                    globals()['ACTIVE_PERSONA'] = target_persona
                                    globals()['ACTIVE_PERSONA_DATA'] = personas_data.get("personas", {}).get(target_persona, {})
                                    print(f"Configuration updated, new active persona: {target_persona}")
                                except Exception as reload_error:
                                    print(f"Error updating configuration: {reload_error}")
                                
                                # Step 10: Update display manager with new persona path
                                try:
                                    await self.display_manager.update_display_path(
                                        str(Path(f'/home/user/LAURA/pygame/{target_persona}'))
                                    )
                                    success = True
                                except Exception as e:
                                    print(f"Error updating display path: {e}")
                                    success = False
                            except Exception as e:
                                print(f"Error during persona switch: {e}")
                                success = False
                        else:
                            print(f"No audio files found in {audio_path}")
                            success = False
                    else:
                        print(f"Audio path not found: {audio_path}")
                        success = False
                else:
                    print(f"Persona '{arguments}' not found")
                    success = False
            
            # Step 11: Return to listening state
            try:
                await self.display_manager.update_display('listening')
            except Exception as e:
                print(f"Warning: Failed to update display to listening state: {e}")
            
            return success
            
        except Exception as e:
            # Step 12: Handle any unexpected errors
            print(f"Persona command error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return False
