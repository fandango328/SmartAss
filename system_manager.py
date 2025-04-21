import asyncio
import os
import random
import json
import traceback
import importlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from enum import Enum
from token_manager import TokenManager
from tts_handler import TTSHandler
import config
from secret import ELEVENLABS_KEY

# Configuration imports
from config import (
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
        
        # Initialize TTS handler
        from secret import ELEVENLABS_KEY
        self.tts_handler = TTSHandler({
            "TTS_ENGINE": config.TTS_ENGINE,
            "ELEVENLABS_KEY": ELEVENLABS_KEY,
            "VOICE": config.VOICE,
            "ELEVENLABS_MODEL": config.ELEVENLABS_MODEL,
        })

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
                    "tools off", "disable tool use", "stop tool",
                    "tools stop", "tool stop"
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
                    "change character to", "talk to", "switch to", "change to",
                    "load personality", "load character", "load assistant",
                    "switch personality", "change personality",
                    "switch character to", "switch voice", "change voice"
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
                    folder_path = os.path.join(f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/file_sentences/{'loaded' if load_success else 'error'}")
                    if os.path.exists(folder_path):
                        mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
                        if mp3_files:
                            audio_file = os.path.join(folder_path, random.choice(mp3_files))
                    if not load_success:
                        return False, None, None
                else:  # offload
                    await self.document_manager.offload_all_files()
                    folder_path = os.path.join(f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/file_sentences/offloaded")
                    if os.path.exists(folder_path):
                        mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
                        if mp3_files:
                            audio_file = os.path.join(folder_path, random.choice(mp3_files))

            elif command_type == "tool":
                try:
                    # Get the appropriate status type based on action
                    status_type = 'enabled' if action == 'enable' else 'disabled'
                    
                    # Show the specific tool state image
                    await self.display_manager.update_display('tools', specific_image=status_type)
                    
                    # Get the appropriate status folder based on action
                    folder_path = os.path.join(f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/tool_sentences/status/{status_type}")
                    
                    # Execute the tool state change
                    if action == "enable":
                        result = self.token_tracker.enable_tools()
                    else:  # disable
                        result = self.token_tracker.disable_tools()
                    
                    print(f"Tool {action} result: {result}")  # Debug output
                    
                    # Check if operation was successful
                    if isinstance(result, dict) and result.get("state") == status_type:
                        print(f"Tools successfully {status_type}")
                        
                        # Play appropriate audio feedback if available
                        if os.path.exists(folder_path):
                            mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
                            if mp3_files:
                                audio_file = os.path.join(folder_path, random.choice(mp3_files))
                                if os.path.exists(audio_file):
                                    await self.audio_manager.play_audio(audio_file)
                                    await self.audio_manager.wait_for_audio_completion()
                        
                        # Return success without changing display state
                        return True, None, None
                    else:
                        print(f"Failed to {action} tools")
                        return False, None, None
                
                except Exception as e:
                    print(f"Error during tool {action}: {e}")
                    traceback.print_exc()
                    return False, None, None

            elif command_type == "calibration":
                success = await self._run_calibration()
                if success:
                    folder_path = os.path.join(f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/calibration_sentences")
                    if os.path.exists(folder_path):
                        mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
                        if mp3_files:
                            audio_file = os.path.join(folder_path, random.choice(mp3_files))

            elif command_type == "reminder":
                return await self._handle_reminder_command(action, arguments)

            elif command_type == "persona":
                return await self._handle_persona_command(action, arguments)

            if audio_file and os.path.exists(audio_file):
                await self.audio_manager.play_audio(audio_file)
            await self.audio_manager.wait_for_audio_completion()

            await self.display_manager.update_display('listening')
            return True, None, None

        except Exception as e:
            error_msg = f"Command error: {str(e)}"
            print(error_msg)
            print(f"Traceback: {traceback.format_exc()}")
            return False, None, None

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
                    audio_folder = os.path.join(f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/reminder_sentences/{'cleared' if success else 'error'}")
                else:
                    # No reminder ID provided, show list of active reminders
                    active_reminders = await self.notification_manager.get_active_reminders()
                    if active_reminders:
                        print("Active reminders:")
                        for rid, details in active_reminders.items():
                            print(f"- {rid}: {details['type']} ({details['created']})")
                        success = True
                        audio_folder = os.path.join(f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/reminder_sentences/list")
                    else:
                        audio_folder = os.path.join(f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/reminder_sentences/none")

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
                            audio_folder = os.path.join(f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/reminder_sentences/added")
                        except ValueError as e:
                            print(f"Error adding reminder: {e}")
                            audio_folder = os.path.join(f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/reminder_sentences/error")
                    else:
                        audio_folder = os.path.join(f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/reminder_sentences/error")
                else:
                    audio_folder = os.path.join(f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/reminder_sentences/error")

            # Play appropriate audio feedback
            if audio_folder and os.path.exists(audio_folder):
                mp3_files = [f for f in os.listdir(audio_folder) if f.endswith('.mp3')]
                if mp3_files:
                    audio_file = os.path.join(audio_folder, random.choice(mp3_files))
                    await self.audio_manager.play_audio(audio_file)
                    await self.audio_manager.wait_for_audio_completion()
            return success

        except Exception as e:
            print(f"Reminder command error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return False


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
        print(f"DEBUG - Normalized transcript: '{normalized_transcript}'")
        
        # Load available personas for command context
        try:
            with open("personalities.json", 'r') as f:
                personas_data = json.load(f)
                available_personas = [p.lower() for p in personas_data.get("personas", {}).keys()]
        except Exception as e:
            print(f"Error loading personas: {e}")
            available_personas = []
        
        # For persona commands, check if any persona name exists in the transcript
        persona_found = None
        for persona in available_personas:
            if persona in normalized_transcript.lower():
                persona_found = persona
                break
        
        # Handle all commands including persona
        for command_type, actions in self.command_patterns.items():
            for action, patterns in actions.items():
                for pattern in patterns:
                    if pattern in normalized_transcript:
                        # Extract arguments after the pattern if any
                        start_idx = normalized_transcript.find(pattern) + len(pattern)
                        args = normalized_transcript[start_idx:].strip()
                        
                        # Special handling for persona commands - only trigger if valid persona found
                        if command_type == "persona":
                            if persona_found:
                                print(f"DEBUG - Matched persona command with target: {persona_found}")
                                return True, command_type, action, persona_found
                            else:
                                print(f"DEBUG - Persona command matched but no valid persona found")
                                return False, None, None, None
                        
                        return True, command_type, action, args if args else None
        
        print("DEBUG - No command pattern matched")
        return False, None, None, None

    async def _handle_persona_command(self, action: str, arguments: str = None) -> bool:
        """Handle persona-related commands with transition animations"""
        try:
            # Log the command details
            print(f"\nDEBUG: Handling persona command - Action: {action}, Arguments: {arguments}")
            
            # Import config module at function start
            import config as config_module
            
            # Step 1: Update to system state with current persona's "out" animation
            current_persona = config_module.ACTIVE_PERSONA.lower()
            out_path = f"/home/user/LAURA/pygame/{current_persona}/system/persona/out"
            default_image = "/home/user/LAURA/pygame/laura/system/persona/dont_touch_this_image.png"
            
            print(f"DEBUG: Transitioning from {current_persona} with path: {out_path}")
            
            # Check if persona out directory exists with images
            out_path_dir = Path(out_path)
            if out_path_dir.exists() and any(out_path_dir.glob('*.png')):
                # Use transition path to display persona-specific "out" animation
                await self.display_manager.update_display('system', transition_path=str(out_path_dir))
            else:
                # Fall back to default persona transition image
                print(f"Warning: No persona exit animations found at {out_path}")
                print(f"Using default persona transition image: {default_image}")
                default_dir = Path(default_image).parent
                await self.display_manager.update_display('system', transition_path=str(default_dir), 
                                                         specific_image=default_image)
            
            # Load personality configuration
            persona_path = "personalities.json"
            try:
                with open(persona_path, 'r') as f:
                    personas_data = json.load(f)
            except FileNotFoundError:
                print("DEBUG: Creating default personalities configuration")
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
            
            # Find target persona
            if action == "switch":
                # If arguments is empty or None, handle it gracefully
                if not arguments:
                    print("DEBUG: No persona specified in switch command")
                    return False
                    
                normalized_input = arguments.strip().lower()
                target_persona = None
                
                # Check if arguments exactly matches a persona key
                if normalized_input in personas_data.get("personas", {}):
                    target_persona = normalized_input
                else:
                    # Try to find a matching persona
                    for key in personas_data.get("personas", {}):
                        if key.lower() == normalized_input:
                            target_persona = key
                            break
                
                if target_persona:
                    print(f"DEBUG: Switching to persona: {target_persona}")
                    
                    # Step 2: Update the base display path
                    new_base_path = str(Path(f'/home/user/LAURA/pygame/{target_persona.lower()}'))
                    await self.display_manager.update_display_path(new_base_path)
                    
                    # Step 3: Show incoming animation
                    in_path = f"/home/user/LAURA/pygame/{target_persona.lower()}/system/persona/in"
                    print(f"DEBUG: Loading incoming animation from: {in_path}")
                    
                    # Check if persona in directory exists with images
                    in_path_dir = Path(in_path)
                    if in_path_dir.exists() and any(in_path_dir.glob('*.png')):
                        # Use transition path to display persona-specific "in" animation
                        await self.display_manager.update_display('system', transition_path=str(in_path_dir))
                        # Add a small delay to show the animation
                        await asyncio.sleep(0.5)
                    else:
                        # Fall back to default persona transition image
                        print(f"Warning: No persona entry animations found at {in_path}")
                        print(f"Using default persona transition image: {default_image}")
                        default_dir = Path(default_image).parent
                        await self.display_manager.update_display('system', transition_path=str(default_dir), 
                                                                 specific_image=default_image)
                        # Add a small delay to show the animation
                        await asyncio.sleep(0.5)
                    
                    # Step 4: Update configuration
                    try:
                        # Update active persona
                        personas_data["active_persona"] = target_persona
                        with open(persona_path, 'w') as f:
                            json.dump(personas_data, f, indent=2)
                        
                        # Reload config and update system
                        importlib.reload(config_module)
                        
                        config_module.ACTIVE_PERSONA = target_persona
                        config_module.ACTIVE_PERSONA_DATA = personas_data["personas"][target_persona]
                        config_module.VOICE = personas_data["personas"][target_persona].get("voice", "L.A.U.R.A.")
                        new_prompt = personas_data["personas"][target_persona].get("system_prompt", "You are an AI assistant.")
                        config_module.SYSTEM_PROMPT = f"{new_prompt}\n\n{config_module.UNIVERSAL_SYSTEM_PROMPT}"
                        
                        # Reinitialize TTS handler
                        from secret import ELEVENLABS_KEY
                        new_config = {
                            "TTS_ENGINE": config_module.TTS_ENGINE,
                            "ELEVENLABS_KEY": ELEVENLABS_KEY,
                            "VOICE": config_module.VOICE,
                            "ELEVENLABS_MODEL": config_module.ELEVENLABS_MODEL,
                        }
                        self.tts_handler = TTSHandler(new_config)
                        
                        print(f"DEBUG: Successfully switched to persona: {target_persona}")
                        print(f"DEBUG: Using voice: {config_module.VOICE}")
                        
                        # Step 5: Transition to listening state
                        await asyncio.sleep(0.1)  # Small buffer for state change
                        await self.display_manager.update_display('listening')
                        
                        return True
                        
                    except Exception as e:
                        print(f"ERROR: Failed to update configuration: {e}")
                        traceback.print_exc()
                        return False
                else:
                    print(f"ERROR: Persona '{arguments}' not found")
                    return False
            
            return False
            
        except Exception as e:
            print(f"ERROR: Persona command failed: {e}")
            traceback.print_exc()
            return False
            
    async def show_tool_state(self, action):
        """
        Display tool enabling/disabling state image
        
        Args:
            action: Either 'enable' or 'disable'
        """
        status_type = 'enabled' if action == 'enable' else 'disabled'
        
        if status_type not in ['enabled', 'disabled']:
            print(f"Invalid tool action: {action}")
            return False
            
        try:
            await self.display_manager.update_display('tools', specific_image=status_type)
            return True
        except Exception as e:
            print(f"Error showing tool {status_type} image: {e}")
            return False

            
    async def show_calibration_image(self):
        """Display calibration image during voice calibration"""
        try:
            await self.display_manager.update_display('calibration')
            return True
        except Exception as e:
            print(f"Error showing calibration image: {e}")
            return False

    async def show_document_state(self, action):
        """
        Display document loading/unloading state image
        
        Args:
            action: Either 'load' or 'unload'
        """
        if action not in ['load', 'unload']:
            print(f"Invalid document action: {action}")
            return False
            
        try:
            await self.display_manager.update_display('document', specific_image=action)
            return True
        except Exception as e:
            print(f"Error showing document {action} image: {e}")
            return False
