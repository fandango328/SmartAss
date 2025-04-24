#!/usr/bin/env python3

import asyncio
import os
import random
import json
import traceback
import importlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from token_manager import TokenManager
from tts_handler import TTSHandler
from laura_tools import tool_registry
import config
from secret import ELEVENLABS_KEY

try:
    from LAURA_email import (
        get_random_audio,
        get_current_time,
        get_location,
        run_vad_calibration,
        create_calendar_event, 
        update_calendar_event,
        cancel_calendar_event, 
        handle_calendar_query,
        manage_tasks, 
        create_task_from_email,
        create_task_for_event
    )
except ImportError:
    # Fallback if import fails
    def get_random_audio(category, context=None):
        """Fallback implementation if main function unavailable"""
        print(f"Warning: Using fallback get_random_audio for {category}/{context}")
        return None
        
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
    Manages system-level operations with synchronized display and audio feedback.
    
    The system follows a hierarchical structure:
    /system/
    ├── tools/                # Tool-related states
    │   ├── enabled/          # Tool activation state
    │   ├── disabled/         # Tool deactivation state
    │   └── use/             # Tool execution state
    ├── calibration/          # Calibration resources
    ├── document/            # Document operation states
    │   ├── load/           # Document loading state
    │   └── unload/         # Document unloading state
    └── persona/            # Persona transition states
        ├── in/            # Persona activation
        └── out/           # Persona deactivation
        
    State Transitions:
    1. Tool Operations:
       - Pre-execution: thinking state during validation
       - Execution: tool use display + synchronized audio
       - Post-execution: return to listening
       
    2. Resource Management:
       - Uses DisplayManager's path resolution
       - Implements proper audio synchronization
       - Maintains consistent state during transitions
       
    3. Error Handling:
       - Uses existing states for error conditions
       - Provides graceful degradation
       - Ensures state consistency
    """

    def __init__(self,
                 display_manager,
                 audio_manager,
                 document_manager,
                 notification_manager,
                 token_manager):
        """
        Initialize SystemManager with required components.
        
        Args:
            display_manager: Handles visual feedback and animations
            audio_manager: Controls audio playback and synchronization
            document_manager: Manages document loading and state
            notification_manager: Handles system notifications
            token_manager: Manages token usage
        """
        self.display_manager = display_manager
        self.audio_manager = audio_manager
        self.document_manager = document_manager
        self.notification_manager = notification_manager
        self.token_manager = token_manager
        
        # Set up bidirectional reference if needed
        if hasattr(token_manager, 'set_system_manager'):
            token_manager.set_system_manager(self)
                
        # Initialize TTS handler
        self.tts_handler = TTSHandler({
            "TTS_ENGINE": config.TTS_ENGINE,
            "ELEVENLABS_KEY": ELEVENLABS_KEY,
            "VOICE": config.VOICE,
            "ELEVENLABS_MODEL": config.ELEVENLABS_MODEL,
        })

        # Register core tool handlers with the global tool registry
        try:
            tool_registry.register_handlers({
                "get_current_time": get_current_time,
                "get_location": get_location,
                "calibrate_voice_detection": run_vad_calibration,
                "create_calendar_event": create_calendar_event,
                "update_calendar_event": update_calendar_event,
                "cancel_calendar_event": cancel_calendar_event,
                "calendar_query": handle_calendar_query,
                "draft_email": email_manager.draft_email,
                "read_emails": email_manager.read_emails,
                "email_action": email_manager.email_action,
                "manage_tasks": manage_tasks,
                "create_task_from_email": create_task_from_email,
                "create_task_for_event": create_task_for_event
            })
            print("Tool handlers registered successfully")
        except Exception as e:
            print(f"Error registering tool handlers: {e}")
            traceback.print_exc()

        # Command patterns with enhanced natural language support
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
                    "clear notification", "dismiss notification", 
                    "I've finished my task", "I've taken my medicine"
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
        self.debug_detection = False

    def _validate_llm_response(self, content) -> str:
        """
        Validate and sanitize LLM response content before processing.
        Prevents binary data contamination and ensures proper content type.
        
        Args:
            content: Raw response from LLM API
            
        Returns:
            str: Validated and sanitized text content
            
        Raises:
            ValueError: If content contains binary data or invalid formats
        """
        if isinstance(content, (bytes, bytearray)):
            raise ValueError("Binary content detected in LLM response")
            
        if isinstance(content, str):
            text = content
        elif hasattr(content, 'text'):
            text = content.text
        elif isinstance(content, list):
            text = ""
            for block in content:
                if hasattr(block, 'text'):
                    text += block.text
                elif isinstance(block, dict) and block.get('type') == 'text':
                    text += block.get('text', '')
                elif isinstance(block, str):
                    text += block
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
            
        if any(indicator in text for indicator in [
            'LAME3', '\xFF\xFB',  # MP3 headers
            '\x89PNG',            # PNG header
            '\xFF\xD8\xFF'       # JPEG header
        ]):
            raise ValueError("Binary data detected in text content")
            
        return text.strip()

    async def _handle_tool_use_sequence(self, tool_response: str) -> bool:
        """
        Coordinate tool use sequence with proper state transitions and audio sync.
        
        The sequence follows:
        1. Update display to tool use state
        2. Start pre-recorded audio
        3. Process response content
        4. Wait for TTS generation
        5. Handle audio transition
        6. Restore appropriate display state
        
        Args:
            tool_response: Response from tool execution
            
        Returns:
            bool: Success status of the sequence
        """
        try:
            # Step 1: Update display to tool use state
            await self.display_manager.update_display('tools', specific_image='use')
            
            # Step 2: Queue and start pre-recorded audio
            audio_folder = os.path.join(
                f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/tool_sentences/use"
            )
            prerecorded_task = None
            if os.path.exists(audio_folder):
                mp3_files = [f for f in os.listdir(audio_folder) if f.endswith('.mp3')]
                if mp3_files:
                    audio_file = os.path.join(audio_folder, random.choice(mp3_files))
                    prerecorded_task = asyncio.create_task(
                        self.audio_manager.play_audio(audio_file)
                    )
            
            # Step 3: Process response content
            try:
                validated_content = self._validate_llm_response(tool_response)
                if not validated_content:
                    raise RuntimeError("Failed to process tool response content")
                    
                # Step 4: Generate TTS audio
                tts_audio = await self.tts_handler.generate_audio(validated_content)
                if not tts_audio:
                    raise RuntimeError("Failed to generate TTS audio")
                    
                # Save TTS audio
                with open("speech.mp3", "wb") as f:
                    f.write(tts_audio)
                    
                # Step 5: Handle audio transition
                if prerecorded_task:
                    try:
                        await prerecorded_task
                    except Exception as e:
                        print(f"Warning: Error in pre-recorded audio: {e}")
                        
                # Add mandatory buffer
                await asyncio.sleep(0.5)
                
                # Step 6: Update display and play TTS
                await self.display_manager.update_display('speaking', mood='casual')
                await self.audio_manager.play_audio("speech.mp3")
                
                # Wait for completion
                await self.audio_manager.wait_for_audio_completion()
                
                # Return to listening state
                await self.display_manager.update_display('listening')
                return True
                
            except Exception as e:
                print(f"Error in tool use sequence: {e}")
                traceback.print_exc()
                await self.display_manager.update_display('listening')
                return False
                
        except Exception as e:
            print(f"Critical error in tool use sequence: {e}")
            traceback.print_exc()
            await self.display_manager.update_display('listening')
            return False

    def _normalize_command_input(self, transcript: str) -> str:
        """
        Normalize command input for consistent pattern matching.
        Handles common variations and misspellings.
        
        Args:
            transcript: Raw user input
            
        Returns:
            str: Normalized command text
        """
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

    def detect_command(self, transcript: str) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Parse user input for system commands with enhanced pattern matching.
        
        Args:
            transcript: User's speech input
            
        Returns:
            Tuple containing:
            - bool: Is this a command?
            - str or None: Command type
            - str or None: Action to take
            - str or None: Additional arguments
        """
        normalized_transcript = self._normalize_command_input(transcript)
        
        # Load available personas for context
        try:
            with open("personalities.json", 'r') as f:
                personas_data = json.load(f)
                available_personas = [p.lower() for p in personas_data.get("personas", {}).keys()]
        except Exception as e:
            print(f"Error loading personas: {e}")
            available_personas = []
        
        # Check for persona names in transcript
        persona_found = None
        for persona in available_personas:
            if persona in normalized_transcript.lower():
                persona_found = persona
                break
        
        # Process all command patterns
        for command_type, actions in self.command_patterns.items():
            for action, patterns in actions.items():
                for pattern in patterns:
                    if pattern in normalized_transcript:
                        # Extract arguments after pattern
                        start_idx = normalized_transcript.find(pattern) + len(pattern)
                        rest_of_text = normalized_transcript[start_idx:].strip()
                        
                        # Handle tool commands
                        if command_type == "tool":
                            return True, command_type, action, None
                        
                        # Special handling for persona commands
                        if command_type == "persona":
                            if persona_found:
                                return True, command_type, action, persona_found
                            return False, None, None, None
                        
                        # Return with arguments if present
                        return True, command_type, action, rest_of_text if rest_of_text else None
        
        return False, None, None, None

    async def handle_command(self, command_type: str, action: str, arguments: str = None) -> bool:
        """
        Execute system commands with proper state management and feedback.
        
        Args:
            command_type: Category of command (tool/document/calibration/etc.)
            action: Specific action to take
            arguments: Optional parameters for the command
            
        Returns:
            bool: Success status of command execution
        """
        try:
            if command_type == "tool":
                if action not in ["enable", "disable"]:
                    return False
                    
                success = await self._handle_tool_state_change(action)
                if success:
                    audio_folder = os.path.join(
                        f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/tool_sentences/status/{action}d"
                    )
                    if os.path.exists(audio_folder):
                        mp3_files = [f for f in os.listdir(audio_folder) if f.endswith('.mp3')]
                        if mp3_files:
                            audio_file = os.path.join(audio_folder, random.choice(mp3_files))
                            await self.audio_manager.queue_audio(audio_file=audio_file)
                return success
    
            elif command_type == "document":
                if action == "load":
                    success = await self.document_manager.load_all_files(clear_existing=False)
                    if success:
                        folder_path = os.path.join(f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/file_sentences/loaded")
                        if os.path.exists(folder_path):
                            mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
                            if mp3_files:
                                audio_file = os.path.join(folder_path, random.choice(mp3_files))
                                await self.audio_manager.queue_audio(audio_file=audio_file)
                    return success
                else:  # offload
                    await self.document_manager.offload_all_files()
                    folder_path = os.path.join(f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/file_sentences/offloaded")
                    if os.path.exists(folder_path):
                        mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
                        if mp3_files:
                            audio_file = os.path.join(folder_path, random.choice(mp3_files))
                            await self.audio_manager.queue_audio(audio_file=audio_file)
                    return True
    
            elif command_type == "calibration":
                success = await self._run_calibration()
                if success:
                    folder_path = os.path.join(f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/calibration_sentences")
                    if os.path.exists(folder_path):
                        mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
                        if mp3_files:
                            audio_file = os.path.join(folder_path, random.choice(mp3_files))
                            await self.audio_manager.queue_audio(audio_file=audio_file)
                return success
    
            elif command_type == "reminder":
                return await self._handle_reminder_command(action, arguments)
    
            elif command_type == "persona":
                return await self._handle_persona_command(action, arguments)
    
            return False
    
        except Exception as e:
            print(f"Command error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return False

    async def _handle_tool_state_change(self, state: str) -> bool:
        """
        Handle tool state transitions with proper display and audio sync.
        
        Args:
            state: Desired tool state ('enable' or 'disable')
            
        Returns:
            bool: Success of the state change
        """
        try:
            # Update token manager state
            if state == "enable":
                success = self.token_manager.enable_tools()
            else:
                success = self.token_manager.disable_tools()
                
            if success:
                # Use DisplayManager's path resolution
                await self.display_manager.update_display('tools', specific_image=f"{state}d")
                return True
            
            await self.display_manager.update_display('listening')
            return False
            
        except Exception as e:
            print(f"Error in tool state change: {e}")
            traceback.print_exc()
            await self.display_manager.update_display('listening')
            return False

    async def show_tool_use(self) -> None:
        """
        Update display and start audio for tool use feedback.
        Returns before audio completion to allow concurrent processing.
        """
        try:
            # Start audio playback (NON-BLOCKING)
            audio_task = None
            audio_folder = os.path.join(
                f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/tool_sentences/use"
            )
            if os.path.exists(audio_folder):
                mp3_files = [f for f in os.listdir(audio_folder) if f.endswith('.mp3')]
                if mp3_files:
                    audio_file = os.path.join(audio_folder, random.choice(mp3_files))
                    audio_task = asyncio.create_task(self.audio_manager.play_audio(audio_file))

            # Store for later checking
            if audio_task:
                self.current_tool_audio = audio_task

        except Exception as e:
            print(f"Error starting tool audio: {e}")
            traceback.print_exc()

    async def _run_calibration(self) -> bool:
        """
        Run voice calibration process.
        Updates display state and provides audio feedback.
        
        Returns:
            bool: Success status of calibration
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
        Process reminder-related commands with proper state management.
        
        Args:
            action: Reminder action (clear/add/update/list)
            arguments: Command parameters
            
        Returns:
            bool: Success status
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
                    # Show active reminders
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

    def _parse_schedule_days(self, day_input: str) -> list:
        """
        Parse day input into standardized schedule format.
        
        Args:
            day_input: Raw day specification
            
        Returns:
            list: Standardized day names
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
            days = [d.strip().lower()[:3] for d in day_input.split(',')]
            return [day_mapping[d] for d in days if d in day_mapping]

    async def execute_tool_with_feedback(self, tool_call, initial_response=None):
        """
        Execute a tool with coordinated audio and visual feedback.
        
        Args:
            tool_call: Tool execution data from Claude
            initial_response: Optional initial response text from Claude
            
        Returns:
            str: Processed response ready for TTS
        """
        try:
            # Start with thinking state
            await self.display_manager.update_display('thinking')
            
            # Show tool use state and play acknowledgment
            await self.display_manager.update_display('tools', specific_image='use')
            
            # Queue and play tool acknowledgment sound
            tool_sound = get_random_audio('tool', 'use')
            if tool_sound:
                await self.notification_manager.queue_notification(
                    text="Processing tool request",
                    priority=2,
                    sound_file=tool_sound
                )
                await self.notification_manager.process_pending_notifications()
            
            # Handle initial response if provided and not generic
            if initial_response and not any(phrase in initial_response.lower() 
                                          for phrase in ['let me check', 'one moment', 'just a second']):
                try:
                    initial_audio = self.tts_handler.generate_audio(str(initial_response))
                    with open("speech.mp3.initial", "wb") as f:
                        f.write(initial_audio)
                    await self.notification_manager.queue_notification(
                        text=initial_response,
                        priority=1,
                        sound_file="speech.mp3.initial"
                    )
                    await self.notification_manager.process_pending_notifications()
                except Exception as e:
                    print(f"Error handling initial response: {e}")
            
            # Execute the tool
            if not hasattr(tool_call, 'name') or not tool_call.name:
                raise ValueError("Invalid tool call - missing tool name")
                
            handler = tool_registry.get_handler(tool_call.name)
            if not handler:
                raise ValueError(f"Unsupported tool: {tool_call.name}")
                
            # Get tool arguments
            tool_args = {}
            if hasattr(tool_call, 'input'):
                tool_args = tool_call.input
            elif hasattr(tool_call, 'arguments'):
                try:
                    tool_args = json.loads(tool_call.arguments)
                except json.JSONDecodeError:
                    tool_args = {}
            
            # Execute tool with proper async handling
            if asyncio.iscoroutinefunction(handler):
                tool_result = await handler(**tool_args)
            else:
                tool_result = handler(**tool_args)
            
            # Record successful tool usage
            self.token_manager.record_tool_usage(tool_call.name)
            
            # Process the result
            if tool_result:
                processed_content = str(tool_result)
                
                # Generate and queue final audio
                final_audio = self.tts_handler.generate_audio(processed_content)
                with open("speech.mp3", "wb") as f:
                    f.write(final_audio)
                    
                # Update display and play response
                await self.display_manager.update_display('speaking', mood='casual')
                await self.notification_manager.queue_notification(
                    text=processed_content,
                    priority=1,
                    sound_file="speech.mp3"
                )
                await self.notification_manager.process_pending_notifications()
                
                # Return to listening state
                await self.display_manager.update_display('listening')
                return processed_content
            
            raise ValueError("Empty result from tool execution")
            
        except Exception as e:
            print(f"Error in tool execution: {e}")
            traceback.print_exc()
            
            # Ensure error state is properly displayed
            await self.display_manager.update_display('speaking', mood='disappointed')
            error_msg = f"Sorry, there was an error executing the tool: {str(e)}"
            
            try:
                error_audio = self.tts_handler.generate_audio(error_msg)
                with open("speech.mp3", "wb") as f:
                    f.write(error_audio)
                await self.notification_manager.queue_notification(
                    text=error_msg,
                    priority=1,
                    sound_file="speech.mp3"
                )
                await self.notification_manager.process_pending_notifications()
            except Exception as audio_err:
                print(f"Error generating error audio: {audio_err}")
                
            # Return to listening state after error
            await self.display_manager.update_display('listening')
            return error_msg

    async def _handle_persona_command(self, action: str, arguments: str = None) -> bool:
        """
        Handle persona switching with proper resource management.
        
        Args:
            action: Persona action (typically 'switch')
            arguments: Target persona name
            
        Returns:
            bool: Success status
        """
        try:
            print(f"\nDEBUG: Handling persona command - Action: {action}, Arguments: {arguments}")
            
            import config as config_module
            
            # Show current persona's exit animation
            current_persona = config_module.ACTIVE_PERSONA.lower()
            out_path = f"/home/user/LAURA/pygame/{current_persona}/system/persona/out"
            default_image = "/home/user/LAURA/pygame/laura/system/persona/dont_touch_this_image.png"
            
            # Display transition animation
            out_path_dir = Path(out_path)
            if out_path_dir.exists() and any(out_path_dir.glob('*.png')):
                await self.display_manager.update_display('system', transition_path=str(out_path_dir))
            else:
                default_dir = Path(default_image).parent
                await self.display_manager.update_display('system', transition_path=str(default_dir), 
                                                       specific_image=default_image)

            # Load personality configuration
            try:
                with open("personalities.json", 'r') as f:
                    personas_data = json.load(f)
            except FileNotFoundError:
                print("Creating default personalities configuration")
                personas_data = {
                    "personas": {
                        "laura": {
                            "voice": "L.A.U.R.A.",
                            "system_prompt": "You are Laura (Language & Automation User Response Agent), a professional and supportive AI-powered smart assistant."
                        }
                    },
                    "active_persona": "laura"
                }
                with open("personalities.json", 'w') as f:
                    json.dump(personas_data, f, indent=2)

            if action == "switch":
                if not arguments:
                    print("DEBUG: No persona specified")
                    return False
                    
                normalized_input = arguments.strip().lower()
                target_persona = None
                
                if normalized_input in personas_data.get("personas", {}):
                    target_persona = normalized_input
                else:
                    for key in personas_data.get("personas", {}).keys():
                        if key.lower() == normalized_input:
                            target_persona = key
                            break
                
                if target_persona:
                    print(f"DEBUG: Switching to persona: {target_persona}")
                    
                    try:
                        # Update configuration
                        personas_data["active_persona"] = target_persona
                        with open("personalities.json", 'w') as f:
                            json.dump(personas_data, f, indent=2)
                        
                        importlib.reload(config_module)
                        
                        config_module.ACTIVE_PERSONA = target_persona
                        config_module.ACTIVE_PERSONA_DATA = personas_data["personas"][target_persona]
                        config_module.VOICE = personas_data["personas"][target_persona].get("voice", "L.A.U.R.A.")
                        new_prompt = personas_data["personas"][target_persona].get("system_prompt", "You are an AI assistant.")
                        config_module.SYSTEM_PROMPT = f"{new_prompt}\n\n{config_module.UNIVERSAL_SYSTEM_PROMPT}"
                        
                        # Update TTS configuration
                        from secret import ELEVENLABS_KEY
                        new_config = {
                            "TTS_ENGINE": config_module.TTS_ENGINE,
                            "ELEVENLABS_KEY": ELEVENLABS_KEY,
                            "VOICE": config_module.VOICE,
                            "ELEVENLABS_MODEL": config_module.ELEVENLABS_MODEL,
                        }
                        self.tts_handler = TTSHandler(new_config)
                        
                        # Start introduction audio
                        audio_task = None
                        persona_audio_path = f"/home/user/LAURA/sounds/{target_persona}/persona_sentences"
                        if os.path.exists(persona_audio_path):
                            mp3_files = [f for f in os.listdir(persona_audio_path) if f.endswith('.mp3')]
                            if mp3_files:
                                audio_file = os.path.join(persona_audio_path, random.choice(mp3_files))
                                audio_task = asyncio.create_task(self.audio_manager.play_audio(audio_file))
                        
                        # Buffer before showing new persona
                        await asyncio.sleep(1.0)
                        
                        # Update display path
                        new_base_path = str(Path(f'/home/user/LAURA/pygame/{target_persona.lower()}'))
                        await self.display_manager.update_display_path(new_base_path)
                        
                        # Show entrance animation
                        in_path = f"/home/user/LAURA/pygame/{target_persona.lower()}/system/persona/in"
                        in_path_dir = Path(in_path)
                        
                        if in_path_dir.exists() and any(in_path_dir.glob('*.png')):
                            await self.display_manager.update_display('system', transition_path=str(in_path_dir))
                        else:
                            default_dir = Path(default_image).parent
                            await self.display_manager.update_display('system', transition_path=str(default_dir), 
                                                                   specific_image=default_image)
                        
                        # Wait for audio completion
                        if audio_task:
                            try:
                                await audio_task
                            except Exception as e:
                                print(f"Warning: Error waiting for audio completion: {e}")
                        
                        # Return to listening state
                        await self.display_manager.update_display('listening')
                        print("DEBUG: Persona switch complete")
                        return True
                        
                    except Exception as e:
                        print(f"ERROR: Failed to update configuration: {e}")
                        traceback.print_exc()
                        return False
                print(f"ERROR: Persona '{arguments}' not found")
                return False
                
            return False
            
        except Exception as e:
            print(f"Error in persona command: {e}")
            traceback.print_exc()
            return False
