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
from tool_registry import ToolRegistry, tool_registry
import config
from secret import ELEVENLABS_KEY

from core_functions import (
    run_vad_calibration,
    handle_calendar_query,
    process_response_content,
    validate_llm_response,
    handle_tool_sequence,
    get_random_audio
)

from function_definitions import (
    get_random_audio,
    get_current_time,
    get_location,
    create_calendar_event,
    update_calendar_event,
    cancel_calendar_event,
    manage_tasks,
    create_task_from_email,
    create_task_for_event
)

from email_manager import EmailManager
        
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
    def __init__(self, email_manager=None, display_manager=None, audio_manager=None,
                 document_manager=None, notification_manager=None, token_manager=None,
                 tts_handler=None, anthropic_client=None):
        """
        Initialize SystemManager with all required dependencies.
        
        Args:
            email_manager: EmailManager instance for handling email operations
            display_manager: DisplayManager for UI updates
            audio_manager: AudioManager for sound handling
            document_manager: DocumentManager for file operations
            notification_manager: NotificationManager for system notifications
            token_manager: TokenManager for API token management
            tts_handler: TTSHandler for text-to-speech
            anthropic_client: Anthropic client for AI operations
        """
        print("\nInitializing SystemManager with provided managers:")
        
        # Store manager instances
        self.email_manager = email_manager
        print(f"- Email Manager: {'Present' if email_manager else 'Missing'}")
        
        self.display_manager = display_manager
        print(f"- Display Manager: {'Present' if display_manager else 'Missing'}")
        
        self.audio_manager = audio_manager
        print(f"- Audio Manager: {'Present' if audio_manager else 'Missing'}")
        
        self.document_manager = document_manager
        print(f"- Document Manager: {'Present' if document_manager else 'Missing'}")
        
        self.notification_manager = notification_manager
        print(f"- Notification Manager: {'Present' if notification_manager else 'Missing'}")
        
        self.token_manager = token_manager
        print(f"- Token Manager: {'Present' if token_manager else 'Missing'}")
        
        self.tts_handler = tts_handler
        print(f"- TTS Handler: {'Present' if tts_handler else 'Missing'}")
        
        self.anthropic_client = anthropic_client
        print(f"- Anthropic Client: {'Present' if anthropic_client else 'Missing'}")
        
        # Initialize tracking dictionary with actual manager states
        self._initialized_managers = {
            'email': bool(email_manager),
            'display': bool(display_manager),
            'audio': bool(audio_manager),
            'document': bool(document_manager),
            'notification': bool(notification_manager),
            'token': bool(token_manager),
            'tts': bool(tts_handler),
            'anthropic': bool(anthropic_client)
        }
        
        # Print initialization summary
        print("\nRequired manager status:")
        required = ['token', 'display', 'audio', 'notification']
        for manager in required:
            status = self._initialized_managers.get(manager, False)
            print(f"- {manager}: {'✓ Initialized' if status else '✗ Not initialized'}")
        
        self.debug_detection = False
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

    def _update_initialization_status(self):
        """Update the initialization status of all managers."""
        self._initialized_managers.update({
            'email': self.email_manager is not None,
            'display': self.display_manager is not None,
            'audio': self.audio_manager is not None,
            'document': self.document_manager is not None,
            'notification': self.notification_manager is not None,
            'token': self.token_manager is not None,
            'tts': self.tts_handler is not None,
            'anthropic': self.anthropic_client is not None
        })
                
    async def _initialize_clients(self):
        """Initialize TTS and Anthropic clients with configuration."""
        try:
            if not self.tts_handler:
                self.tts_handler = TTSHandler({
                    "TTS_ENGINE": config.TTS_ENGINE,
                    "ELEVENLABS_KEY": ELEVENLABS_KEY,
                    "VOICE": config.VOICE,
                    "ELEVENLABS_MODEL": config.ELEVENLABS_MODEL,
                })
            self._initialized_managers['tts'] = bool(self.tts_handler)
            
            if not self.anthropic_client:
                from secret import ANTHROPIC_API_KEY
                from anthropic import Anthropic
                self.anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
            self._initialized_managers['anthropic'] = bool(self.anthropic_client)
                
        except Exception as e:
            print(f"Error initializing clients: {e}")
            traceback.print_exc()
            
    async def register_managers(self):
        """Register all available managers with the tool registry and confirm initialization."""
        try:
            print("\nStarting manager registration process...")
            
            # Debug output for initial manager state
            print("\nInitial manager states:")
            for manager_name in ['display', 'audio', 'document', 'notification', 'token', 'tts', 'anthropic']:
                manager = getattr(self, f"{manager_name}_manager", None)
                print(f"- {manager_name}_manager: {'Present' if manager else 'Missing'}")
            
            # Register and track each manager
            if self.display_manager:
                print("\nRegistering display manager...")
                tool_registry.register_manager('display', self.display_manager)
                self._initialized_managers['display'] = True
                print("✓ Display manager registered")
                
            if self.audio_manager:
                print("\nRegistering audio manager...")
                tool_registry.register_manager('audio', self.audio_manager)
                self._initialized_managers['audio'] = True
                print("✓ Audio manager registered")
                
            if self.document_manager:
                print("\nRegistering document manager...")
                tool_registry.register_manager('document', self.document_manager)
                self._initialized_managers['document'] = True
                print("✓ Document manager registered")
                
            if self.notification_manager:
                print("\nRegistering notification manager...")
                tool_registry.register_manager('notification', self.notification_manager)
                self._initialized_managers['notification'] = True
                print("✓ Notification manager registered")
                
            if self.token_manager:
                print("\nRegistering token manager...")
                tool_registry.register_manager('token', self.token_manager)
                if hasattr(self.token_manager, 'set_system_manager'):
                    self.token_manager.set_system_manager(self)
                self._initialized_managers['token'] = True
                print("✓ Token manager registered")
            
            # Register system manager before initializing clients
            print("\nRegistering system manager...")
            tool_registry.register_manager('system', self)
            print("✓ System manager registered")
            
            # Initialize and register clients
            print("\nInitializing clients...")
            await self._initialize_clients()
            
            # Register tool handlers after system manager is registered
            print("\nRegistering tool handlers...")
            success = await self._register_tool_handlers()
            if not success:
                raise RuntimeError("Tool handler registration failed")
            print("✓ Tool handlers registered")
            
            # Final verification and status report
            print("\nFinal Initialization Status:")
            required_managers = ['token', 'display', 'audio', 'notification']
            all_initialized = True
            
            for manager in required_managers:
                status = self._initialized_managers.get(manager, False)
                print(f"- {manager}_manager: {'✓ Initialized' if status else '✗ Not initialized'}")
                if not status:
                    all_initialized = False
            
            if not all_initialized:
                missing = [m for m in required_managers if not self._initialized_managers.get(m, False)]
                print(f"\n⚠️  Warning: Required managers not initialized: {', '.join(missing)}")
            else:
                print("\n✅ All required managers successfully initialized")
                
        except Exception as e:
            print(f"\n❌ Error during manager registration: {e}")
            traceback.print_exc()
            raise  # Re-raise the exception to ensure proper error handling

    async def _register_tool_handlers(self):
        """Register core tool handlers with proper dependency injection."""
        try:
            # Phase 1: Register all managers first
            if not tool_registry.get_manager('email'):
                from email_manager import EmailManager
                email_manager = self.email_manager  # Use existing email manager
                if email_manager:
                    tool_registry.register_manager('email', email_manager)
                else:
                    print("Warning: No email manager available")

            # Phase 2: Register basic tool handlers
            from core_functions import execute_calendar_query
            from function_definitions import (
                get_current_time, get_location, create_calendar_event,
                update_calendar_event, cancel_calendar_event, manage_tasks,
                create_task_from_email, create_task_for_event
            )

            basic_handlers = {
                "get_current_time": get_current_time,
                "get_location": get_location,
                "calendar_query": lambda **kwargs: handle_calendar_query(
                    self.email_manager.service if self.email_manager else None,
                    kwargs.get("query_type"),
                    **{k:v for k,v in kwargs.items() if k != "query_type"}
                ),
                "create_calendar_event": create_calendar_event,
                "update_calendar_event": update_calendar_event,
                "cancel_calendar_event": cancel_calendar_event,
                "manage_tasks": manage_tasks,
                "create_task_from_email": create_task_from_email,
                "create_task_for_event": create_task_for_event
            }
            
            # Register basic handlers
            tool_registry.register_handlers(basic_handlers)
            
            # Phase 3: Register email-dependent handlers
            email_manager = tool_registry.get_manager('email')
            if email_manager:
                email_handlers = {
                    "draft_email": email_manager.draft_email,
                    "read_emails": email_manager.read_emails,
                    "email_action": email_manager.email_action
                }
                tool_registry.register_handlers(email_handlers)
            else:
                print("Warning: Email manager not available, email-related tools will be disabled")
                
            tool_registry.initialize()  # Removed await since this isn't an async method
            print("Tool handlers registered successfully")
            return True
            
        except Exception as e:
            print(f"Error registering tool handlers: {e}")
            traceback.print_exc()
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

    def is_initialized(self) -> tuple[bool, list[str]]:
        """
        Check if the system manager is fully initialized with required components.
        
        Returns:
            tuple: (bool: initialization status, list: missing manager names)
        """
        required_managers = {
            'token_manager': self.token_manager,
            'display_manager': self.display_manager,
            'audio_manager': self.audio_manager,
            'notification_manager': self.notification_manager
        }
        
        missing = [name for name, manager in required_managers.items() 
                  if manager is None]
        
        return (len(missing) == 0, missing)

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
        # Enhanced initialization check with specific feedback
        is_init, missing_managers = self.is_initialized()
        if not is_init:
            print("Warning: System Manager not fully initialized")
            print(f"Missing required managers: {', '.join(missing_managers)}")
            return False
            
        try:
            print(f"\nExecuting command: {command_type} - {action}")
            
            if command_type == "tool":
                if action not in ["enable", "disable"]:
                    print(f"Invalid tool action: {action}")
                    return False
                    
                success = await self._handle_tool_state_change(action)
                return success

            elif command_type == "document":
                if action == "load":
                    success = await self.document_manager.load_all_files(clear_existing=False)
                    if success and self.audio_manager:
                        folder_path = os.path.join(f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/file_sentences/loaded")
                        if os.path.exists(folder_path):
                            mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
                            if mp3_files:
                                audio_file = os.path.join(folder_path, random.choice(mp3_files))
                                await self.audio_manager.queue_audio(audio_file=audio_file)
                    return success
                else:  # offload
                    await self.document_manager.offload_all_files()
                    if self.audio_manager:
                        folder_path = os.path.join(f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/file_sentences/offloaded")
                        if os.path.exists(folder_path):
                            mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
                            if mp3_files:
                                audio_file = os.path.join(folder_path, random.choice(mp3_files))
                                await self.audio_manager.queue_audio(audio_file=audio_file)
                    return True

            elif command_type == "calibration":
                success = await self._run_calibration()
                if success and self.audio_manager:
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

            print(f"Unknown command type: {command_type}")
            return False

        except Exception as e:
            print(f"Command error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return False

    def _verify_required_managers(self, *manager_names) -> bool:
        """
        Verify that required managers are initialized.
        
        Args:
            *manager_names: Variable list of manager names to check
            
        Returns:
            bool: True if all required managers are initialized
            
        Raises:
            RuntimeError: If any required manager is not initialized
        """
        missing_managers = []
        for manager_name in manager_names:
            manager = getattr(self, f"{manager_name}_manager", None)
            if manager is None:
                missing_managers.append(manager_name)
        
        if missing_managers:
            raise RuntimeError(
                f"Required managers not initialized: {', '.join(missing_managers)}"
            )
        return True

    async def _handle_tool_state_change(self, state: str) -> bool:
        """
        Handle tool state transitions with proper display and audio sync.
        
        Args:
            state: Desired tool state ('enable' or 'disable')
            
        Returns:
            bool: Success of the state change
        """
        try:
            # Verify required managers are initialized
            self._verify_required_managers('token', 'display')
            
            # Update token manager state
            if state == "enable":
                success = self.token_manager.enable_tools()
            else:
                success = self.token_manager.disable_tools()
                
            if success:
                # Use DisplayManager's path resolution
                await self.display_manager.update_display('tools', specific_image=f"{state}d")
                
                # Play appropriate audio feedback if audio manager is available
                if self.audio_manager:
                    audio_folder = os.path.join(
                        f"/home/user/LAURA/sounds/{config.ACTIVE_PERSONA.lower()}/tool_sentences/status/{state}d"
                    )
                    if os.path.exists(audio_folder):
                        mp3_files = [f for f in os.listdir(audio_folder) if f.endswith('.mp3')]
                        if mp3_files:
                            audio_file = os.path.join(audio_folder, random.choice(mp3_files))
                            await self.audio_manager.queue_audio(audio_file=audio_file)
                return True
            
            await self.display_manager.update_display('listening')
            return False
            
        except Exception as e:
            print(f"Error in tool state change: {e}")
            traceback.print_exc()
            
            # Try to return to listening state if display manager is available
            if self.display_manager:
                try:
                    await self.display_manager.update_display('listening')
                except Exception as display_error:
                    print(f"Error updating display: {display_error}")
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
        Execute a tool and manage feedback for the first part of a two-stage API interaction.
        This method handles the sequence between first and final API calls.
        
        Flow:
        1. Log assistant acknowledgment (from API or default)
        2. Update display for tool state
        3. Execute tool and get result
        4. Queue pre-recorded notification
        5. Return result for final API call in generate_response
        
        Args:
            tool_call: Tool execution data from first API call
            initial_response: Optional acknowledgment from API
            
        Returns:
            dict: Contains tool result and metadata for final API call
            {
                'tool_id': str,            # ID of executed tool
                'result': str,             # Tool execution result
                'acknowledgment': str,     # What assistant said before execution
            }
        """
        try:
            # STEP 1: Log assistant acknowledgment
            acknowledgment = initial_response if initial_response else "I'll help you with that right away."
            chat_log.append({
                "role": "assistant",
                "content": acknowledgment
            })
            save_to_log_file({
                "role": "assistant",
                "content": acknowledgment
            })
            
            # STEP 2: Show tool execution state
            await self.display_manager.update_display('tools', specific_image='use')
            
            # STEP 3: Execute tool
            if not hasattr(tool_call, 'name') or not tool_call.name:
                raise ValueError("Invalid tool call - missing tool name")
                
            handler = tool_registry.get_handler(tool_call.name)
            if not handler:
                raise ValueError(f"Unsupported tool: {tool_call.name}")
                
            # Execute tool directly with input if available
            tool_args = getattr(tool_call, 'input', {}) or {}
            
            # Execute with proper async handling
            if asyncio.iscoroutinefunction(handler):
                tool_result = await handler(**tool_args)
            else:
                tool_result = handler(**tool_args)
                
            if not tool_result:
                raise ValueError("Empty result from tool execution")
                
            # Record successful usage
            self.token_manager.record_tool_usage(tool_call.name)
            
            # STEP 4: Queue pre-recorded notification
            tool_sound = get_random_audio('tool', 'processing')
            if tool_sound:
                await self.notification_manager.queue_notification(
                    text="Processing tool request",
                    priority=2,
                    sound_file=tool_sound
                )
                await self.notification_manager.process_pending_notifications()
            
            # STEP 5: Return packaged result for final API call
            return {
                'tool_id': tool_call.id,
                'result': str(tool_result),
                'acknowledgment': acknowledgment
            }
            
        except Exception as e:
            print(f"Error in tool execution: {e}")
            traceback.print_exc()
            
            error_msg = f"I encountered an error while processing your request: {str(e)}"
            chat_log.append({
                "role": "assistant",
                "content": error_msg
            })
            save_to_log_file({
                "role": "assistant",
                "content": error_msg
            })
            
            return {
                'tool_id': getattr(tool_call, 'id', 'error'),
                'result': error_msg,
                'acknowledgment': error_msg
            }
            
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
