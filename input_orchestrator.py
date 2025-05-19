import asyncio
import json
import uuid
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum
from pathlib import Path

class InputType(Enum):
    TEXT = "text"
    VOICE = "voice"
    DOCUMENT = "document"

class OutputMode(Enum):
    TEXT = "text"
    AUDIO = "audio"
    BOTH = "both"

class DeviceSession:
    """Tracks device connection and capabilities, including active persona."""
    def __init__(self, device_id: str, capabilities: Dict[str, Any]):
        self.device_id = device_id
        self.session_id = str(uuid.uuid4())
        self.capabilities = capabilities
        self.connected_at = datetime.now()
        self.last_activity = datetime.now()
        self.active_documents = []
        self.active_persona = "laura"  # default persona
        
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
        
    def add_document(self, document_data):
        """Add a document to the active documents list"""
        self.active_documents.append(document_data)
        
    def clear_documents(self):
        """Clear all active documents"""
        self.active_documents = []
        
    def get_documents(self):
        """Get all active documents"""
        return self.active_documents

    def set_persona(self, persona: str):
        """Set the active persona for this session"""
        if persona:
            self.active_persona = persona

    def get_persona(self) -> str:
        """Get the current active persona"""
        return self.active_persona

class InputOrchestrator:
    """
    Orchestrates input from multiple devices to a single conversation context.
    Routes and normalizes different input types for processing by the main loop.
    
    - Handles device registration and capabilities tracking
    - Routes input to appropriate processors
    - Manages a single input queue for FIFO processing 
    - Keeps documents separate from conversation history
    - Logs session events for troubleshooting
    - Tracks active persona per session
    """
    def __init__(
        self,
        main_loop_handler: Callable[[Dict[str, Any]], Any],
        document_manager=None,
        chat_log_dir: str = "logs"
    ):
        self.main_loop_handler = main_loop_handler
        self.document_manager = document_manager
        self.chat_log_dir = Path(chat_log_dir)
        self.input_queue = asyncio.Queue()
        self.device_sessions: Dict[str, DeviceSession] = {}
        self.running = False
        self.chat_log_dir.mkdir(parents=True, exist_ok=True)
        self.session_start = datetime.now()
        self.log_event("orchestrator_start", {
            "timestamp": self.session_start.isoformat()
        })

    async def start(self):
        """Start the orchestrator processing loop"""
        self.running = True
        print(f"Orchestrator started at {self.session_start.isoformat()}")
        
        while self.running:
            try:
                input_event = await self.input_queue.get()
                await self._process_input(input_event)
                self.input_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error processing input: {e}")
                self.log_event("processing_error", {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                await asyncio.sleep(0.1)

    async def register_device(self, device_id: str, capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new device with capability information."""
        session = DeviceSession(device_id, capabilities)
        self.device_sessions[session.session_id] = session
        self.log_event("device_registered", {
            "device_id": device_id,
            "session_id": session.session_id,
            "capabilities": capabilities,
            "timestamp": session.connected_at.isoformat()
        })
        return {
            "session_id": session.session_id,
            "status": "connected",
            "connected_at": session.connected_at.isoformat()
        }

    async def unregister_device(self, session_id: str) -> bool:
        """Unregister a device when it disconnects"""
        if session_id in self.device_sessions:
            device_id = self.device_sessions[session_id].device_id
            self.log_event("device_unregistered", {
                "device_id": device_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })
            del self.device_sessions[session_id]
            return True
        return False

    async def add_input(self, input_event: Dict[str, Any]) -> bool:
        """
        Add input to processing queue.
        Validates session and updates activity timestamp.
        Handles persona change commands if present.
        """
        session_id = input_event.get("session_id")
        if session_id not in self.device_sessions:
            print(f"Warning: Input received with invalid session_id: {session_id}")
            return False

        session = self.device_sessions[session_id]
        session.update_activity()

        # Handle persona change commands
        if input_event.get("command") == "change_persona":
            persona = input_event.get("persona")
            if persona:
                session.set_persona(persona)
                self.log_event("persona_changed", {
                    "device_id": session.device_id,
                    "session_id": session_id,
                    "new_persona": persona,
                    "timestamp": datetime.now().isoformat()
                })
                # No need to queue further input for this event
                return True

        # Special handling for documents
        if input_event.get("type") == "document":
            document_data = input_event.get("files", [])
            if document_data:
                session.add_document(document_data)
                if self.document_manager and hasattr(self.document_manager, "load_all_files"):
                    await self.document_manager.load_all_files()
                self.log_event("document_stored", {
                    "device_id": session.device_id,
                    "session_id": session_id,
                    "document_count": len(document_data),
                    "timestamp": datetime.now().isoformat()
                })
                return True

        await self.input_queue.put(input_event)
        return True
        
    async def clear_documents(self, session_id: str) -> bool:
        """Clear all documents for a session"""
        if session_id in self.device_sessions:
            self.device_sessions[session_id].clear_documents()
            if self.document_manager and hasattr(self.document_manager, "offload_all_files"):
                await self.document_manager.offload_all_files()
            self.log_event("documents_cleared", {
                "device_id": self.device_sessions[session_id].device_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })
            return True
        return False

    async def _process_input(self, input_event: Dict[str, Any]):
        """
        Process and route input to the main loop handler.
        Passes the current active persona for the session in the input event.
        """
        try:
            session_id = input_event.get("session_id")
            if session_id not in self.device_sessions:
                raise ValueError(f"Invalid session ID: {session_id}")
            session = self.device_sessions[session_id]

            # Determine output mode from explicit setting or device capabilities
            if "output_mode" not in input_event:
                supported_outputs = session.capabilities.get("output", ["text"])
                input_event["output_mode"] = supported_outputs[0]

            input_type = input_event.get("type", "unknown")

            self.log_event("input_received", {
                "device_id": session.device_id,
                "session_id": session_id,
                "type": input_type,
                "timestamp": datetime.now().isoformat()
            })

            # Add current persona to input_event for downstream handler
            input_event["active_persona"] = session.get_persona()

            print(f"Processing {input_type} input from device {session.device_id} (persona: {session.get_persona()})")
            await self.main_loop_handler(input_event)

            self.log_event("input_processed", {
                "device_id": session.device_id,
                "session_id": session_id,
                "type": input_type,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            print(f"Error processing input: {e}")
            self.log_event("input_error", {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log orchestrator events to JSON file."""
        try:
            log_entry = {
                "event_type": event_type,
                **data
            }
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = self.chat_log_dir / f"orchestrator_{today}.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Error logging event: {e}")
    
    def get_device_info(self, session_id: str = None) -> Dict[str, Any]:
        """Get information about connected devices"""
        if session_id:
            if session_id in self.device_sessions:
                session = self.device_sessions[session_id]
                return {
                    "device_id": session.device_id,
                    "session_id": session_id,
                    "connected_at": session.connected_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "capabilities": session.capabilities,
                    "active_persona": session.get_persona(),
                    "document_count": len(session.active_documents)
                }
            return None

        return {
            "active_devices": len(self.device_sessions),
            "devices": [
                {
                    "device_id": session.device_id,
                    "session_id": sid,
                    "connected_at": session.connected_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "active_persona": session.get_persona(),
                    "document_count": len(session.active_documents)
                }
                for sid, session in self.device_sessions.items()
            ]
        }
        
    def stop(self):
        """Stop the orchestrator processing loop"""
        self.running = False
        self.log_event("orchestrator_stop", {
            "timestamp": datetime.now().isoformat()
        })
        print("Orchestrator stopped")
