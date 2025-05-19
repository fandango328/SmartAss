import asyncio
import json
import uuid
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum
from pathlib import Path

# Import directly from main_loop.py
from main_loop import process_input as main_loop_process_input
from main_loop import conversations as main_loop_conversations


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
    Directly calls the main_loop.py logic.
    """
    def __init__(
        self,
        response_formatter: Any, # Instance of ResponseHandler
        document_manager=None,
        chat_log_dir: str = "logs"
        # main_loop_handler is no longer a parameter
    ):
        self.response_formatter = response_formatter
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
        """Start the orchestrator processing loop for queued inputs."""
        self.running = True
        print(f"Orchestrator started at {self.session_start.isoformat()} for queued inputs.")
        
        while self.running:
            try:
                input_event = await self.input_queue.get()
                # If processing queued events, the response might not be directly returned
                # to an external caller of _process_input, but handled internally or broadcast.
                # For now, it will process and log.
                await self._process_input(input_event) 
                self.input_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error processing queued input: {e}")
                self.log_event("queued_processing_error", {
                    "error": str(e),
                    "input_event": input_event, 
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
            "connected_at": session.connected_at.isoformat(),
            "capabilities": capabilities 
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
        Add input to processing queue for asynchronous handling via the start() loop.
        """
        session_id = input_event.get("session_id")
        if session_id not in self.device_sessions:
            print(f"Warning: Input received with invalid session_id: {session_id}")
            self.log_event("invalid_session_input", {"session_id": session_id, "input_event": input_event})
            return False

        session = self.device_sessions[session_id]
        session.update_activity()

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
                return True 

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

    async def _process_input(self, input_event_from_server: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an input event directly, call main_loop.py logic, 
        and then use response_formatter to return the final formatted response.
        """
        try:
            session_id = input_event_from_server.get("session_id")
            if session_id not in self.device_sessions:
                self.log_event("processing_error_invalid_session", {"session_id": session_id, "input_event": input_event_from_server})
                return {"error": f"Invalid session ID: {session_id}", "text": "", "mood": "error", "session_id": session_id, "active_persona": "unknown"}

            session = self.device_sessions[session_id]
            session.update_activity() 

            active_persona = session.get_persona()
            session_capabilities = session.capabilities
            user_text = input_event_from_server.get("text", "")
            input_type = input_event_from_server.get("type", "text")


            self.log_event("direct_input_received_for_processing", {
                "device_id": session.device_id,
                "session_id": session_id,
                "type": input_type,
                "persona": active_persona,
                "timestamp": datetime.now().isoformat()
            })

            print(f"[Orchestrator] Processing {input_type} input from device {session.device_id} (persona: {active_persona})")
            
            # --- Logic from former actual_llm_main_loop_handler integrated here ---
            event_for_main_loop = {
                "session_id": session_id,
                "type": input_type,
                "payload": {"text": user_text}
                # "active_persona" could be passed if main_loop.process_input uses it
            }
            
            # Call the LLM processing logic from main_loop.py
            llm_response_dict = await main_loop_process_input(event_for_main_loop)
            
            raw_assistant_text = llm_response_dict.get("text", "")
            # Fetch the updated chat history for this session from main_loop's state
            chat_log_for_response_handler = main_loop_conversations.get(session_id, [])
            # --- End of integrated logic ---

            self.log_event("main_loop_handler_completed", {
                "device_id": session.device_id,
                "session_id": session_id,
                "type": input_type,
                "raw_text_length": len(raw_assistant_text),
                "chat_log_length": len(chat_log_for_response_handler),
                "timestamp": datetime.now().isoformat()
            })

            final_payload = await self.response_formatter.handle_response(
                assistant_content=raw_assistant_text,
                chat_log=chat_log_for_response_handler,
                session_capabilities=session_capabilities,
                session_id=session_id,
                active_persona=active_persona
            )
            
            self.log_event("response_formatter_completed", {
                "device_id": session.device_id,
                "session_id": session_id,
                "final_payload_text": final_payload.get("text", "")[:100], 
                "timestamp": datetime.now().isoformat()
            })
            
            return final_payload

        except Exception as e:
            print(f"Error in _process_input: {e}")
            self.log_event("direct_processing_error", {
                "error": str(e),
                "input_event": input_event_from_server, 
                "timestamp": datetime.now().isoformat()
            })
            active_persona_on_error = "unknown"
            if 'session_id' in input_event_from_server and input_event_from_server['session_id'] in self.device_sessions:
                active_persona_on_error = self.device_sessions[input_event_from_server['session_id']].get_persona()
            
            return {
                "error": f"Error processing input: {str(e)}", 
                "text": f"An error occurred: {str(e)}", 
                "mood": "error", 
                "session_id": input_event_from_server.get("session_id", "unknown"), 
                "active_persona": active_persona_on_error
            }

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
    
    def get_device_info(self, session_id: str = None) -> Optional[Dict[str, Any]]:
        """Get information about a specific connected device or all devices."""
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
                    "document_count": len(session.active_documents),
                    "capabilities": session.capabilities 
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
