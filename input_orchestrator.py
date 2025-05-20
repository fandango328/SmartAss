import asyncio
import json
import uuid
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum
from pathlib import Path

# Import the MainLoop CLASS from main_loop.py
from main_loop import MainLoop

# Import your TokenManager and DocumentManager.
from token_manager import TokenManager
from document_manager import DocumentManager

# ADDED: For Anthropic client
import anthropic
from secret import ANTHROPIC_API_KEY # Ensure ANTHROPIC_API_KEY is in your secret.py


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
        self.last_activity = datetime.now()
        
    def add_document(self, document_data):
        self.active_documents.append(document_data)
        
    def clear_documents(self):
        self.active_documents = []
        
    def get_documents(self):
        return self.active_documents

    def set_persona(self, persona: str):
        if persona:
            self.active_persona = persona

    def get_persona(self) -> str:
        return self.active_persona

class InputOrchestrator:
    def __init__(
        self,
        response_formatter: Any, 
        document_manager_param: Optional[DocumentManager] = None,
        chat_log_dir: str = "logs"
    ):
        self.response_formatter = response_formatter
        self.chat_log_dir = Path(chat_log_dir)
        self.input_queue = asyncio.Queue()
        self.device_sessions: Dict[str, DeviceSession] = {}
        self.running = False
        self.chat_log_dir.mkdir(parents=True, exist_ok=True)
        self.session_start = datetime.now()
        self.log_event("orchestrator_start", {
            "timestamp": self.session_start.isoformat()
        })

        # --- Initialize Anthropic Client, Managers and MainLoop Instance ---
        print("[Orchestrator __init__] Creating Anthropic client...")
        self.anthropic_client_instance = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        print("[Orchestrator __init__] Anthropic client created.")

        print("[Orchestrator __init__] Initializing TokenManager for MainLoop...")
        self.token_manager_for_main_loop = TokenManager(anthropic_client=self.anthropic_client_instance)
        print("[Orchestrator __init__] TokenManager initialized.")

        print("[Orchestrator __init__] Initializing DocumentManager for MainLoop...")
        if document_manager_param:
            self.document_manager_for_main_loop = document_manager_param
            print("[Orchestrator __init__] Using provided DocumentManager instance.")
        else:
            self.document_manager_for_main_loop = DocumentManager() # Assumes DM needs no args or gets them from config
            print("[Orchestrator __init__] Created new DocumentManager instance.")
        
        print("[Orchestrator __init__] TokenManager and DocumentManager ready for MainLoop.")

        print("[Orchestrator __init__] Initializing MainLoop instance...")
        # If MainLoop's AnthropicLLMAdapter also needs this specific client instance,
        # MainLoop's __init__ would need to be adapted to accept it and pass it down.
        # For now, MainLoop's adapter will create its own client or use API key directly.
        self.main_loop_processor = MainLoop(
            token_manager_instance=self.token_manager_for_main_loop,
            document_manager_instance=self.document_manager_for_main_loop
            # anthropic_client=self.anthropic_client_instance # Pass if MainLoop is adapted
        )
        print("[Orchestrator __init__] MainLoop instance is ready.")
        # --- END OF INITIALIZATION ---


    async def start(self):
        self.running = True
        print(f"Orchestrator started at {self.session_start.isoformat()} for queued inputs.")
        
        while self.running:
            try:
                input_event = await self.input_queue.get()
                await self._process_input(input_event) 
                self.input_queue.task_done()
            except asyncio.CancelledError:
                print("Orchestrator processing loop cancelled.")
                break
            except Exception as e:
                print(f"Error processing queued input: {e}")
                self.log_event("queued_processing_error", {
                    "error": str(e),
                    "input_event": str(input_event), 
                    "timestamp": datetime.now().isoformat()
                })
                await asyncio.sleep(0.1)

    async def register_device(self, device_id: str, capabilities: Dict[str, Any]) -> Dict[str, Any]:
        session = DeviceSession(device_id, capabilities)
        self.device_sessions[session.session_id] = session
        self.log_event("device_registered", {
            "device_id": device_id,
            "session_id": session.session_id,
            "capabilities": capabilities,
            "timestamp": session.connected_at.isoformat()
        })
        print(f"[Orchestrator register_device] Device {device_id} registered with session ID {session.session_id}")
        return {
            "session_id": session.session_id,
            "status": "connected",
            "connected_at": session.connected_at.isoformat(),
            "capabilities": capabilities 
        }

    async def unregister_device(self, session_id: str) -> bool:
        if session_id in self.device_sessions:
            device_id = self.device_sessions[session_id].device_id
            self.log_event("device_unregistered", {
                "device_id": device_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })
            del self.device_sessions[session_id]
            print(f"[Orchestrator unregister_device] Device session {session_id} (Device ID: {device_id}) unregistered.")
            return True
        print(f"[Orchestrator unregister_device] Attempted to unregister unknown session ID: {session_id}")
        return False

    async def add_input(self, input_event: Dict[str, Any]) -> bool:
        session_id = input_event.get("session_id")
        if not session_id or session_id not in self.device_sessions:
            print(f"[Orchestrator add_input] Warning: Input received with invalid or missing session_id: {session_id}")
            self.log_event("invalid_session_input", {"session_id": session_id, "input_event": input_event})
            return False

        session = self.device_sessions[session_id]
        session.update_activity()
        print(f"[Orchestrator add_input] Received input for session {session_id}: {input_event.get('type')}")

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
                print(f"[Orchestrator add_input] Persona changed to '{persona}' for session {session_id}")
                return True 

        if input_event.get("type") == "document":
            document_data = input_event.get("files", []) 
            if document_data:
                session.add_document(document_data) 
                print(f"[Orchestrator add_input] Document data received for session {session_id}. MainLoop's DocumentManager should handle it.")
                self.log_event("document_received_for_processing", {
                    "device_id": session.device_id,
                    "session_id": session_id,
                    "document_info_count": len(document_data),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                print(f"[Orchestrator add_input] Document event received for session {session_id} but no file data found.")
                return False

        await self.input_queue.put(input_event)
        print(f"[Orchestrator add_input] Input event for session {session_id} added to processing queue.")
        return True
        
    async def clear_documents(self, session_id: str) -> bool:
        if session_id in self.device_sessions:
            self.device_sessions[session_id].clear_documents() 
            
            if self.document_manager_for_main_loop and hasattr(self.document_manager_for_main_loop, "offload_all_files"):
                try:
                    await self.document_manager_for_main_loop.offload_all_files() 
                    print(f"[Orchestrator clear_documents] Called offload_all_files on MainLoop's DocumentManager for session {session_id}.")
                except Exception as e:
                    print(f"[Orchestrator clear_documents] Error calling offload_all_files: {e}")

            self.log_event("documents_cleared", {
                "device_id": self.device_sessions[session_id].device_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })
            return True
        print(f"[Orchestrator clear_documents] Attempted to clear documents for unknown session ID: {session_id}")
        return False

    async def _process_input(self, input_event_from_queue: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        session_id = input_event_from_queue.get("session_id")
        try:
            if not session_id or session_id not in self.device_sessions:
                self.log_event("processing_error_invalid_session", {"session_id": session_id, "input_event": input_event_from_queue})
                print(f"[Orchestrator _process_input] Error: Invalid or missing session ID '{session_id}' in event from queue.")
                return {"error": f"Invalid session ID: {session_id}", "text": "", "mood": "error", "session_id": session_id, "active_persona": "unknown"}

            session = self.device_sessions[session_id]
            session.update_activity() 

            active_persona = session.get_persona()
            session_capabilities = session.capabilities
            
            event_for_main_loop = {
                "session_id": session_id,
                "type": input_event_from_queue.get("type", "text"),
                "payload": input_event_from_queue.get("payload", {}) 
            }
            if "text" in input_event_from_queue and "text" not in event_for_main_loop["payload"]:
                 event_for_main_loop["payload"]["text"] = input_event_from_queue["text"]

            self.log_event("direct_input_received_for_processing", {
                "device_id": session.device_id,
                "session_id": session_id,
                "type": event_for_main_loop["type"],
                "persona": active_persona,
                "payload_sample": str(event_for_main_loop["payload"])[:100],
                "timestamp": datetime.now().isoformat()
            })

            print(f"[Orchestrator _process_input] Processing event for session {session_id} (persona: {active_persona}) via MainLoop.")
            
            llm_response_dict = await self.main_loop_processor.process_input(event_for_main_loop)
            
            raw_assistant_text = llm_response_dict.get("text", "")
            if llm_response_dict.get("error"):
                print(f"[Orchestrator _process_input] MainLoop returned an error for session {session_id}: {llm_response_dict['error']}")
            
            chat_log_for_response_handler = [] 

            self.log_event("main_loop_handler_completed", {
                "device_id": session.device_id,
                "session_id": session_id,
                "type": event_for_main_loop["type"],
                "raw_text_length": len(raw_assistant_text),
                "llm_error": llm_response_dict.get("error"),
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
                "final_payload_text_sample": final_payload.get("text", "")[:100], 
                "timestamp": datetime.now().isoformat()
            })
            
            return final_payload

        except Exception as e:
            print(f"[Orchestrator _process_input] Critical error: {e}")
            import traceback
            traceback.print_exc() 
            self.log_event("direct_processing_error", {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "input_event": str(input_event_from_queue), 
                "timestamp": datetime.now().isoformat()
            })
            
            active_persona_on_error = "unknown"
            current_session_id = session_id or input_event_from_queue.get("session_id", "unknown")
            if current_session_id in self.device_sessions:
                active_persona_on_error = self.device_sessions[current_session_id].get_persona()
            
            return {
                "error": f"Critical error processing input: {str(e)}", 
                "text": f"I encountered a problem. Please try again.", 
                "mood": "error", 
                "session_id": current_session_id, 
                "active_persona": active_persona_on_error
            }

    def log_event(self, event_type: str, data: Dict[str, Any]):
        try:
            log_entry = {
                "event_type": event_type,
                **data,
                "log_timestamp": datetime.now().isoformat() 
            }
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = self.chat_log_dir / f"orchestrator_{today}.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Error logging event '{event_type}': {e}")
    
    def get_device_info(self, session_id: str = None) -> Optional[Dict[str, Any]]:
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
        self.running = False
        self.log_event("orchestrator_stop", {
            "timestamp": datetime.now().isoformat()
        })
        print("Orchestrator processing loop stopping...")
