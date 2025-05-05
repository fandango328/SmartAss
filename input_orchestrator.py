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
    """Tracks device connection and capabilities"""
    def __init__(self, device_id: str, capabilities: Dict[str, Any]):
        self.device_id = device_id
        self.session_id = str(uuid.uuid4())
        self.capabilities = capabilities
        self.connected_at = datetime.now()
        self.last_activity = datetime.now()
        # Store active documents separately from conversation
        self.active_documents = []
        
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

class InputOrchestrator:
    """
    Orchestrates input from multiple devices to a single conversation context.
    Routes and normalizes different input types for processing by the main loop.
    
    - Handles device registration and capabilities tracking
    - Routes input to appropriate processors
    - Manages a single input queue for FIFO processing 
    - Keeps documents separate from conversation history
    - Logs session events for troubleshooting
    """
    def __init__(
        self,
        main_loop_handler: Callable[[Dict[str, Any]], Any],
        document_manager=None,
        chat_log_dir: str = "logs"
    ):
        # Core handler
        self.main_loop_handler = main_loop_handler
        self.document_manager = document_manager
        self.chat_log_dir = Path(chat_log_dir)
        
        # Input queue (single queue for all devices)
        self.input_queue = asyncio.Queue()
        
        # Device tracking
        self.device_sessions: Dict[str, DeviceSession] = {}
        self.running = False
        
        # Ensure log directory exists
        self.chat_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Save session start timestamp
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
                # Get next input event from queue
                input_event = await self.input_queue.get()
                
                # Process the input
                await self._process_input(input_event)
                
                # Mark task as done
                self.input_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error processing input: {e}")
                # Simple error logging
                self.log_event("processing_error", {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                await asyncio.sleep(0.1)

    async def register_device(self, device_id: str, capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new device with capability information.
        Returns session information to the client.
        """
        # Create new session
        session = DeviceSession(device_id, capabilities)
        
        # Store in sessions dictionary
        self.device_sessions[session.session_id] = session
        
        # Log the registration
        self.log_event("device_registered", {
            "device_id": device_id,
            "session_id": session.session_id,
            "capabilities": capabilities,
            "timestamp": session.connected_at.isoformat()
        })
        
        # Return session information to client
        return {
            "session_id": session.session_id,
            "status": "connected",
            "connected_at": session.connected_at.isoformat()
        }

    async def unregister_device(self, session_id: str) -> bool:
        """Unregister a device when it disconnects"""
        if session_id in self.device_sessions:
            device_id = self.device_sessions[session_id].device_id
            
            # Log the disconnection
            self.log_event("device_unregistered", {
                "device_id": device_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })
            
            # Remove from active sessions
            del self.device_sessions[session_id]
            return True
        return False

    async def add_input(self, input_event: Dict[str, Any]) -> bool:
        """
        Add input to processing queue.
        Validates session and updates activity timestamp.
        
        Special handling for document uploads:
        - If type is 'document', it's stored in the device session
        - It doesn't generate a chat response directly
        """
        # Check if this has a valid session
        session_id = input_event.get("session_id")
        if session_id not in self.device_sessions:
            print(f"Warning: Input received with invalid session_id: {session_id}")
            return False
        
        # Update activity timestamp
        self.device_sessions[session_id].update_activity()
        
        # Special handling for documents
        if input_event.get("type") == "document":
            # Store document in session for later use
            document_data = input_event.get("files", [])
            if document_data:
                self.device_sessions[session_id].add_document(document_data)
                
                # If we have a document manager, load the files
                if self.document_manager and hasattr(self.document_manager, "load_all_files"):
                    await self.document_manager.load_all_files()
                
                # Log document storage
                self.log_event("document_stored", {
                    "device_id": self.device_sessions[session_id].device_id,
                    "session_id": session_id,
                    "document_count": len(document_data),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Return success without adding to input queue
                return True
            
        # Add to queue and return success
        await self.input_queue.put(input_event)
        return True
        
    async def clear_documents(self, session_id: str) -> bool:
        """Clear all documents for a session"""
        if session_id in self.device_sessions:
            self.device_sessions[session_id].clear_documents()
            
            # If we have a document manager, unload files
            if self.document_manager and hasattr(self.document_manager, "offload_all_files"):
                await self.document_manager.offload_all_files()
                
            # Log document clearance
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
        Adds minimal tracking info but no processing metadata.
        Documents are maintained separately from conversation.
        """
        try:
            # Get session info
            session_id = input_event.get("session_id")
            if session_id not in self.device_sessions:
                raise ValueError(f"Invalid session ID: {session_id}")
            
            session = self.device_sessions[session_id]
            
            # Determine output mode from explicit setting or device capabilities
            if "output_mode" not in input_event:
                # Default to the first output mode the device supports
                supported_outputs = session.capabilities.get("output", ["text"])
                input_event["output_mode"] = supported_outputs[0]
            
            # Document support is handled separately by document manager
            # We don't add documents to the input event to avoid cluttering chat logs
            # The main loop will access documents directly from document_manager
            
            # Record basic processing info
            input_type = input_event.get("type", "unknown")
            
            # Log event before processing
            self.log_event("input_received", {
                "device_id": session.device_id,
                "session_id": session_id,
                "type": input_type,
                "timestamp": datetime.now().isoformat()
            })
            
            # Route to main loop handler
            print(f"Processing {input_type} input from device {session.device_id}")
            await self.main_loop_handler(input_event)
            
            # Log completion  
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
        """
        Log orchestrator events to JSON file.
        Simple append-only logging for troubleshooting.
        """
        try:
            # Create log entry
            log_entry = {
                "event_type": event_type,
                **data
            }
            
            # Get log file path for today
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = self.chat_log_dir / f"orchestrator_{today}.jsonl"
            
            # Append to log file
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            print(f"Error logging event: {e}")
    
    def get_device_info(self, session_id: str = None) -> Dict[str, Any]:
        """Get information about connected devices"""
        if session_id:
            # Return info for specific session
            if session_id in self.device_sessions:
                session = self.device_sessions[session_id]
                return {
                    "device_id": session.device_id,
                    "session_id": session_id,
                    "connected_at": session.connected_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "capabilities": session.capabilities,
                    "document_count": len(session.active_documents)
                }
            return None
        
        # Return summary of all devices
        return {
            "active_devices": len(self.device_sessions),
            "devices": [
                {
                    "device_id": session.device_id,
                    "session_id": sid,
                    "connected_at": session.connected_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
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
