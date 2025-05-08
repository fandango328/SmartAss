#!/usr/bin/env python3

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from mcp.server.fastmcp import FastMCP, Context

# Your orchestrator and response handler imports
from input_orchestrator import InputOrchestrator
from response_handler import ResponseHandler

# Import the main loop handler
from main_loop import process_input
from tts_handler import TTSHandler

# Constants and configuration
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Session registry
SESSION_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Initialize response handler with TTS capabilities if needed
tts_handler = None  # Replace with TTSHandler() if you want TTS functionality
response_handler = ResponseHandler(tts_handler=tts_handler)

# Initialize the orchestrator with the main loop handler
orchestrator = InputOrchestrator(main_loop_handler=process_input)

# Create the MCP server
mcp = FastMCP("LAURA MCP Server")

def generate_session_id(device_id: str) -> str:
    """Generate a unique session ID using device ID and timestamp"""
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{device_id}_{now}"

def get_log_path(session_id: str) -> Path:
    """Get the log file path for a session"""
    return LOGS_DIR / f"{session_id}.jsonl"

async def log_event(session_id: str, event_type: str, data: Dict[str, Any]):
    """Log an event to the session's log file"""
    log_obj = {
        "event": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
    }
    log_path = get_log_path(session_id)
    with open(log_path, "a") as f:
        f.write(json.dumps(log_obj) + "\n")

@mcp.tool()
async def register_device(
    device_id: str,
    capabilities: Dict[str, Any],
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Register a new device and return a session ID.
    
    Args:
        device_id: Unique identifier for the device
        capabilities: Device capabilities (input/output methods, etc.)
        ctx: MCP context object
        
    Returns:
        Session details including ID and timestamp
    """
    session_id = generate_session_id(device_id)
    created_at = datetime.utcnow().isoformat()
    
    SESSION_REGISTRY[session_id] = {
        "device_id": device_id,
        "capabilities": capabilities,
        "created_at": created_at,
    }
    
    await log_event(session_id, "register", {
        "device_id": device_id, 
        "capabilities": capabilities
    })
    
    return {
        "session_id": session_id,
        "created_at": created_at,
        "capabilities": capabilities,
    }

@mcp.tool("run_LAURA")
async def run_LAURA(
    session_id: str,
    input_type: str,
    payload: Dict[str, Any],
    output_mode: List[str],
    broadcast: bool = False,
    timestamp: str = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Engage the agent with an input event.

    Args:
        session_id: Active session ID from register_device
        input_type: Type of input ("text", "audio", "image", "document")
        payload: Input payload (contents depend on input_type)
        output_mode: List of requested output formats ["text", "audio", etc]
        broadcast: Whether to broadcast to multiple sessions
        timestamp: Client-side timestamp (optional)
        ctx: MCP context object

    Returns:
        Response with requested output formats
    """
    if session_id not in SESSION_REGISTRY:
        raise ValueError("Invalid session_id. Please register your device first.")

    event_data = {
        "session_id": session_id,
        "type": input_type,
        "payload": payload,
        "output_mode": output_mode,
        "broadcast": broadcast,
        "timestamp": timestamp or datetime.utcnow().isoformat()
    }
    await log_event(session_id, "input", event_data)

    try:
        # Add input to orchestrator queue if appropriate (optional)
        await orchestrator.add_input(event_data)

        # Create an input event for processing
        input_event = {
            "session_id": session_id,
            "type": input_type,
            "payload": payload,
        }

        # Delegate to the main loop handler (from main_loop)
        processed_result = await orchestrator.main_loop_handler(input_event)

        if "error" in processed_result:
            await log_event(session_id, "error", {"error": processed_result["error"]})
            raise ValueError(processed_result["error"])

        # Format response based on requested output modes
        response_payload = await response_handler.handle_response(
            assistant_content=processed_result.get("text", ""),
            chat_log=None,  # Enhance to pass chat logs if available
            session_capabilities={"output": output_mode},
            session_id=session_id
        )

        await log_event(session_id, "response", response_payload)

        # Handle broadcast if needed
        if broadcast and ctx:
            for other_id in SESSION_REGISTRY:
                if other_id != session_id:
                    await ctx.notify("broadcast", {
                        "from_session": session_id,
                        "message": processed_result.get("text", ""),
                        "timestamp": datetime.utcnow().isoformat()
                    })

        return response_payload

    except Exception as e:
        error_data = {"error": str(e), "input": event_data}
        await log_event(session_id, "error", error_data)
        raise

@mcp.tool()
async def push_notification(
    session_id: str,
    message: str,
    level: str = "info",
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Send a notification to a connected client.
    
    Args:
        session_id: Active session ID
        message: Notification message
        level: Importance level ("info", "warning", "error")
        ctx: MCP context object
        
    Returns:
        Status information
    """
    if session_id not in SESSION_REGISTRY:
        raise ValueError("Invalid session_id")
        
    # Log the notification
    await log_event(session_id, "notification", {
        "message": message,
        "level": level
    })
    
    # Send notification via MCP notification mechanism
    if ctx:
        await ctx.notify("notification", {
            "session_id": session_id,
            "message": message,
            "level": level,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    return {
        "status": "sent",
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    try:
        print("[MCP] Starting LAURA MCP Server on ws://localhost:8765")
        mcp.run()
    except KeyboardInterrupt:
        print("\n[MCP] Server stopped by user.")
