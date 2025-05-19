#!/usr/bin/env python3

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import functools
from mcp.server.fastmcp.server import FastMCP, Context

from input_orchestrator import InputOrchestrator, InputType
from response_handler import ResponseHandler
# Assuming core_functions.py contains process_response_content and is accessible
# from core_functions import process_response_content # Ensure this is imported if ResponseHandler needs it

# ==== Server Setup ====
mcp = FastMCP(
    "LAURA MCP Server",
    message_path="/events/messages/",
    sse_path="/events/sse",
    host="0.0.0.0",
    port=8765,
)
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# ==== Placeholder for Core Assistant Logic ====
class CoreLogicHandler:
    async def process_user_input(self, input_event: Dict[str, Any]) -> str:
        user_text = input_event.get("text", "").lower()
        active_persona = input_event.get("active_persona", "LAURA")
        if "hello" in user_text or "hi" in user_text:
            return f"Hello from {active_persona}! How can I assist you today?"
        if "how are you" in user_text:
            return f"I am functioning optimally, thank you for asking! This is {active_persona}."
        # Add more example logic based on your screenshot for 'we're gonna give this one shot...'
        if "one shot" in user_text and "acknowledging" in user_text:
            return f"Acknowledged, {active_persona} is here and listening. What's on your mind?"
        else:
            return f"{active_persona} received: '{input_event.get('text', '')}'. How can I help?"

# ==== Dummy Manager Placeholders ====
# These are basic placeholders. If process_response_content needs them to have specific
# methods or attributes, these classes will need to be more sophisticated.
class DummyManager:
    def __init__(self, name="dummy_manager"):
        self.name = name
        print(f"[Server INFO] {self.name} initialized.")
    # Add any methods that process_response_content might call, e.g.:
    # async def update_display_state(self, state): pass

# Instantiate managers (globally for simplicity in this example)
dummy_system_manager = DummyManager("SystemManager")
dummy_display_manager = DummyManager("DisplayManager")
dummy_audio_manager = DummyManager("AudioManager")
dummy_notification_manager = DummyManager("NotificationManager")

# ==== Instantiate Core Components ====
core_logic_handler = CoreLogicHandler()
response_formatter = ResponseHandler()

async def dummy_orchestrator_main_loop_handler(event):
    print(f"[Orchestrator Dummy Loop] Received event: {event}")
    pass

input_coordinator = InputOrchestrator(
    main_loop_handler=dummy_orchestrator_main_loop_handler,
    chat_log_dir=str(LOGS_DIR)
)

# ==== Logging and Auth (mostly unchanged from your previous version) ====
def get_log_path(session_id: str) -> Path:
    safe_session_id = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in str(session_id))
    return LOGS_DIR / f"{safe_session_id}.jsonl"

async def log_event(session_id: str, event_type: str, data: Dict[str, Any]):
    log_obj = {"event": event_type, "timestamp": datetime.utcnow().isoformat(), "data": data}
    log_path = get_log_path(session_id)
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(log_obj) + "\n")
    except Exception as e:
        print(f"[ERROR] Failed to write to log file {log_path}: {e}")

def is_token_valid(token: str) -> bool:
    return bool(token and isinstance(token, str) and len(token) > 10)

def require_bearer_token(func):
    @functools.wraps(func)
    async def wrapper(*args, ctx: Context = None, **kwargs):
        auth_header = ctx.headers.get("Authorization", "") if ctx and hasattr(ctx, "headers") else ""
        token = auth_header.split(" ", 1)[1] if auth_header.startswith("Bearer ") else None
        if token and not is_token_valid(token):
            raise ValueError("Invalid or expired token.")
        return await func(*args, ctx=ctx, **kwargs)
    return wrapper

# ==== Modified Tools ====
@require_bearer_token
@mcp.tool()
async def register_device(device_id: str, capabilities: Dict[str, Any], ctx: Context = None) -> Dict[str, Any]:
    print(f"[SERVER] register_device called with device_id={device_id}")
    orchestrator_response = await input_coordinator.register_device(device_id, capabilities)
    session_id = orchestrator_response["session_id"]
    created_at = orchestrator_response["connected_at"]
    response_to_client = {"session_id": session_id, "created_at": created_at, "capabilities": capabilities}
    await log_event(session_id, "register_device_tool", {"device_id": device_id, "client_response": response_to_client})
    print(f"[SERVER] Device registered. Session ID: {session_id}")
    return response_to_client

@require_bearer_token
@mcp.tool("run_LAURA")
async def run_LAURA(
    session_id: str, input_type: str, payload: Dict[str, Any], output_mode: List[str],
    broadcast: bool = False, timestamp: str = None, ctx: Context = None
) -> Dict[str, Any]:
    print(f"[SERVER] run_LAURA called for session_id={session_id}")
    device_info = input_coordinator.get_device_info(session_id)
    if not device_info:
        await log_event(session_id, "run_LAURA_error", {"error": "Invalid session_id"})
        raise ValueError("Invalid session_id. Please register your device first.")

    active_persona = device_info.get("active_persona", "LAURA")
    session_capabilities = device_info.get("capabilities", {})
    user_text = payload.get("text", "")

    input_event_for_core = {
        "session_id": session_id, "text": user_text, "input_type": input_type,
        "active_persona": active_persona, "timestamp": timestamp or datetime.utcnow().isoformat(),
    }
    await log_event(session_id, "run_LAURA_request_to_core", input_event_for_core)
    raw_assistant_content = await core_logic_handler.process_user_input(input_event_for_core)

    # Pass the manager instances to handle_response
    final_response_payload = await response_formatter.handle_response(
        assistant_content=raw_assistant_content,
        chat_log=[],
        session_capabilities=session_capabilities,
        session_id=session_id,
        active_persona=active_persona,
        system_manager=dummy_system_manager,         # Pass instance
        display_manager=dummy_display_manager,       # Pass instance
        audio_manager=dummy_audio_manager,           # Pass instance
        notification_manager=dummy_notification_manager # Pass instance
    )
    
    await log_event(session_id, "run_LAURA_response_from_core", final_response_payload)
    print(f"[SERVER] run_LAURA sending response for session {session_id}: {final_response_payload.get('text')}")
    return final_response_payload

@require_bearer_token
@mcp.tool()
async def push_notification(session_id: str, message: str, level: str = "info", ctx: Context = None) -> Dict[str, Any]:
    # (This function remains largely the same as your previous correct version)
    print(f"[DEBUG] push_notification: session_id={session_id}, message={message}, level={level}")
    device_info = input_coordinator.get_device_info(session_id)
    if not device_info:
        raise ValueError("Invalid session_id for push_notification")
    await log_event(session_id, "notification_received_by_server", {"message": message, "level": level})
    status = "received_no_sse_context"
    if ctx:
        await ctx.notify("notification", {
            "session_id": session_id, "message": message, "level": level,
            "timestamp": datetime.utcnow().isoformat()
        })
        await log_event(session_id, "notification_sent_to_client_sse", {"message": message})
        status = "sent_via_sse"
    else:
        print(f"[SERVER] push_notification: No SSE context for session {session_id}")
    return {"status": status, "timestamp": datetime.utcnow().isoformat()}

# ==== SSE APP ONLY ====
app = mcp.sse_app()

if __name__ == "__main__":
    import uvicorn
    print("Starting MCP server with SSE only ...")
    uvicorn.run(app, host="0.0.0.0", port=8765)
