#!/usr/bin/env python3

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import functools
from mcp.server.fastmcp.server import FastMCP, Context

# Assuming these files are in the same directory or accessible in PYTHONPATH
from input_orchestrator import InputOrchestrator
from response_handler import ResponseHandler
# No longer need to import from main_loop.py here, as InputOrchestrator will handle it

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

# ==== Instantiate Core Components ====
response_formatter = ResponseHandler()

# The actual_llm_main_loop_handler function has been removed.
# Its logic will be integrated into InputOrchestrator._process_input.

# ==== Instantiate Input Orchestrator ====
# InputOrchestrator will now directly import and use main_loop.process_input
input_coordinator = InputOrchestrator(
    chat_log_dir=str(LOGS_DIR),
    response_formatter=response_formatter
    # No main_loop_handler is passed anymore
)

# ==== Logging and Auth ====
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

def is_token_valid(token: str) -> bool: # Placeholder for actual token validation
    return bool(token and isinstance(token, str) and len(token) > 10)

def require_bearer_token(func):
    @functools.wraps(func)
    async def wrapper(*args, ctx: Context = None, **kwargs):
        auth_header = ctx.headers.get("Authorization", "") if ctx and hasattr(ctx, "headers") else ""
        token = auth_header.split(" ", 1)[1] if auth_header.startswith("Bearer ") else None
        if token and not is_token_valid(token): # Only validate if a token was actually provided
            raise ValueError("Invalid or expired token.")
        return await func(*args, ctx=ctx, **kwargs)
    return wrapper

# ==== Tools ====
@require_bearer_token
@mcp.tool()
async def register_device(device_id: str, capabilities: Dict[str, Any], ctx: Context = None) -> Dict[str, Any]:
    print(f"[SERVER] register_device called with device_id={device_id}")
    orchestrator_response = await input_coordinator.register_device(device_id, capabilities)
    session_id = orchestrator_response["session_id"]
    created_at = orchestrator_response["connected_at"] 
    response_to_client = {
        "session_id": session_id,
        "created_at": created_at,
        "capabilities": capabilities,
    }
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

    # active_persona is retrieved by input_coordinator._process_input from device_info
    # session_capabilities are also retrieved by input_coordinator._process_input
    user_text = payload.get("text", "")

    input_event_for_orchestrator = {
        "session_id": session_id,
        "text": user_text, 
        "type": input_type, 
        "timestamp": timestamp or datetime.utcnow().isoformat(),
        # client_capabilities will be added by the orchestrator if needed from device_info
        "output_mode": output_mode 
    }
    
    await log_event(session_id, "run_LAURA_request_to_orchestrator", {"input_event": input_event_for_orchestrator})

    # InputOrchestrator._process_input now handles the call to main_loop.py directly
    final_response_payload = await input_coordinator._process_input(input_event_for_orchestrator)
    
    await log_event(session_id, "run_LAURA_response_from_orchestrator_pipeline", final_response_payload)
    print(f"[SERVER] run_LAURA sending response for session {session_id}: {final_response_payload.get('text')}")
    return final_response_payload


@require_bearer_token
@mcp.tool()
async def push_notification(session_id: str, message: str, level: str = "info", ctx: Context = None) -> Dict[str, Any]:
    print(f"[SERVER DEBUG] push_notification: session_id={session_id}, message={message}, level={level}")
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
        print(f"[SERVER] push_notification: No SSE context to send notification for session {session_id}")
        
    return {"status": status, "timestamp": datetime.utcnow().isoformat()}

# ==== SSE APP ONLY ====
app = mcp.sse_app()

if __name__ == "__main__":
    import uvicorn
    print("Starting MCP server with SSE only ...")
    uvicorn.run(app, host="0.0.0.0", port=8765)
