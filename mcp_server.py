#!/usr/bin/env python3

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import functools
from mcp.server.fastmcp.server import FastMCP, Context #type: ignore

# Assuming these files are in the same directory or accessible in PYTHONPATH
from input_orchestrator import InputOrchestrator
from response_handler import ResponseHandler

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

# ==== Instantiate Input Orchestrator ====
input_coordinator = InputOrchestrator(
    chat_log_dir=str(LOGS_DIR),
    response_formatter=response_formatter
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

# ==== Tools ====
# @require_bearer_token  # Commented out for testing
@mcp.tool()
async def register_device(device_id: str, capabilities: Dict[str, Any], ctx: Context = None) -> Dict[str, Any]:
    import time
    start_time = time.time()
    print(f"[SERVER DEBUG] register_device starting for {device_id}")
    
    orchestrator_response = await input_coordinator.register_device(device_id, capabilities)
    
    end_time = time.time()
    print(f"[SERVER DEBUG] Registration took {end_time - start_time:.2f} seconds")
    
    session_id = orchestrator_response["session_id"]
    created_at = orchestrator_response["connected_at"] 
    response_to_client = {
        "session_id": session_id,
        "created_at": created_at,
        "capabilities": capabilities,
    }
    
    print(f"[SERVER DEBUG] About to return: {response_to_client}")
    return response_to_client

# @require_bearer_token  # Commented out for testing
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

    user_text = payload.get("text", "")

    input_event_for_orchestrator = {
        "session_id": session_id,
        "text": user_text, 
        "type": input_type, 
        "timestamp": timestamp or datetime.utcnow().isoformat(),
        "output_mode": output_mode 
    }
    
    await log_event(session_id, "run_LAURA_request_to_orchestrator", {"input_event": input_event_for_orchestrator})
    final_response_payload = await input_coordinator._process_input(input_event_for_orchestrator)
    
    await log_event(session_id, "run_LAURA_response_from_orchestrator_pipeline", final_response_payload)
    print(f"[SERVER] run_LAURA sending response for session {session_id}: {final_response_payload.get('text')}")
    return final_response_payload

# @require_bearer_token  # Commented out for testing
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

# @require_bearer_token  # Commented out for testing
@mcp.tool()
async def upload_document(session_id: str, filename: str, content: str, content_type: str = "application/octet-stream", ctx: Context = None) -> Dict[str, Any]:
    print(f"[SERVER] upload_document called for session_id={session_id}, filename={filename}")
    device_info = input_coordinator.get_device_info(session_id)
    if not device_info:
        raise ValueError("Invalid session_id for document upload")
    
    # Decode base64 content 
    import base64
    file_data = base64.b64decode(content)
    
    # Save file to query_files directory
    from pathlib import Path
    query_files_dir = Path("/home/user/LAURA/query_files")
    query_files_dir.mkdir(exist_ok=True)
    
    file_path = query_files_dir / filename
    with open(file_path, "wb") as f:
        f.write(file_data)
    
    session = input_coordinator.device_sessions.get(session_id)
    if session and input_coordinator.document_manager_for_main_loop:
        # Trigger document manager to reload all files including the new one
        await input_coordinator.document_manager_for_main_loop.load_all_files(clear_existing=False)

    await log_event(session_id, "document_uploaded", {"filename": filename, "size": len(file_data)})
    return {"status": "uploaded", "filename": filename, "timestamp": datetime.utcnow().isoformat()}

# Add this after your upload_document tool
@mcp.tool()
async def list_available_tools(ctx: Context = None) -> Dict[str, Any]:
    """List all available tools on the server"""
    return {
        "tools": [
            {"name": "register_device", "description": "Register a new device with the system"},
            {"name": "run_LAURA", "description": "Process input through LAURA"},
            {"name": "upload_document", "description": "Upload a document"}
        ]
    }
# ==== SSE APP ONLY ====
app = mcp.sse_app()

if __name__ == "__main__":
    import uvicorn  # type: ignore
    print("Starting MCP server with SSE only ...")
    uvicorn.run(app, host="0.0.0.0", port=8765)
