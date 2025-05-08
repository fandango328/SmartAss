#!/usr/bin/env python3

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests  # For token validation

# Correct import paths for your project
from mcp.server.fastmcp.server import FastMCP, Context
from input_orchestrator import InputOrchestrator
from response_handler import ResponseHandler
from main_loop import process_input
from tts_handler import TTSHandler

print("[LAURA] Initializing LAURA MCP Server components...")

# Set environment variables before creating the FastMCP instance
os.environ['FASTMCP_HOST'] = '0.0.0.0'
os.environ['FASTMCP_PORT'] = '8765'

# Constants and configuration
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
print(f"[LAURA] Log directory created at {LOGS_DIR}")

# Session registry
SESSION_REGISTRY: Dict[str, Dict[str, Any]] = {}
print("[LAURA] Session registry initialized")

# Initialize response handler (add TTSHandler() if you want TTS)
print("[LAURA] Setting up response handler...")
tts_handler = None
response_handler = ResponseHandler(tts_handler=tts_handler)
print("[LAURA] Response handler initialized")

# Initialize the orchestrator with the main loop handler
print("[LAURA] Setting up input orchestrator...")
orchestrator = InputOrchestrator(main_loop_handler=process_input)
print("[LAURA] Input orchestrator initialized")

# Create the MCP server
print("[LAURA] Creating MCP server...")
mcp = FastMCP("LAURA MCP Server")
print("[LAURA] MCP server created")

def generate_session_id(device_id: str) -> str:
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{device_id}_{now}"

def get_log_path(session_id: str) -> Path:
    return LOGS_DIR / f"{session_id}.jsonl"

async def log_event(session_id: str, event_type: str, data: Dict[str, Any]):
    log_obj = {
        "event": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
    }
    log_path = get_log_path(session_id)
    with open(log_path, "a") as f:
        f.write(json.dumps(log_obj) + "\n")
    print(f"[LAURA] Logged {event_type} event for session {session_id}")

# --- TOKEN VALIDATION ---
def is_token_valid(token: str) -> bool:
    # Replace this with a real check against your Auth server if needed!
    # For development: just accept any non-empty token (simulate "valid")
    # For production: query your auth server's introspection endpoint here
    return bool(token and isinstance(token, str) and len(token) > 10)
    # Example for real validation:
    # try:
    #     resp = requests.post(
    #         "http://192.168.0.50:5000/introspect",
    #         data={"token": token},
    #         auth=("test-client", "secret"),
    #         timeout=3
    #     )
    #     data = resp.json()
    #     return data.get("active", False)
    # except Exception as e:
    #     print(f"[LAURA] Token introspection error: {e}")
    #     return False

# --- AUTH DECORATOR ---
def require_bearer_token(func):
    async def wrapper(*args, ctx: Context = None, **kwargs):
        if ctx is None or not hasattr(ctx, "headers"):
            print("[LAURA] No context or headers found for authentication.")
            raise ValueError("Missing authentication context.")
        auth_header = ctx.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            print("[LAURA] No Bearer token found in Authorization header.")
            raise ValueError("Missing Bearer token.")
        token = auth_header.split(" ", 1)[1]
        if not is_token_valid(token):
            print("[LAURA] Invalid or expired bearer token.")
            raise ValueError("Invalid or expired token.")
        return await func(*args, ctx=ctx, **kwargs)
    return wrapper

# --- MCP ENDPOINTS (PROTECTED) ---

@mcp.tool()
@require_bearer_token
async def register_device(
    device_id: str,
    capabilities: Dict[str, Any],
    ctx: Context = None
) -> Dict[str, Any]:
    print(f"[LAURA] Registering new device: {device_id} with capabilities: {capabilities}")
    session_id = generate_session_id(device_id)
    created_at = datetime.utcnow().isoformat()
    SESSION_REGISTRY[session_id] = {
        "device_id": device_id,
        "capabilities": capabilities,
        "created_at": created_at,
    }
    await log_event(session_id, "register", {"device_id": device_id, "capabilities": capabilities})
    print(f"[LAURA] Device registered with session ID: {session_id}")
    return {
        "session_id": session_id,
        "created_at": created_at,
        "capabilities": capabilities,
    }

@mcp.tool("run_LAURA")
@require_bearer_token
async def run_LAURA(
    session_id: str,
    input_type: str,
    payload: Dict[str, Any],
    output_mode: List[str],
    broadcast: bool = False,
    timestamp: str = None,
    ctx: Context = None
) -> Dict[str, Any]:
    print(f"[LAURA] Received {input_type} input from session {session_id}")
    print(f"[LAURA] Payload: {payload}")
    print(f"[LAURA] Requested output modes: {output_mode}")

    if session_id not in SESSION_REGISTRY:
        print(f"[LAURA] ERROR: Invalid session_id: {session_id}")
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
        print(f"[LAURA] Adding input to orchestrator queue")
        await orchestrator.add_input(event_data)

        input_event = {
            "session_id": session_id,
            "type": input_type,
            "payload": payload,
        }

        print(f"[LAURA] Delegating to main loop handler for processing")
        processed_result = await orchestrator.main_loop_handler(input_event)
        print(f"[LAURA] Main loop processing complete")

        if "error" in processed_result:
            print(f"[LAURA] Error in processing: {processed_result['error']}")
            await log_event(session_id, "error", {"error": processed_result["error"]})
            raise ValueError(processed_result["error"])

        print(f"[LAURA] Formatting response for output modes: {output_mode}")
        response_payload = await response_handler.handle_response(
            assistant_content=processed_result.get("text", ""),
            chat_log=None,
            session_capabilities={"output": output_mode},
            session_id=session_id
        )
        print(f"[LAURA] Response payload prepared: {response_payload}")

        await log_event(session_id, "response", response_payload)

        if broadcast and ctx:
            print(f"[LAURA] Broadcasting message to other sessions")
            for other_id in SESSION_REGISTRY:
                if other_id != session_id:
                    await ctx.notify("broadcast", {
                        "from_session": session_id,
                        "message": processed_result.get("text", ""),
                        "timestamp": datetime.utcnow().isoformat()
                    })

        print(f"[LAURA] Returning response to client")
        return response_payload

    except Exception as e:
        print(f"[LAURA] Exception in run_LAURA: {str(e)}")
        error_data = {"error": str(e), "input": event_data}
        await log_event(session_id, "error", error_data)
        raise

@mcp.tool()
@require_bearer_token
async def push_notification(
    session_id: str,
    message: str,
    level: str = "info",
    ctx: Context = None
) -> Dict[str, Any]:
    print(f"[LAURA] Pushing notification to session {session_id}: {message} (level: {level})")
    if session_id not in SESSION_REGISTRY:
        print(f"[LAURA] ERROR: Invalid session_id for notification: {session_id}")
        raise ValueError("Invalid session_id")
    await log_event(session_id, "notification", {
        "message": message,
        "level": level
    })
    if ctx:
        print(f"[LAURA] Sending notification via MCP context")
        await ctx.notify("notification", {
            "session_id": session_id,
            "message": message,
            "level": level,
            "timestamp": datetime.utcnow().isoformat()
        })
    print(f"[LAURA] Notification sent successfully")
    return {
        "status": "sent",
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    try:
        print("[LAURA] ==============================================")
        print("[LAURA] Starting LAURA MCP Server on ws://0.0.0.0:8765")
        print("[LAURA] Waiting for device connections...")
        print("[LAURA] ==============================================")
        mcp.run(transport="sse")
    except KeyboardInterrupt:
        print("\n[LAURA] Server stopped by user.")
    except Exception as e:
        print(f"\n[LAURA] Server error: {str(e)}")
