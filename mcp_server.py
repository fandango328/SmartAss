#!/usr/bin/env python3

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import functools
from starlette.applications import Starlette
from mcp.server.fastmcp.server import FastMCP, Context

# ==== Server Setup ====
mcp = FastMCP(
    "LAURA MCP Server",
    message_path="/events/messages/",
    sse_path="/events/sse",
    host="0.0.0.0",
    port=8765,
)
print("FastMCP methods:", dir(mcp))
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
SESSION_REGISTRY: Dict[str, Dict[str, Any]] = {}

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

def is_token_valid(token: str) -> bool:
    return bool(token and isinstance(token, str) and len(token) > 10)

def require_bearer_token(func):
    @functools.wraps(func)
    async def wrapper(*args, ctx: Context = None, **kwargs):
        if ctx is None or not hasattr(ctx, "headers"):
            raise ValueError("Missing authentication context.")
        auth_header = ctx.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise ValueError("Missing Bearer token.")
        token = auth_header.split(" ", 1)[1]
        if not is_token_valid(token):
            raise ValueError("Invalid or expired token.")
        return await func(*args, ctx=ctx, **kwargs)
    return wrapper

@require_bearer_token
@mcp.tool()
async def register_device(
    device_id: str,
    capabilities: Dict[str, Any],
    ctx: Context = None
) -> Dict[str, Any]:
    print(f"[DEBUG] register_device called with device_id={device_id}, capabilities={capabilities}")
    session_id = generate_session_id(device_id)
    created_at = datetime.utcnow().isoformat()
    SESSION_REGISTRY[session_id] = {
        "device_id": device_id,
        "capabilities": capabilities,
        "created_at": created_at,
    }
    await log_event(session_id, "register", {"device_id": device_id, "capabilities": capabilities})
    print(f"[DEBUG] Registered session: {session_id}")
    return {
        "session_id": session_id,
        "created_at": created_at,
        "capabilities": capabilities,
    }

@require_bearer_token
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
    print(f"[DEBUG] run_LAURA called with session_id={session_id}, input_type={input_type}, payload={payload}")
    if session_id not in SESSION_REGISTRY:
        raise ValueError("Invalid session_id. Please register your device first.")

    # Here, insert your actual input processing logic!
    # For now, we'll just echo back the input for testing.
    result_text = f"You said: {payload.get('text', '')}"

    response_payload = {
        "persona": "default",
        "voice": "default",
        "mood": "casual",
        "text": result_text,
        "audio": None,  # You can generate audio bytes here if needed
    }
    await log_event(session_id, "response", response_payload)
    print(f"[DEBUG] run_LAURA response: {response_payload}")
    return response_payload

@require_bearer_token
@mcp.tool()
async def push_notification(
    session_id: str,
    message: str,
    level: str = "info",
    ctx: Context = None
) -> Dict[str, Any]:
    print(f"[DEBUG] push_notification: session_id={session_id}, message={message}, level={level}")
    if session_id not in SESSION_REGISTRY:
        raise ValueError("Invalid session_id")
    await log_event(session_id, "notification", {
        "message": message,
        "level": level
    })
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

# ==== ASGI APP for BOTH HTTP and SSE ====
# HTTP API: /api
# SSE (notifications/streaming): /events

app = mcp.sse_app()

print("SSE App:", app)
print("SSE App routes:", getattr(app, 'routes', None))
print("ROUTES:", [route.path for route in app.routes])

# ==== MAIN for direct execution ====
if __name__ == "__main__":
    import uvicorn
    print("Starting MCP server with HTTP API on /api and SSE on /events ...")
    uvicorn.run(app, host="0.0.0.0", port=8765)
