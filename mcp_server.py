#!/usr/bin/env python3

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests  # For token validation

from mcp.server.fastmcp.server import FastMCP, Context
from input_orchestrator import InputOrchestrator
from response_handler import ResponseHandler
from main_loop import process_input
from tts_handler import TTSHandler

# Environment and config setup
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

SESSION_REGISTRY: Dict[str, Dict[str, Any]] = {}

tts_handler = None
response_handler = ResponseHandler(tts_handler=tts_handler)
orchestrator = InputOrchestrator(main_loop_handler=process_input)

# --- MCP Server ---
mcp = FastMCP("LAURA MCP Server")

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

@mcp.tool()
@require_bearer_token
async def register_device(
    device_id: str,
    capabilities: Dict[str, Any],
    ctx: Context = None
) -> Dict[str, Any]:
    session_id = generate_session_id(device_id)
    created_at = datetime.utcnow().isoformat()
    SESSION_REGISTRY[session_id] = {
        "device_id": device_id,
        "capabilities": capabilities,
        "created_at": created_at,
    }
    await log_event(session_id, "register", {"device_id": device_id, "capabilities": capabilities})
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
        await orchestrator.add_input(event_data)
        input_event = {
            "session_id": session_id,
            "type": input_type,
            "payload": payload,
        }
        processed_result = await orchestrator.main_loop_handler(input_event)

        if "error" in processed_result:
            await log_event(session_id, "error", {"error": processed_result["error"]})
            raise ValueError(processed_result["error"])

        response_payload = await response_handler.handle_response(
            assistant_content=processed_result.get("text", ""),
            chat_log=None,
            session_capabilities={"output": output_mode},
            session_id=session_id
        )
        await log_event(session_id, "response", response_payload)

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
@require_bearer_token
async def push_notification(
    session_id: str,
    message: str,
    level: str = "info",
    ctx: Context = None
) -> Dict[str, Any]:
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

# ASGI app for deployment! (this is what uvicorn/hypercorn will serve)
app = mcp.sse_app()
