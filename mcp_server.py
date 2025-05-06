import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from mcp.server.fastmcp import FastMCP, Context

# Your orchestrator and response handler imports
from input_orchestrator import InputOrchestrator
from response_handler import ResponseHandler

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

# Initialize the orchestrator and response handler
# You'll want to wire this up to your actual pipeline/handlers
response_handler = ResponseHandler()
orchestrator = InputOrchestrator(main_loop_handler=None)  # You'll inject your main loop handler here

# MCP server
mcp = FastMCP("SmartAss MCP Server")

@mcp.tool()
async def register_device(
    device_id: str,
    capabilities: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Register a new device and return a session ID.
    """
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

@mcp.tool()
async def process_input_event(
    session_id: str,
    input_type: str,
    payload: Dict[str, Any],
    output_mode: list,
    broadcast: bool = False,
    timestamp: str = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Main entrypoint for all input events (text, audio, document, image).
    Handles per-request output_mode, routes through orchestrator, and logs the request/response.
    """
    if session_id not in SESSION_REGISTRY:
        raise ValueError("Invalid session_id. Please register your device first.")

    event_data = {
        "session_id": session_id,
        "input_type": input_type,
        "payload": payload,
        "output_mode": output_mode,
        "broadcast": broadcast,
        "timestamp": timestamp or datetime.utcnow().isoformat()
    }
    await log_event(session_id, "input", event_data)

    # Here you would queue this event with the orchestrator and await a result
    # For demonstration, we'll simulate a response
    # Replace the following with your actual orchestrator/main loop pipeline:
    # processed_result = await orchestrator.queue_input(event_data)
    processed_result = {
        "text": "Echo: " + str(payload.get("text", "")),
        "audio": None,
        "mood": "casual",
        "session_id": session_id
    }

    # Use response handler to format the response based on the requested output modes
    response_payload = await response_handler.handle_response(
        assistant_content=processed_result["text"],
        chat_log=None,
        session_capabilities={"output": output_mode},
        session_id=session_id
    )

    await log_event(session_id, "response", response_payload)

    # Optionally: send notification to clients via MCP notification mechanism if needed
    # await ctx.notify("notification_method", params={...})

    # If broadcast, you would need to loop over sessions/devices as required

    return response_payload

# Example notification tool (sends a notification to clients)
@mcp.tool()
async def push_notification(
    session_id: str,
    message: str,
    level: str = "info",
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Send a notification to the connected client(s) via MCP notification.
    """
    # This sends a one-way notification to the client
    await ctx.notify("notification", {
        "session_id": session_id,
        "message": message,
        "level": level,
        "timestamp": datetime.utcnow().isoformat()
    })
    await log_event(session_id, "notification", {
        "message": message,
        "level": level
    })
    return {"status": "sent"}

if __name__ == "__main__":
    mcp.run()