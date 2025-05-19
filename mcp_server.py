#!/usr/bin/env python3

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import functools
from mcp.server.fastmcp.server import FastMCP, Context

# Assuming these files are in the same directory or accessible in PYTHONPATH
from input_orchestrator import InputOrchestrator # InputType might be needed if used explicitly
from response_handler import ResponseHandler
# from core_functions import process_response_content # Imported by ResponseHandler

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

# ==== Dummy Manager Placeholders ====
class DummyManager:
    def __init__(self, name="dummy_manager"):
        self.name = name
        print(f"[Server INFO] DummyManager '{self.name}' initialized.")
        if self.name == "NotificationManager":
            self.notification_queue = asyncio.Queue()
            print(f"[Server INFO] DummyManager '{self.name}' now has an asyncio.Queue for 'notification_queue'.")

dummy_system_manager = DummyManager("SystemManager")
dummy_display_manager = DummyManager("DisplayManager")
dummy_audio_manager = DummyManager("AudioManager")
dummy_notification_manager = DummyManager("NotificationManager")

# ==== Instantiate Core Components ====
response_formatter = ResponseHandler() # ResponseHandler is used by run_LAURA after orchestrator

#--------------------------------------------------------------------
# THIS IS THE CRITICAL PART:
# The main_loop_handler for the InputOrchestrator needs to be
# the function that actually calls your LLM (e.g., from LAURA_gradio.py's MainLoop)
# and returns the RAW ASSISTANT CONTENT (string) from the LLM.
# The InputOrchestrator's _process_input will then take this raw content
# and pass it to the ResponseHandler.
#--------------------------------------------------------------------

# Placeholder for your ACTUAL LLM processing logic from LAURA_gradio.py or main_loop.py
# This function should take the input_event from the orchestrator,
# call your LLM, and return the LLM's raw text response.
async def actual_llm_main_loop_handler(input_event_from_orchestrator: Dict[str, Any]) -> str:
    """
    This function represents the entry point to your REAL LLM logic.
    It should:
    1. Extract necessary info from input_event_from_orchestrator (query, session_id, chat_log, persona).
    2. Call your LLM (e.g., Anthropic client via an LLMAdapter or directly).
    3. Handle any tool use loop if initiated by the LLM.
    4. Return the final raw text string from the assistant.
    """
    query = input_event_from_orchestrator.get("text", "")
    session_id = input_event_from_orchestrator.get("session_id")
    active_persona = input_event_from_orchestrator.get("active_persona", "LAURA")
    # chat_history = get_chat_history_for_session(session_id) # You need a chat history mechanism

    print(f"[Actual LLM Handler] Processing query: '{query}' for session: {session_id} as {active_persona}")
    
    # --- THIS IS WHERE YOU INTEGRATE YOUR LLM CALL ---
    # Example:
    # raw_assistant_text = await your_llm_adapter.generate_response(
    #     query=query,
    #     chat_log=chat_history,
    #     system_prompt=config.SYSTEM_PROMPT, # Make sure config is accessible
    #     tools=your_tools_definitions # If applicable
    # )
    # For now, a slightly more intelligent placeholder than the original:
    if "bland" in query.lower() or "boring" in query.lower():
        raw_assistant_text = f"[thoughtful] As {active_persona}, I understand your feedback. I'm always striving to be more engaging!"
    elif "hello" in query.lower():
        raw_assistant_text = f"[cheerful] Hello there! This is {active_persona}, ready to assist."
    else:
        raw_assistant_text = f"[casual] {active_persona} here. I got: '{query}'. My main LLM brain is currently simulated in this handler."
    
    print(f"[Actual LLM Handler] Raw response: {raw_assistant_text}")
    return raw_assistant_text


# Now, InputOrchestrator's _process_input method needs to be adjusted
# if it's not already set up to call its main_loop_handler and then pass
# the result to the response_formatter.
# For this example, I'll assume InputOrchestrator's _process_input
# will call its main_loop_handler and get raw_assistant_content,
# and then the run_LAURA tool will take that and pass it to response_formatter.
# This means the CoreLogicHandler is no longer needed.

input_coordinator = InputOrchestrator(
    # This is the key: main_loop_handler should be your actual LLM processing function
    main_loop_handler=actual_llm_main_loop_handler,
    chat_log_dir=str(LOGS_DIR)
)
# The CoreLogicHandler class is no longer the primary brain.
# We will call input_coordinator directly from run_LAURA.

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

def is_token_valid(token: str) -> bool: # Placeholder
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

    active_persona = device_info.get("active_persona", "LAURA")
    session_capabilities = device_info.get("capabilities", {})
    user_text = payload.get("text", "")

    # Construct the event for the InputOrchestrator
    # The orchestrator's _process_input will call its main_loop_handler (actual_llm_main_loop_handler)
    # This main_loop_handler will return the RAW assistant text.
    input_event_for_orchestrator = {
        "session_id": session_id,
        "text": user_text, # Changed from "query" to "text" to match orchestrator
        "type": input_type, # Changed from "input_type" to "type"
        "active_persona": active_persona, # This will be added/overridden by orchestrator's _process_input
        "timestamp": timestamp or datetime.utcnow().isoformat(),
        "client_capabilities": session_capabilities, # Pass along for ResponseHandler
        # Add any other fields your InputOrchestrator or actual_llm_main_loop_handler might expect
    }
    
    await log_event(session_id, "run_LAURA_request_to_orchestrator", {"input_event": input_event_for_orchestrator})

    # The InputOrchestrator's _process_input method should:
    # 1. Call its self.main_loop_handler (which is now actual_llm_main_loop_handler)
    # 2. Get the raw_assistant_content string back.
    # For this script, we are simplifying: run_LAURA will directly call the
    # actual_llm_main_loop_handler (representing the full pipeline up to getting raw LLM text)
    # and then run_LAURA itself will call the response_formatter.
    # This means InputOrchestrator's _process_input just needs to call its main_loop_handler.
    
    # Call the main LLM processing logic (which is now actual_llm_main_loop_handler)
    # This simulates the orchestrator calling its configured main loop handler.
    raw_assistant_content = await actual_llm_main_loop_handler(input_event_for_orchestrator)
    
    # Now, take this raw_assistant_content and pass it to the ResponseHandler
    final_response_payload = await response_formatter.handle_response(
        assistant_content=raw_assistant_content,
        chat_log=[], # Placeholder: You need to fetch/manage actual conversation history here
        session_capabilities=session_capabilities,
        session_id=session_id,
        active_persona=active_persona,
        system_manager=dummy_system_manager,
        display_manager=dummy_display_manager,
        audio_manager=dummy_audio_manager,
        notification_manager=dummy_notification_manager
    )
    
    await log_event(session_id, "run_LAURA_response_from_llm_pipeline", final_response_payload)
    print(f"[SERVER] run_LAURA sending response for session {session_id}: {final_response_payload.get('text')}")
    return final_response_payload


@require_bearer_token
@mcp.tool()
async def push_notification(session_id: str, message: str, level: str = "info", ctx: Context = None) -> Dict[str, Any]:
    # (No changes to this tool needed for the placeholder issue)
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
    # To run InputOrchestrator's internal loop if it uses its queue for other inputs:
    # async def main_async():
    #    asyncio.create_task(input_coordinator.start()) # Start orchestrator's queue processing
    #    config = uvicorn.Config(app, host="0.0.0.0", port=8765, loop="asyncio")
    #    server = uvicorn.Server(config)
    #    await server.serve()
    # asyncio.run(main_async())
    
    # Standard way to run:
    uvicorn.run(app, host="0.0.0.0", port=8765)
