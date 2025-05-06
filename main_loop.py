import asyncio
import config
import json
from typing import Dict, Any, Optional
from laura_tools import tool_registry, get_llm_tool_definitions
from function_definitions import sanitize_messages_for_api

from llm_integrations.anthropic_adapter import AnthropicLLMAdapter
from llm_integrations.openai_adapter import OpenAILLMAdapter

# Initialize the appropriate LLM adapter based on configuration
if config.ACTIVE_PROVIDER == "anthropic":
    AdapterClass = AnthropicLLMAdapter
    tool_definitions = tool_registry.get_available_tools()
elif config.ACTIVE_PROVIDER == "openai":
    AdapterClass = OpenAILLMAdapter
    tool_definitions = tool_registry.get_llm_tool_definitions()
else:
    raise ValueError(f"Unsupported provider: {config.ACTIVE_PROVIDER}")

# Use these config parameters for API calls
max_tokens = getattr(config, "MAX_TOKENS", 4096)
temperature = getattr(config, "TEMPERATURE", 0.7)

# Initialize the LLM adapter
llm_adapter = AdapterClass(
    api_key=config.MODELS_DATA[config.ACTIVE_PROVIDER + "_api_key"],
    model=config.ANTHROPIC_MODEL if config.ACTIVE_PROVIDER == "anthropic" else config.OPENAI_MODEL,
    functions=tool_definitions,
    system_prompt=config.SYSTEM_PROMPT,
    max_tokens=max_tokens,
    temperature=temperature,
)

# Global conversation state (can be enhanced with session-specific conversations)
conversations = {}

def tool_handler(function_name, function_args, call_id):
    """Handler for tool/function calls from the LLM"""
    try:
        args = json.loads(function_args) if isinstance(function_args, str) else function_args
    except Exception:
        args = function_args
    
    handler = tool_registry.get_handler(function_name)
    if handler is None:
        return f"Unknown tool: {function_name}"
    
    try:
        return handler(**args)
    except Exception as e:
        return f"Tool {function_name} error: {e}"

async def process_input(input_event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process input through the LLM and return results.
    This is the main handler function that can be called by the InputOrchestrator.
    
    Args:
        input_event: Dict with input data including:
            - session_id: Unique session identifier
            - type: Input type (text, audio, etc)
            - payload: The actual input content
            
    Returns:
        Dict with processed response:
            - text: The assistant's response text
            - function_calls: List of any function calls that were made
            - error: Any error message (if applicable)
    """
    try:
        session_id = input_event.get("session_id")
        input_type = input_event.get("type", "text")
        payload = input_event.get("payload", {})
        
        # Get or initialize conversation history for this session
        if session_id not in conversations:
            conversations[session_id] = []
            
        conversation = conversations[session_id]
        
        # Extract the user message based on input type
        user_message = ""
        if input_type == "text":
            user_message = payload.get("text", "")
        elif input_type == "voice":
            # Assuming voice has been transcribed to text
            user_message = payload.get("transcript", "")
        else:
            user_message = f"[{input_type} input received]"
            
        # Add user message to conversation history
        conversation.append({"role": "user", "content": user_message})
        
        # Sanitize messages for API call
        sanitized_msgs = sanitize_messages_for_api(conversation)
        if not sanitized_msgs:
            return {"error": "No valid messages to send to LLM API after sanitization"}
        
        # Generate response using the LLM adapter
        result = await llm_adapter.generate_response(
            messages=sanitized_msgs,
            tool_handler=tool_handler,
            system_prompt=config.SYSTEM_PROMPT,
            functions=tool_definitions,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Add assistant response to conversation if available
        if "text" in result and result["text"]:
            conversation.append({"role": "assistant", "content": result["text"]})
        
        # Limit conversation history to prevent excessive token usage
        if len(conversation) > 20:  # Arbitrary limit, adjust as needed
            conversations[session_id] = conversation[-20:]
            
        return result
        
    except Exception as e:
        return {"error": f"Error processing input: {str(e)}"}

async def main_loop():
    """
    Original main conversational loop for standalone operation.
    Preserved for backward compatibility.
    """
    conversation = []

    print("LAURA assistant ready! Type your message (or 'exit' to quit).")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break

        # Create input event similar to what the orchestrator would produce
        input_event = {
            "session_id": "console_session",
            "type": "text",
            "payload": {"text": user_input}
        }
        
        # Process using the same handler function
        result = await process_input(input_event)

        if result.get("error"):
            print(f"[Error]: {result['error']}")
        else:
            assistant_text = result.get("text", "")
            if assistant_text:
                print(f"Laura: {assistant_text}")
            if result.get("function_calls"):
                print(f"[Tool calls handled: {result['function_calls']}]")

if __name__ == "__main__":
    asyncio.run(main_loop())
