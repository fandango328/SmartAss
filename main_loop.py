import asyncio
import config
from laura_tools import tool_registry, get_openai_functions
from function_definitions import sanitize_messages_for_api

from anthropic_adapter import AnthropicLLMAdapter
from openai_adapter import OpenAILLMAdapter

if config.ACTIVE_PROVIDER == "anthropic":
    AdapterClass = AnthropicLLMAdapter
    tool_definitions = tool_registry.get_available_tools()
elif config.ACTIVE_PROVIDER == "openai":
    AdapterClass = OpenAILLMAdapter
    tool_definitions = get_openai_functions()
else:
    raise ValueError(f"Unsupported provider: {config.ACTIVE_PROVIDER}")

# Use these config parameters for API calls:
max_tokens = getattr(config, "MAX_TOKENS", 4096)
temperature = getattr(config, "TEMPERATURE", 0.7)

llm_adapter = AdapterClass(
    api_key=config.MODELS_DATA[config.ACTIVE_PROVIDER + "_api_key"],
    model=config.ANTHROPIC_MODEL if config.ACTIVE_PROVIDER == "anthropic" else config.OPENAI_MODEL,
    functions=tool_definitions,
    system_prompt=config.SYSTEM_PROMPT,
    max_tokens=max_tokens,
    temperature=temperature,
)

def tool_handler(function_name, function_args, call_id):
    import json
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

async def main_loop():
    """
    Main conversational loop for the LAURA assistant, with API message sanitization.

    Steps:
    - Accept user input and append to conversation history
    - Sanitize the message history before every LLM API call using `sanitize_messages_for_api`
    - Pass sanitized messages to the LLM adapter
    - Handle tool/function calls as needed
    - Display or process the assistant's replies

    Note: This ensures only valid, API-compatible messages are sent to the LLM backend.
    """
    conversation = []

    print("LAURA assistant ready! Type your message (or 'exit' to quit).")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break

        conversation.append({"role": "user", "content": user_input})

        sanitized_msgs = sanitize_messages_for_api(conversation)
        if not sanitized_msgs:
            print("[Error]: No valid messages to send to LLM API after sanitization!")
            continue

        result = await llm_adapter.generate_response(
            messages=sanitized_msgs,
            tool_handler=tool_handler,
            system_prompt=config.SYSTEM_PROMPT,
            functions=tool_definitions,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if result.get("error"):
            print(f"[Error]: {result['error']}")
        else:
            assistant_text = result.get("text", "")
            if assistant_text:
                conversation.append({"role": "assistant", "content": assistant_text})
                print(f"Laura: {assistant_text}")
            if result.get("function_calls"):
                print(f"[Tool calls handled: {result['function_calls']}]")

if __name__ == "__main__":
    asyncio.run(main_loop())