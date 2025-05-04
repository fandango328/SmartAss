import httpx
import asyncio
import logging

class AnthropicLLMAdapter:
    BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, api_key, model, tools=None, system_prompt=None, max_tokens=1024, temperature=0.7):
        self.api_key = api_key
        self.model = model
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def generate_response(
        self,
        messages,
        tool_handler,
        system_prompt=None,
        tools=None,
        max_tokens=None,
        temperature=None,
        metadata=None,
    ):
        """
        Generate a response from Anthropic's Claude model, handling the tool-use loop.

        - messages: List of message dicts (Anthropic format)
        - tool_handler: Callable: (tool_name, tool_input, tool_id) → result (sync or async)
        - system_prompt: Optional system prompt override
        - tools: Optional list of tool definitions (Anthropic format)
        - Returns: dict with keys: text, tool_calls, raw_response, stop_reason, tokens, error (if any)
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json"
        }
        system_content = system_prompt if system_prompt is not None else self.system_prompt
        current_tools = tools if tools is not None else self.tools
        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        current_temp = temperature if temperature is not None else self.temperature

        api_payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": current_max_tokens,
            "temperature": current_temp,
        }
        if system_content:
            api_payload["system"] = system_content
        if current_tools:
            api_payload["tools"] = current_tools
            api_payload["tool_choice"] = "auto"
        if metadata:
            api_payload["metadata"] = metadata

        async with httpx.AsyncClient(timeout=90) as client:
            while True:
                try:
                    resp = await client.post(self.BASE_URL, headers=headers, json=api_payload)
                except Exception as e:
                    logging.error(f"Anthropic request exception: {e}")
                    return {
                        "text": None,
                        "tool_calls": [],
                        "raw_response": None,
                        "stop_reason": None,
                        "tokens": None,
                        "error": f"network_error: {e}"
                    }
                if resp.status_code != 200:
                    try:
                        error_json = resp.json()
                    except Exception:
                        error_json = {}
                    error_type = error_json.get("error", {}).get("type", "unknown_error")
                    error_message = error_json.get("error", {}).get("message", resp.text)
                    logging.error(f"Anthropic API error ({error_type}): {error_message}")
                    return {
                        "text": None,
                        "tool_calls": [],
                        "raw_response": error_json,
                        "stop_reason": None,
                        "tokens": None,
                        "error": f"{error_type}: {error_message}"
                    }
                response = resp.json()

                # Extract content blocks, stop reason, token usage
                content_blocks = response.get("content", [])
                stop_reason = response.get("stop_reason", None)
                usage = response.get("usage", {})
                input_tokens = usage.get("input_tokens")
                output_tokens = usage.get("output_tokens")
                # Aggregate all text blocks
                all_text = " ".join(
                    block["text"] for block in content_blocks if block.get("type") == "text"
                ).strip()
                tool_calls = [
                    block for block in content_blocks if block.get("type") == "tool_use"
                ]

                # Handle tool-use loop as needed
                if stop_reason == "tool_use" and tool_calls:
                    tool_results = []
                    for tc in tool_calls:
                        tool_name = tc["name"]
                        tool_input = tc["input"]
                        tool_id = tc["id"]
                        try:
                            result = tool_handler(tool_name, tool_input, tool_id)
                            if asyncio.iscoroutine(result):
                                result = await result
                        except Exception as tool_exc:
                            result = {"error": f"Tool handler error: {tool_exc}"}
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result
                        })
                    # Add tool results to messages history for the next loop
                    api_payload["messages"] = (
                        api_payload["messages"]
                        + [{"role": "assistant", "content": content_blocks}]
                        + [{"role": "user", "content": tool_results}]
                    )
                    # Remove system/tool_choice for followups per API best practice
                    api_payload.pop("system", None)
                    api_payload.pop("tool_choice", None)
                    continue  # Next API call

                # Return structured result for the main loop
                return {
                    "text": all_text,
                    "tool_calls": tool_calls,
                    "raw_response": response,
                    "stop_reason": stop_reason,
                    "tokens": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                    "error": None
                }