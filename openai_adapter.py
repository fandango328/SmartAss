import httpx
import asyncio
import logging

class OpenAILLMAdapter:
    BASE_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key, model, functions=None, system_prompt=None, max_tokens=1024, temperature=0.7):
        self.api_key = api_key
        self.model = model
        self.functions = functions or []
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def generate_response(
        self,
        messages,
        tool_handler,
        system_prompt=None,
        functions=None,
        max_tokens=None,
        temperature=None,
        metadata=None,
    ):
        """
        Generate a response from OpenAI's chat models, with function-calling/tool-use loop.

        - messages: List of message dicts (OpenAI format)
        - tool_handler: Callable: (function_name, function_args, call_id) → result (sync or async)
        - system_prompt: Optional system prompt override
        - functions: Optional list of function definitions (OpenAI format)
        - metadata: Optional metadata, ignored here
        - Returns: dict with keys: text, function_calls, raw_response, stop_reason, tokens, error
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        system_content = system_prompt if system_prompt is not None else self.system_prompt
        current_functions = functions if functions is not None else self.functions
        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        current_temp = temperature if temperature is not None else self.temperature

        # Build initial message list
        msgs = list(messages)
        if system_content:
            # Insert system prompt as the first message if not present
            has_system = any(m["role"] == "system" for m in msgs)
            if not has_system:
                msgs = [{"role": "system", "content": system_content}] + msgs

        api_payload = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": current_max_tokens,
            "temperature": current_temp,
        }
        if current_functions:
            api_payload["tools"] = [
                {
                    "type": "function",
                    "function": fn
                } for fn in current_functions
            ]
            api_payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=90) as client:
            while True:
                try:
                    resp = await client.post(self.BASE_URL, headers=headers, json=api_payload)
                except Exception as e:
                    logging.error(f"OpenAI request exception: {e}")
                    return {
                        "text": None,
                        "function_calls": [],
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
                    logging.error(f"OpenAI API error ({error_type}): {error_message}")
                    return {
                        "text": None,
                        "function_calls": [],
                        "raw_response": error_json,
                        "stop_reason": None,
                        "tokens": None,
                        "error": f"{error_type}: {error_message}"
                    }
                response = resp.json()

                choices = response.get("choices", [])
                if not choices:
                    return {
                        "text": None,
                        "function_calls": [],
                        "raw_response": response,
                        "stop_reason": None,
                        "tokens": None,
                        "error": "no_choices"
                    }
                choice = choices[0]
                message = choice.get("message", {})
                stop_reason = choice.get("finish_reason", None)
                usage = response.get("usage", {})
                input_tokens = usage.get("prompt_tokens")
                output_tokens = usage.get("completion_tokens")

                # Detect function/tool calls
                function_calls = []
                if "tool_calls" in message:
                    # GPT-4o, GPT-4-turbo, etc. (multi-tool)
                    function_calls = message["tool_calls"]
                elif "function_call" in message:
                    # GPT-3.5-turbo-0613/1106 (single-tool)
                    fc = message["function_call"]
                    if fc:
                        function_calls = [{"id": "function_call", "function": fc}]

                # Tool-use loop if needed
                if function_calls:
                    tool_results = []
                    for call in function_calls:
                        fn = call["function"] if "function" in call else call
                        function_name = fn.get("name")
                        function_args = fn.get("arguments", "{}")
                        call_id = call.get("id")
                        try:
                            result = tool_handler(function_name, function_args, call_id)
                            if asyncio.iscoroutine(result):
                                result = await result
                        except Exception as tool_exc:
                            result = {"error": f"Tool handler error: {tool_exc}"}
                        tool_results.append({
                            "tool_call_id": call_id,
                            "role": "tool",
                            "name": function_name,
                            "content": result if isinstance(result, str) else str(result)
                        })
                    # Append tool_results as messages and loop again
                    api_payload["messages"] = (
                        api_payload["messages"] +
                        [{"role": "assistant", "tool_calls": function_calls}] +
                        tool_results
                    )
                    # Remove system/tool_choice for followup per OpenAI docs
                    api_payload.pop("tool_choice", None)
                    api_payload.pop("tools", None)
                    continue

                # Return structured result for the main loop
                all_text = message.get("content", "").strip() if message else ""
                return {
                    "text": all_text,
                    "function_calls": function_calls,
                    "raw_response": response,
                    "stop_reason": stop_reason,
                    "tokens": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                    "error": None
                }