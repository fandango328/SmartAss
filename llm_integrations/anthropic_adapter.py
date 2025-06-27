import httpx # type: ignore
import asyncio
import logging
import json # For pretty printing dicts
import traceback # For detailed error logging
from typing import List, Dict, Any, Optional, Callable

class AnthropicLLMAdapter:
    BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, api_key, model, tools=None, system_prompt=None, max_tokens=1024, temperature=0.7):
        self.api_key = api_key
        self.model = model
        self.base_tools = tools or [] # Default tools for the adapter instance if no override
        self.base_system_prompt = system_prompt # Default system prompt
        self.base_max_tokens = max_tokens
        self.base_temperature = temperature
        print(f"[DEBUG AnthropicAdapter __init__] Initialized with model: {self.model}, system_prompt: '{str(self.base_system_prompt)[:100]}...'")
        # print(f"[DEBUG AnthropicAdapter __init__] Default tools: {json.dumps(self.base_tools, indent=2)}") # Can be verbose

    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tool_handler: Optional[Callable] = None,  # <-- NOW OPTIONAL for caching calls
        system_prompt: Optional[str] = None,
        system_blocks: Optional[List[Dict[str, Any]]] = None,  # 🎂 NEW: For proper caching!
        tools: Optional[List[Dict[str, Any]]] = None, # Tools for this specific call
        tool_choice: Optional[Dict[str, Any]] = None, # Tool choice for this specific call
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        custom_headers: Optional[Dict[str, str]] = None  # <-- NEW: For caching headers
    ):
        print(f"\n[DEBUG AnthropicAdapter generate_response] Called.")
        # print(f"  Initial messages (count: {len(messages)}):")
        # for i, msg in enumerate(messages):
        #     content_str = str(msg.get('content'))
        #     print(f"    {i}: role={msg.get('role')}, content='{content_str[:70]}{'...' if len(content_str) > 70 else ''}'")

        # Prepare headers with custom header support
        request_specific_headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json"
        }
        
        # Merge custom headers (for caching)
        if custom_headers:
            request_specific_headers.update(custom_headers)
            print(f"[DEBUG AnthropicAdapter] Using custom headers: {custom_headers}")

        current_system_prompt = system_prompt if system_prompt is not None else self.base_system_prompt
        # If 'tools' is explicitly passed as None for this call, it means no tools, even if base_tools exist.
        # If 'tools' is not passed (i.e., is PENDING/default), then use base_tools.
        current_tools = tools if tools is not None else self.base_tools
        current_tool_choice = tool_choice # Passed directly from main_loop
        current_max_tokens = max_tokens if max_tokens is not None else self.base_max_tokens
        current_temp = temperature if temperature is not None else self.base_temperature

        # Create a mutable copy of messages for the internal tool loop
        current_api_messages = [msg.copy() for msg in messages] # Deep copy items if they are dicts

        print(f"  Effective system_prompt: '{str(current_system_prompt)[:70]}...'")
        print(f"  Effective tools (count): {len(current_tools)}")
        print(f"  Effective tool_choice: {current_tool_choice}")
        print(f"  Effective max_tokens: {current_max_tokens}, temperature: {current_temp}")
        if metadata: print(f"  Metadata: {metadata}")

        loop_count = 0
        while True:
            loop_count += 1
            print(f"[DEBUG AnthropicAdapter generate_response] Tool Loop Iteration: {loop_count}")

            api_payload = {
                "model": self.model,
                "messages": current_api_messages,
                "max_tokens": current_max_tokens,
                "temperature": current_temp,
            }
            # 🎂 GBBO-worthy system structure for proper caching!
            if loop_count == 1:  # System content only for the first API call
                if system_blocks:
                    # Use structured system blocks for caching
                    system_content = []
                    
                    # Add base system prompt first
                    if current_system_prompt:
                        system_content.append({
                            "type": "text",
                            "text": current_system_prompt
                        })
                    
                    # Add cached memory blocks
                    system_content.extend(system_blocks)
                    
                    api_payload["system"] = system_content
                    print(f"[DEBUG] 🧁 Using {len(system_content)} system blocks with caching")
                    
                elif current_system_prompt:
                    # Fallback to simple string system prompt
                    api_payload["system"] = current_system_prompt
                    print(f"[DEBUG] Using simple system prompt (no caching)")
            
            if current_tools: # If there are tools for this call
                api_payload["tools"] = current_tools
                if current_tool_choice: # And a specific choice is made (e.g. "auto")
                    api_payload["tool_choice"] = current_tool_choice
                # If no tool_choice is provided but tools are, Anthropic defaults to auto,
                # but being explicit (as done by main_loop) is good.
            
            if metadata and loop_count == 1: # Metadata typically for the first call
                api_payload["metadata"] = metadata

            # print(f"[DEBUG AnthropicAdapter generate_response] API Payload (Loop {loop_count}):\n{json.dumps(api_payload, indent=2, default=str)}") # Can be very verbose

            async with httpx.AsyncClient(timeout=90) as client:
                try:
                    # print(f"[DEBUG AnthropicAdapter generate_response] Attempting POST to {self.BASE_URL} (Loop {loop_count})")
                    resp = await client.post(self.BASE_URL, headers=request_specific_headers, json=api_payload)
                    # print(f"[DEBUG AnthropicAdapter generate_response] Raw Anthropic Response Status (Loop {loop_count}): {resp.status_code}")
                    try:
                        response_json = resp.json()
                        # print(f"[DEBUG AnthropicAdapter generate_response] Raw Anthropic Response JSON (Loop {loop_count}):\n{json.dumps(response_json, indent=2, default=str)}")
                    except json.JSONDecodeError as json_e:
                        raw_text_response = resp.text
                        print(f"[ERROR AnthropicAdapter generate_response] Failed to decode JSON (Loop {loop_count}): {json_e}. Response text: {raw_text_response[:500]}")
                        return {"text": None, "tool_calls": [], "raw_response": raw_text_response, "stop_reason": "error_json_decode", "tokens": None, "error": f"json_decode_error: {json_e}"}
                except httpx.RequestError as req_exc:
                    logging.error(f"Anthropic request exception (Loop {loop_count}): {req_exc}")
                    print(f"[ERROR AnthropicAdapter generate_response] httpx.RequestError (Loop {loop_count}): {req_exc}")
                    return {"text": None, "tool_calls": [], "raw_response": None, "stop_reason": "network_error", "tokens": None, "error": f"network_error: {req_exc}"}
                except Exception as e:
                    logging.error(f"Unexpected Anthropic request exception (Loop {loop_count}): {e}")
                    print(f"[ERROR AnthropicAdapter generate_response] Unexpected exception during request (Loop {loop_count}): {e}")
                    return {"text": None, "tool_calls": [], "raw_response": None, "stop_reason": "request_exception", "tokens": None, "error": f"request_exception: {e}"}

            if resp.status_code != 200:
                error_type = response_json.get("error", {}).get("type", "unknown_api_error")
                error_message_detail = response_json.get("error", {}).get("message", "No error message in JSON.")
                logging.error(f"Anthropic API error ({error_type}) (Loop {loop_count}): {error_message_detail}")
                print(f"[ERROR AnthropicAdapter generate_response] API Error. Status: {resp.status_code}, Type: {error_type}, Message: {error_message_detail} (Loop {loop_count})")
                return {"text": None, "tool_calls": [], "raw_response": response_json, "stop_reason": error_type, "tokens": None, "error": f"{error_type}: {error_message_detail}"}

            content_blocks = response_json.get("content", [])
            stop_reason = response_json.get("stop_reason", None)
            usage = response_json.get("usage", {})
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")

            print(f"[DEBUG AnthropicAdapter generate_response] Stop Reason: {stop_reason} (Loop {loop_count})")
            # print(f"[DEBUG AnthropicAdapter generate_response] Usage: input={input_tokens}, output={output_tokens} (Loop {loop_count})")
            # print(f"[DEBUG AnthropicAdapter generate_response] Content blocks from response (Loop {loop_count}): {content_blocks}")

            # Extract text from content blocks and handle internal duplication
            text_blocks = [block.get("text", "") for block in content_blocks if block.get("type") == "text"]
            all_text_content = " ".join(text_blocks).strip()

            # Handle internal duplication within the text (split by double newlines and dedupe)
            if "\n\n" in all_text_content:
                text_segments = [segment.strip() for segment in all_text_content.split("\n\n") if segment.strip()]
                unique_segments = []
                for segment in text_segments:
                    if segment not in unique_segments:
                        unique_segments.append(segment)
                current_turn_text = "\n\n".join(unique_segments) if len(unique_segments) > 1 else unique_segments[0] if unique_segments else ""
            else:
                current_turn_text = all_text_content
            print(f"[DEBUG AnthropicAdapter generate_response] Extracted text from current turn: '{current_turn_text[:100]}...' (Loop {loop_count})")

            tool_use_blocks = [block for block in content_blocks if block.get("type") == "tool_use"]
            # print(f"[DEBUG AnthropicAdapter generate_response] Tool use blocks found: {len(tool_use_blocks)} (Loop {loop_count})")

            # Special handling for caching requests
            if custom_headers and "X-Anthropic-Cache-Control" in custom_headers:
                print(f"[DEBUG AnthropicAdapter] This was a caching-related call. Stop reason: {stop_reason}.")
                # For a successful caching call, we expect stop_reason "end_turn" and specific confirmation text
                expected_cache_confirmation = "OK, Caching Complete."
                if stop_reason == "end_turn" and expected_cache_confirmation in current_turn_text:
                    print("[DEBUG AnthropicAdapter] Caching call confirmation matches expected response.")
                else:
                    print(f"[WARN AnthropicAdapter] Caching call response unexpected. Text: '{current_turn_text}', Stop: {stop_reason}")
                
                # Return the result even if unexpected - let caller handle validation
                return {
                    "text": current_turn_text,
                    "tool_calls": [], # No tool calls expected in caching response
                    "raw_response": response_json,
                    "stop_reason": stop_reason,
                    "tokens": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                    "error": None
                }

            # Handle tool use (only if tool_handler is available)
            if stop_reason == "tool_use" and tool_use_blocks and tool_handler:
                print(f"[DEBUG AnthropicAdapter generate_response] Handling tool_use. Number of tools to call: {len(tool_use_blocks)}")
                # Append assistant's message (that includes tool_use blocks) to current_api_messages
                # The 'content' for an assistant message leading to tool use is a list of blocks.
                assistant_turn_content_blocks = [block.copy() for block in content_blocks] # Ensure we're working with copies
                current_api_messages.append({"role": "assistant", "content": assistant_turn_content_blocks})
                # print(f"[DEBUG AnthropicAdapter generate_response] Appended assistant's tool_use turn to current_api_messages.")

                tool_results_for_api = []
                for tc_block in tool_use_blocks:
                    tool_name = tc_block.get("name")
                    tool_input_args = tc_block.get("input", {}) # Ensure it's a dict
                    tool_id = tc_block.get("id")

                    if not all([tool_name, tool_id]):
                        print(f"[ERROR AnthropicAdapter generate_response] Malformed tool_use block: {tc_block}. Skipping.")
                        # Optionally add an error tool_result
                        tool_results_for_api.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id or f"unknown_tool_id_for_{tool_name}",
                            "content": [{"type": "text", "text": "Error: Malformed tool_use block received from LLM."}],
                            "is_error": True
                        })
                        continue
                    
                    print(f"[DEBUG AnthropicAdapter generate_response] Executing tool: Name='{tool_name}', ID='{tool_id}', Input='{str(tool_input_args)[:100]}...'")
                    try:
                        tool_execution_result_content = await tool_handler(tool_name, tool_input_args, tool_id)
                        
                        # Ensure tool result content is in Anthropic's expected format (list of blocks or string)
                        if isinstance(tool_execution_result_content, str):
                            tool_result_content_for_api = [{"type": "text", "text": tool_execution_result_content}]
                        elif isinstance(tool_execution_result_content, dict) and "error" in tool_execution_result_content: # Handler returned an error dict
                             tool_result_content_for_api = [{"type": "text", "text": f"Error from tool: {tool_execution_result_content['error']}"}]
                        elif isinstance(tool_execution_result_content, list): # Assume it's already a list of blocks
                            tool_result_content_for_api = tool_execution_result_content
                        else: # Try to coerce to string block
                            tool_result_content_for_api = [{"type": "text", "text": str(tool_execution_result_content)}]

                        print(f"[DEBUG AnthropicAdapter generate_response] Result from tool '{tool_name}' (ID: {tool_id}): {str(tool_result_content_for_api)[:100]}...")
                        
                        tool_results_for_api.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": tool_result_content_for_api 
                        })
                    except Exception as tool_exc:
                        print(f"[ERROR AnthropicAdapter generate_response] Exception during tool_handler for '{tool_name}': {tool_exc}")
                        traceback.print_exc()
                        tool_results_for_api.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": [{"type": "text", "text": f"Error: Tool handler for '{tool_name}' crashed: {str(tool_exc)}"}],
                            "is_error": True # Indicate this was an error during execution
                        })
                
                current_api_messages.append({"role": "user", "content": tool_results_for_api})
                # print(f"[DEBUG AnthropicAdapter generate_response] Appended user's tool_results turn to current_api_messages.")
                continue # Next iteration of the while loop

            # Handle tool_use without tool_handler (e.g., caching calls that shouldn't have tools)
            elif stop_reason == "tool_use" and tool_use_blocks and not tool_handler:
                print(f"[WARN AnthropicAdapter generate_response] LLM requested tool use but no tool_handler provided. Stopping.")
                return {
                    "text": current_turn_text,
                    "tool_calls": tool_use_blocks,
                    "raw_response": response_json,
                    "stop_reason": stop_reason,
                    "tokens": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                    "error": "tool_use_requested_but_no_handler"
                }

            # Not "tool_use" or no tool_use_blocks, this is the final response from this sequence
            print(f"[DEBUG AnthropicAdapter generate_response] Exiting tool loop. Final stop_reason: {stop_reason}")
            final_result = {
                "text": current_turn_text,
                "tool_calls": tool_use_blocks, # Tool *requests* from this final turn (should be empty if loop exited cleanly after tool use)
                "raw_response": response_json,
                "stop_reason": stop_reason,
                "tokens": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                "error": None
            }
            # print(f"[DEBUG AnthropicAdapter generate_response] Final result: {json.dumps(final_result, indent=2, default=str)}")
            return final_result