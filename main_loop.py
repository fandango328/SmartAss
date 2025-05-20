import asyncio
import config 
import json
import traceback
from typing import Dict, Any, List

# Single import from function_definitions with all three functions
from function_definitions import sanitize_messages_for_api, trim_chat_log, save_to_log_file

from secret import ANTHROPIC_API_KEY 
from laura_tools import tool_registry 
from llm_integrations.anthropic_adapter import AnthropicLLMAdapter
from token_manager import TokenManager


class MainLoop:
    def __init__(self, token_manager_instance: TokenManager, document_manager_instance: Any): # Type hint for TokenManager
        print("[DEBUG MainLoop __init__] Initializing MainLoop...")
        self.token_manager = token_manager_instance
        self.document_manager = document_manager_instance
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}

        if config.ACTIVE_PROVIDER == "anthropic":
            AdapterClass = AnthropicLLMAdapter
            model_name = config.ANTHROPIC_MODEL
        # elif config.ACTIVE_PROVIDER == "openai":
        #     AdapterClass = OpenAILLMAdapter
        #     model_name = config.OPENAI_MODEL
        else:
            raise ValueError(f"Unsupported provider in MainLoop: {config.ACTIVE_PROVIDER}")

        # Initialize the adapter instance
        self.llm_adapter = AdapterClass(
            api_key=ANTHROPIC_API_KEY, # Directly use imported key
            model=model_name,
            system_prompt=config.SYSTEM_PROMPT, # Default system prompt
            max_tokens=config.MAX_TOKENS,       # Default max tokens for a response
            temperature=config.TEMPERATURE,     # Default temperature
            tools=tool_registry.get_llm_tool_definitions()
        )
        print(f"[DEBUG MainLoop __init__] LLM Adapter initialized for {config.ACTIVE_PROVIDER} with model {model_name}")

    async def _tool_handler(self, function_name: str, function_args: Dict[str, Any], call_id: str) -> Any:
        """Handler for tool/function calls from the LLM, passed to the adapter."""
        print(f"[DEBUG MainLoop _tool_handler] Called for tool: {function_name}, ID: {call_id}, Args: {function_args}")
        handler = tool_registry.get_handler(function_name)
        if handler is None:
            print(f"[ERROR MainLoop _tool_handler] Unknown tool: {function_name}")
            return {"error": f"Unknown tool: {function_name}"}

        try:
            if not isinstance(function_args, dict):
                print(f"[WARN MainLoop _tool_handler] function_args not a dict, attempting to use as is or parse if string: {type(function_args)}")
                if isinstance(function_args, str):
                    try:
                        function_args = json.loads(function_args)
                    except json.JSONDecodeError:
                         print(f"[ERROR MainLoop _tool_handler] Failed to parse function_args string for {function_name}")
                         return {"error": f"Invalid JSON arguments for tool {function_name}"}

            import inspect
            if inspect.iscoroutinefunction(handler):
                result = await handler(**function_args)
            else:
                result = handler(**function_args)
            print(f"[DEBUG MainLoop _tool_handler] Result from {function_name}: {str(result)[:200]}") # Log snippet
            return result
        except Exception as e:
            print(f"[ERROR MainLoop _tool_handler] Tool {function_name} execution error: {e}")
            traceback.print_exc()
            return {"error": f"Tool {function_name} error: {e}"}

    async def process_input(self, input_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input through the LLM and return results.
        """
        print(f"\n[DEBUG MainLoop process_input] Received input_event: {input_event}")
        try:
            session_id = input_event.get("session_id")
            if not session_id:
                print("[ERROR MainLoop process_input] 'session_id' is missing from input_event.")
                return {"text": None, "error": "Session ID is missing"}

            input_type = input_event.get("type", "text")
            payload = input_event.get("payload", {})
            user_message_content = payload.get("text") if input_type == "text" else payload.get("transcript", "")

            if not user_message_content:
                print("[WARN MainLoop process_input] User message content is empty.")
                return {"text": "I didn't catch that. Could you please repeat?", "error": "Empty user message"}

            if session_id not in self.conversations:
                print(f"[DEBUG MainLoop process_input] New session, initializing conversation for: {session_id}")
                self.conversations[session_id] = []
            
            current_conversation = self.conversations[session_id]
            print(f"[DEBUG MainLoop process_input] Conversation history for {session_id} (before adding new user message): {len(current_conversation)} messages")

            new_user_message = {"role": "user", "content": user_message_content}
            current_conversation.append(new_user_message)
            print(f"[DEBUG MainLoop process_input] Added user message to conversation: {str(new_user_message)[:200]}")

            # Save the user message to log file
            save_to_log_file(new_user_message)

            # --- Step 2: Update calls to trim_chat_log ---
            # The trim_chat_log function will be modified to accept token_manager_instance and max_tokens_limit.
            current_conversation = trim_chat_log(current_conversation, self.token_manager, config.CHAT_LOG_MAX_TOKENS)
            self.conversations[session_id] = current_conversation 
            print(f"[DEBUG MainLoop process_input] Conversation history length after user message and trim: {len(current_conversation)}")

            messages_for_api = sanitize_messages_for_api(list(current_conversation)) 

            if self.document_manager and getattr(self.document_manager, "files_loaded", False):
                if hasattr(self.document_manager, "get_claude_message_blocks"):
                    document_blocks = self.document_manager.get_claude_message_blocks()
                    if document_blocks:
                        print(f"[DEBUG MainLoop process_input] Injecting {len(document_blocks)} document blocks.")
                        if messages_for_api and messages_for_api[-1]["role"] == "user":
                            user_content = messages_for_api[-1].get("content", "")
                            user_content_blocks = []
                            if isinstance(user_content, list): 
                                user_content_blocks.extend([item for item in user_content if isinstance(item, dict)])
                            elif isinstance(user_content, str): 
                                user_content_blocks.append({"type": "text", "text": user_content})
                            
                            messages_for_api[-1]["content"] = user_content_blocks + document_blocks
                        else: 
                            messages_for_api.append({"role": "user", "content": document_blocks})
                        print(f"[DEBUG MainLoop process_input] Messages for API after document injection (last message content type): {type(messages_for_api[-1]['content'] if messages_for_api else None)}")

            if not messages_for_api:
                print("[ERROR MainLoop process_input] No valid messages to send to LLM API after sanitization/doc injection.")
                return {"text": None, "error": "No valid messages for LLM"}

            relevant_tools_for_call = []
            tool_choice_for_call = None
            if self.token_manager.tools_are_active(): 
                relevant_tools_for_call = self.token_manager.get_tools_for_query(user_message_content)
                if relevant_tools_for_call:
                    tool_choice_for_call = {"type": "auto"}
                    print(f"[DEBUG MainLoop process_input] Using {len(relevant_tools_for_call)} relevant tools for this call.")
                else:
                    print("[DEBUG MainLoop process_input] Tools are active, but no specific relevant tools found for query.")
            else:
                print("[DEBUG MainLoop process_input] Tools are not active for this call.")

            print(f"[DEBUG MainLoop process_input] Calling llm_adapter.generate_response with:")
            print(f"  messages (count): {len(messages_for_api)}")
            print(f"  system_prompt: {str(config.SYSTEM_PROMPT)[:100]}...")
            print(f"  tools (count): {len(relevant_tools_for_call)}")
            print(f"  tool_choice: {tool_choice_for_call}")

            llm_api_result = await self.llm_adapter.generate_response(
                messages=messages_for_api,
                tool_handler=self._tool_handler, 
                system_prompt=config.SYSTEM_PROMPT, 
                tools=relevant_tools_for_call,
                tool_choice=tool_choice_for_call, 
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE
            )
            print(f"[DEBUG MainLoop process_input] Full result from llm_adapter.generate_response: {str(llm_api_result)[:500]}")

            assistant_response_text = llm_api_result.get("text")
            if llm_api_result.get("error"):
                print(f"[WARN MainLoop process_input] LLM adapter returned an error: {llm_api_result.get('error')}")
            elif assistant_response_text is not None: 
                current_conversation.append({"role": "assistant", "content": assistant_response_text})
                print(f"[DEBUG MainLoop process_input] Added assistant response to conversation: {str(assistant_response_text)[:200]}")
                
                # --- Step 2 (repeated): Update calls to trim_chat_log ---
                current_conversation = trim_chat_log(
                current_conversation,
                self.token_manager,
                config.CHAT_LOG_MAX_TOKENS
            )
                self.conversations[session_id] = current_conversation
                print(f"[DEBUG MainLoop process_input] Conversation history length after assistant response and trim: {len(current_conversation)}")
            else:
                print("[WARN MainLoop process_input] LLM adapter returned no 'text' and no 'error'. This might be unusual.")

            print(f"[DEBUG MainLoop process_input] Final result being returned to orchestrator: {str(llm_api_result)[:500]}")
            return llm_api_result

        except Exception as e:
            print(f"[ERROR MainLoop process_input] Unhandled exception: {e}")
            traceback.print_exc()
            return {"text": None, "error": f"Critical error in main_loop: {str(e)}"}
