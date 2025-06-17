#!/usr/bin/env python3

"""
Main Loop Processing Module for LAURA Voice Assistant
Handles the core conversation loop with tool support and context management.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import os
import re
import traceback

# Configuration imports
from config import (
    ACTIVE_PERSONA,
    ANTHROPIC_MODEL,
    SYSTEM_PROMPT,
    VOICE,
    ELEVENLABS_MODEL,
    CHAT_LOG_MAX_TOKENS,
)

# Core imports
from secret import ANTHROPIC_API_KEY, ELEVENLABS_KEY
from llm_integrations.anthropic_adapter import AnthropicLLMAdapter #type: ignore
from tool_registry import tool_registry
from system_manager import SystemManager
from function_definitions import (
    save_to_log_file,
    load_recent_context,
    get_current_time,
    get_location,
    sanitize_messages_for_api
)
from laura_tools import get_llm_tool_definitions

class MainLoop:
    """Main conversation loop handler with tool support"""
    
    def __init__(self, token_manager_instance=None, document_manager_instance=None):
        """Initialize MainLoop with required managers"""
        print("[DEBUG MainLoop __init__] Initializing MainLoop...")
        
        # Store manager instances
        self.token_manager = token_manager_instance
        self.document_manager = document_manager_instance
        
        # Initialize system manager (but don't register managers yet - that's async)
        self.system_manager = SystemManager(
            document_manager=self.document_manager,
            token_manager=self.token_manager
        )
        
        # Load conversation history
        print("[DEBUG MainLoop __init__] Loading chat history into runtime memory...")
        
        # Load recent context - always use token limiting
        self.chat_log = load_recent_context(
            token_manager=self.token_manager,
            token_limit=CHAT_LOG_MAX_TOKENS
        )
        if not self.chat_log:
            # Only use fallback if load_recent_context completely fails
            self.chat_log = self._load_basic_chat_history()
        
        # Log startup context info
        if self.chat_log:
            print(f"[DEBUG MainLoop __init__] === STARTUP CONTEXT LOADING ===")
            print(f"[DEBUG MainLoop __init__] Loaded {len(self.chat_log)} messages into runtime memory")
            
            # Calculate total tokens if token manager available
            if self.token_manager:
                try:
                    # Use simple estimation during init, proper counting happens in async init
                    from function_definitions import estimate_tokens
                    total_tokens = sum(estimate_tokens(msg) for msg in self.chat_log)
                    print(f"[DEBUG MainLoop __init__] Total estimated tokens: {total_tokens}")
                except Exception as e:
                    print(f"[DEBUG MainLoop __init__] Error counting tokens: {e}")
            
            # Show first and last messages
            if len(self.chat_log) > 0:
                first_msg = self.chat_log[0]
                last_msg = self.chat_log[-1]
                print(f"[DEBUG MainLoop __init__] Oldest message ({first_msg['role']}): '{str(first_msg['content'])[:50]}...'")
                print(f"[DEBUG MainLoop __init__] Newest message ({last_msg['role']}): '{str(last_msg['content'])[:50]}...'")
            
            print(f"[DEBUG MainLoop __init__] === RUNTIME CHATLOG READY ===")
        else:
            print("[DEBUG MainLoop __init__] No previous conversation history loaded")
        
        # Initialize LLM adapter
        self.llm_adapter = AnthropicLLMAdapter(
            api_key=ANTHROPIC_API_KEY,
            model=ANTHROPIC_MODEL,
            tools=get_llm_tool_definitions(),
            system_prompt=SYSTEM_PROMPT,
            max_tokens=1024,
            temperature=0.7
        )
        print(f"[DEBUG MainLoop __init__] LLM Adapter initialized for anthropic with model {ANTHROPIC_MODEL}")
        
        # Cache control state
        self._documents_cached = False
        
        # Flag to track async initialization
        self._async_initialized = False

    async def initialize_async(self):
        """
        Perform async initialization that requires await.
        This includes registering all managers with the system manager.
        """
        if self._async_initialized:
            print("[DEBUG MainLoop initialize_async] Already initialized, skipping.")
            return
            
        print("[DEBUG MainLoop initialize_async] Starting async initialization...")
        
        # Register all managers with system manager
        if self.system_manager:
            print("[DEBUG MainLoop initialize_async] Registering managers with system manager...")
            await self.system_manager.register_managers()
            print("[DEBUG MainLoop initialize_async] Manager registration complete.")
        else:
            print("[ERROR MainLoop initialize_async] No system manager available!")
            
        self._async_initialized = True
        print("[DEBUG MainLoop initialize_async] Async initialization complete.")

    def _load_basic_chat_history(self) -> List[Dict[str, Any]]:
        """Fallback method to load chat history without token counting"""
        try:
            from config import CHAT_LOG_DIR
            log_dir = Path(CHAT_LOG_DIR)
            if not log_dir.exists():
                return []
                
            # Get today's log file
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = log_dir / f"chat_log_{today}.json"
            
            if not log_file.exists():
                return []
                
            with open(log_file, "r") as f:
                logs = json.load(f)
                
            # Convert to message format
            messages = []
            for entry in logs:
                if "role" in entry and "content" in entry:
                    messages.append({
                        "role": entry["role"],
                        "content": entry["content"]
                    })
                    
            # Take last 20 messages
            return messages[-20:] if len(messages) > 20 else messages
            
        except Exception as e:
            print(f"[ERROR MainLoop] Failed to load basic chat history: {e}")
            return []

    async def process_input(self, input_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input event and generate response.
        
        Args:
            input_event: Dictionary containing session_id, type, and payload with user text
            
        Returns:
            Dict containing response text and any metadata
        """
        if not self._async_initialized:
            # Try to initialize if not done yet
            await self.initialize_async()
        
        print(f"\n[MainLoop process_input] Processing: {input_event}")
        
        try:
            # Extract user text from the input event
            user_input = ""
            if isinstance(input_event, dict):
                # Handle dictionary format from orchestrator
                payload = input_event.get("payload", {})
                user_input = payload.get("text", "")
                
                # Fallback: check if text is at top level
                if not user_input:
                    user_input = input_event.get("text", "")
            elif isinstance(input_event, str):
                # Handle direct string input (backward compatibility)
                user_input = input_event
            else:
                print(f"[MainLoop] Warning: Unexpected input type: {type(input_event)}")
                return {
                    "error": "Invalid input format",
                    "text": "I received an invalid input format. Please try again."
                }
            
            if not user_input:
                print("[MainLoop] Warning: No user text found in input")
                return {
                    "error": "Empty input",
                    "text": "I didn't receive any text to process. Please try again."
                }
            
            print(f"[MainLoop] Extracted user text: '{user_input}'")
            
            # Save user input to log
            user_message = {"role": "user", "content": user_input}
            save_to_log_file(user_message)
            
            # Check for system commands
            is_command, cmd_type, action, args = self.system_manager.detect_command(user_input)
            if is_command:
                print(f"[MainLoop] System command detected: {cmd_type} - {action}")
                success = await self.system_manager.handle_command(cmd_type, action, args)
                return {
                    "text": f"Command {action} {'completed' if success else 'failed'}.",
                    "command_executed": True
                }
            
            # Check if tools are enabled
            if not self.token_manager.tools_enabled:
                print("[MainLoop] Tools disabled, using conversation-only mode")
                tools_for_call = None
                tool_choice = None
            else:
                tools_for_call = self.llm_adapter.base_tools
                tool_choice = {"type": "auto"}
            
            # Build messages for API with document caching
            messages_for_api = await self._build_messages_with_caching(user_input)
            messages_for_api = sanitize_messages_for_api(messages_for_api)
            
            if len(messages_for_api) > 80:
                for i in range(max(0, 80), min(len(messages_for_api), 90)):
                    msg = messages_for_api[i]
                    content_type = type(msg.get("content", ""))
                    content_preview = str(msg.get("content", ""))[:100]
                    print(f"[DEBUG] Message {i}: role={msg.get('role')}, content_type={content_type}, preview='{content_preview}...'")
            
            # Make API call
            response = await self.llm_adapter.generate_response(
                messages=messages_for_api,
                system_prompt=SYSTEM_PROMPT,
                tools=tools_for_call,
                tool_choice=tool_choice,
                tool_handler=self._handle_tool_execution if self.token_manager.tools_enabled else None,
                max_tokens=1024,
                temperature=0.7
            )
            
            # Process response
            assistant_text = response.get("text", "")
            if response.get("error"):
                print(f"[MainLoop] API error: {response['error']}")
                return {
                    "error": response["error"],
                    "text": "I encountered an error processing your request."
                }
            
            # Update conversation history - only append clean text content to runtime chat log
            self.chat_log.append({"role": "user", "content": user_input})
            if assistant_text:  # Only append if we have actual text content
                self.chat_log.append({"role": "assistant", "content": assistant_text})

            # Save to persistent log files separately  
            user_message = {"role": "user", "content": user_input}
            save_to_log_file(user_message)
            if assistant_text:
                assistant_message = {"role": "assistant", "content": assistant_text}
                save_to_log_file(assistant_message)

            # Check cache performance
            if response.get("tokens"):
                cache_read = response["tokens"].get("cache_read_input_tokens", 0)
                if cache_read > 0:
                    print(f"[MainLoop] Cache hit! Read {cache_read} tokens from cache")
            
            return {"text": assistant_text}
            
        except Exception as e:
            print(f"[MainLoop] Error in process_input: {e}")
            traceback.print_exc()
            return {
                "error": str(e),
                "text": "I encountered an unexpected error. Please try again."
            }

    async def _build_messages_with_caching(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Build message list with proper document caching support.
        
        Args:
            user_input: Current user message
            
        Returns:
            List of messages formatted for API
        """
        messages = []
        
        # Check if we need to setup cached documents
        need_document_setup = False
        if not hasattr(self, '_documents_cached') or not self._documents_cached:
            if self.document_manager and self.document_manager.files_loaded:
                need_document_setup = True
        
        # If documents need caching, set them up first
        if need_document_setup:
            print("[MainLoop] Setting up document cache...")
            doc_blocks = self.document_manager.get_all_loaded_document_blocks_for_claude()
            
            if doc_blocks:
                # Create content blocks with cache_control on last block
                content_blocks = []
                
                # Add intro text
                content_blocks.append({
                    "type": "text",
                    "text": "I'm providing you with the following documents for reference:"
                })
                
                # Add all document blocks
                for i, block in enumerate(doc_blocks):
                    if i == len(doc_blocks) - 1:
                        # Add cache_control to the last block
                        if block.get("type") == "text":
                            block["cache_control"] = {"type": "ephemeral"}
                        elif block.get("type") in ["image", "document"]:
                            block["cache_control"] = {"type": "ephemeral"}
                    content_blocks.append(block)
                
                # Add document setup as first exchange
                messages.append({
                    "role": "user",
                    "content": content_blocks
                })
                messages.append({
                    "role": "assistant",
                    "content": "I've received the documents and I'm ready to help you with them."
                })
                
                self._documents_cached = True
                print(f"[MainLoop] Cached {len(doc_blocks)} document blocks")
        
        # Add conversation history
        messages.extend(self.chat_log)
        
        # Add current user message
        messages.append({"role": "user", "content": user_input})
        
        return messages

    async def _handle_tool_execution(self, tool_name: str, tool_args: Dict[str, Any], tool_id: str) -> str:
        """
        Execute a tool and return the result.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            tool_id: Unique identifier for this tool call
            
        Returns:
            String result from tool execution
        """
        print(f"[MainLoop] Executing tool: {tool_name}")
        
        try:
            # Show tool use in UI if system manager available
            if self.system_manager and hasattr(self.system_manager, 'show_tool_use'):
                await self.system_manager.show_tool_use()
            
            # Get tool handler
            handler = tool_registry.get_handler(tool_name)
            if not handler:
                return f"Error: Unknown tool '{tool_name}'"
            
            # Execute tool
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**tool_args)
            else:
                result = handler(**tool_args)
            
            # Record tool usage
            if self.token_manager:
                self.token_manager.record_tool_usage(tool_name)
            
            return str(result)
            
        except Exception as e:
            print(f"[MainLoop] Tool execution error: {e}")
            traceback.print_exc()
            return f"Error executing {tool_name}: {str(e)}"

    def reset_document_cache(self):
        """Reset document cache state when documents are cleared."""
        self._documents_cached = False
        print("[MainLoop] Document cache state reset")

    def _get_documents_hash(self) -> str:
        """Generate a hash of current document state for change detection."""
        if not self.document_manager or not self.document_manager.files_loaded:
            return "no_documents"
        
        # Create a simple hash based on loaded files
        import hashlib
        doc_state = f"{len(self.document_manager.loaded_files)}:{','.join(sorted(self.document_manager.loaded_files.keys()))}"
        return hashlib.md5(doc_state.encode()).hexdigest()
