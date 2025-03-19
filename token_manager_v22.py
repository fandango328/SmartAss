#!/usr/bin/env python3

"""
TokenManager for SmartAss Voice Assistant
Last Updated: 2025-03-16 20:08:25

Purpose:
    Manages token usage, costs, and tool state for the SmartAss voice assistant,
    with specific handling of tool-related overheads and model-specific pricing.

Key Features:
    - Token counting and cost calculation for different Claude models
    - Tool usage and definition overhead tracking
    - Simplified binary tool state with auto-disable after 3 non-tool queries
    - Voice-optimized command phrases for reliable tool state management
    - Session-based cost accumulation
"""
import os
import json
import glob
from config_og import CHAT_LOG_MAX_TOKENS, CHAT_LOG_RECOVERY_TOKENS, CHAT_LOG_DIR
from datetime import datetime
from decimal import Decimal
from typing import Dict, Tuple, Optional, Any, List
from config_transcription_v4 import ANTHROPIC_MODEL  # Direct import from config
from laura_tools import AVAILABLE_TOOLS, get_tool_by_name, get_tools_by_category

class TokenManager:
    """
    Manages token counting, cost calculation, and tool usage authorization for SmartAss.
    
    Features:
    - Binary tool state (enabled/disabled)
    - Auto-disable tools after 3 consecutive non-tool queries
    - Phonetically optimized voice commands for tool state changes
    - Mixed command resolution (prioritizing last mentioned command)
    - Token and cost tracking for API interactions
    """

    # Model-specific costs per million tokens (MTok)
    MODEL_COSTS = {
        "claude-3-5-sonnet-20241022": {
            "input": 3.00,     # $3.00 per 1M input tokens = $0.000003 per token
            "output": 15.00    # $15.00 per 1M output tokens = $0.000015 per token
        },
        "claude-3-7-sonnet-20240229": {
            "input": 3.00,     # $3.00 per 1M input tokens = $0.000003 per token
            "output": 15.00    # $15.00 per 1M output tokens = $0.000015 per token
        },
        "claude-3-opus-20240229": {
            "input": 15.00,    # $15.00 per 1M input tokens = $0.000015 per token
            "output": 75.00    # $75.00 per 1M output tokens = $0.000075 per token
        },
        "claude-3-5-haiku-20241022": {
            "input": 0.80,     # $0.80 per 1M input tokens = $0.0000008 per token
            "output": 4.00     # $4.00 per 1M output tokens = $0.000004 per token
        }
    }

    # Token overheads for different operations
    TOOL_COSTS = {
        "definition_overhead": 2600,  # Base tokens for full tool definitions JSON
        "usage_overhead": {
            "auto": 346,     # When tool_choice="auto"
            "any": 313,      # When tool_choice allows any tool
            "tool": 313,     # Specific tool usage
            "none": 0        # No tools used
        }
    }

    # Voice-optimized tool enabling phrases
    # These phrases were specifically selected to have distinctive phonetic patterns
    # that are less likely to be confused by voice transcription systems like VOSK
    TOOL_ENABLE_PHRASES = {
        # Primary commands (most distinctive)
        "tools activate", "launch toolkit", "begin assistance", "enable tool use",
        "start tools", "enable assistant", "tools online", "enable tools",
        
        # Additional distinctive commands
        "assistant power up", "toolkit online", "helper mode active",
        "utilities on", "activate functions", "tools ready",
        
        # Short commands with distinctive sounds
        "tools on", "toolkit on", "functions on",
        
        # Commands with unique phonetic patterns
        "wake up tools", "prepare toolkit", "bring tools online"
    }

    # Voice-optimized tool disabling phrases
    # Selected for clear phonetic distinction from enabling phrases and from
    # common conversation patterns to minimize false positives/negatives
    TOOL_DISABLE_PHRASES = {
        # Primary commands (most distinctive)
        "tools offline", "end toolkit", "close assistant",
        "stop tools", "disable assistant", "conversation only",
        
        # Additional distinctive commands
        "assistant power down", "toolkit offline", "helper mode inactive",
        "utilities off", "deactivate functions", "tools away",
        
        # Short commands with distinctive sounds
        "tools off", "toolkit off", "functions off",
        
        # Commands with unique phonetic patterns
        "sleep tools", "dismiss toolkit", "take tools offline"
    }

    # Tool category keywords for contextual tool detection
    TOOL_CATEGORY_KEYWORDS = {
        'EMAIL': ['email', 'mail', 'send', 'write', 'compose', 'draft'],
        'CALENDAR': ['calendar', 'schedule', 'event', 'meeting', 'appointment'],
        'TASK': ['task', 'todo', 'reminder', 'checklist', 'to-do'],
        'UTILITY': ['time', 'location', 'calibrate', 'voice', 'settings'],
        'CONTACT': ['contact', 'person', 'people', 'address', 'phone']
    }

    def __init__(self, anthropic_client):
        """
        Initialize TokenManager with streamlined session tracking.
        
        Args:
            anthropic_client: Anthropic API client instance
            
        Raises:
            TypeError: If anthropic_client is None
        """
        # CHANGE: Simplified client validation - removed MODEL_COSTS check since we're using native token counting
        if anthropic_client is None:
            raise TypeError("anthropic_client cannot be None")
            
        # KEEP: Core client and model settings
        self.anthropic_client = anthropic_client
        self.query_model = ANTHROPIC_MODEL
        self.tool_model = "claude-3-5-sonnet-20241022"

        # KEEP: Session state tracking
        self.session_active = False
        self.session_start_time = None
        self.last_interaction_time = None
        
        # NEW: Combined session statistics
        self.current_session = {
            'input_tokens': 0,
            'output_tokens': 0,
            'tool_tokens': 0,
            'total_cost': Decimal('0.00')
        }
        
        # NEW: Initialize all cost tracking variables
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_query_costs = Decimal('0.00')
        self.total_tool_usage_costs = Decimal('0.00')
        self.total_tool_definition_costs = Decimal('0.00')
        
        # KEEP: Tool state management
        self.tools_enabled = False
        self.tools_used_in_session = False
        self.consecutive_non_tool_queries = 0
        self.tool_disable_threshold = 3
        self.last_tool_use = None

    def prepare_messages_for_token_count(self, current_query: str, chat_log: list, system_prompt: str = None) -> tuple:
        """
        Returns tuple of (messages, system_prompt) for API compliance
        """
        print("\n=== Token Count Message Preparation ===")
        print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        formatted_messages = []
        
        try:
            from config_og import SYSTEM_PROMPT
            system_content = (system_prompt or SYSTEM_PROMPT).strip()
            print(f"\nSystem Prompt: {len(system_content)} chars (separate parameter)")
        except Exception as e:
            print(f"Warning: System prompt loading error: {e}")
            system_content = ""
        
        if chat_log and isinstance(chat_log, list):
            print(f"\nFormatting {len(chat_log)} chat messages")
            for msg in chat_log:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    formatted_messages.append({
                        "role": msg['role'],
                        "content": [{"type": "text", "text": str(msg['content'])}]
                    })
        
        formatted_messages.append({
            "role": "user",
            "content": [{"type": "text", "text": current_query}]
        })
        print(f"Current Query: {len(current_query)} chars added")
        
        print(f"\nTotal formatted messages: {len(formatted_messages)}")
        print("Message types:")
        role_counts = {}
        for msg in formatted_messages:
            role_counts[msg['role']] = role_counts.get(msg['role'], 0) + 1
        for role, count in role_counts.items():
            print(f"  {role}: {count}")
        
        print("=== End Message Preparation ===\n")
        return formatted_messages, system_content



    def count_message_tokens(self, current_query: str, chat_log: list = None, system_prompt: str = None) -> int:
        """
        Count tokens with API-compliant structure
        """
        try:
            print("\n=== Token Count Operation ===")
            messages, system_content = self.prepare_messages_for_token_count(
                current_query=current_query,
                chat_log=chat_log or [],
                system_prompt=system_prompt
            )
            
            print(f"Calling API with model: {self.query_model}")
            count_result = self.anthropic_client.messages.count_tokens(
                model=self.query_model,
                messages=messages,
                system=system_content
            )
            
            print(f"API Response Type: {type(count_result)}")
            print(f"API Response: {count_result}")
            
            if isinstance(count_result, (int, float)):
                token_count = int(count_result)
            elif isinstance(count_result, dict) and "input_tokens" in count_result:
                token_count = count_result["input_tokens"]
            else:
                print(f"WARNING: Unexpected response format")
                return 0
                
            print(f"Final Token Count: {token_count}")
            print("=== End Token Count ===\n")
            return token_count
            
        except Exception as e:
            print(f"ERROR in token counting: {str(e)}")
            traceback.print_exc()
            return 0
        
    def start_session(self):
        """
        Start a new token tracking session.
        Resets all session statistics and sets the start time.
        Called once at the beginning of the SmartAss application lifecycle.
        """
        self.session_active = True
        self.session_start_time = datetime.now()
        self.last_interaction_time = self.session_start_time
        
        # Reset session statistics
        self.current_session = {
            'input_tokens': 0,
            'output_tokens': 0,
            'tool_tokens': 0,
            'total_cost': Decimal('0.00')
        }
        
        # Reset cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_query_costs = Decimal('0.00')
        self.total_tool_usage_costs = Decimal('0.00')
        self.total_tool_definition_costs = Decimal('0.00')
        
        # Reset tool state for new session
        self.tools_used_in_session = False
        
        print(f"Token tracking session started at {self.session_start_time}")
        return {
            "status": "started",
            "timestamp": self.session_start_time.isoformat()
        }

    def display_token_usage(self, query_tokens: int = None):
        """
        Real-time token usage display with current query and session stats.
        
        Args:
            query_tokens: Number of tokens in current query (optional)
        """
        print("\n┌─ Token Usage Report " + "─" * 40)
        
        # Display current query tokens if provided
        if query_tokens:
            print(f"│ Current Query Tokens: {query_tokens:,}")
            print("│" + "─" * 50)
            
        # Display session statistics
        print(f"│ Session Statistics:")
        print(f"│   Input Tokens:  {self.current_session['input_tokens']:,}")
        print(f"│   Output Tokens: {self.current_session['output_tokens']:,}")
        print(f"│   Tool Tokens:   {self.current_session['tool_tokens']:,}")
        print(f"│   Total Cost:    ${self.current_session['total_cost']:.4f}")
        
        # Show session duration if active
        if self.session_start_time:
            duration = datetime.now() - self.session_start_time
            print(f"│   Duration:      {duration.total_seconds():.1f}s")
        
        # Show tool status if enabled
        if self.tools_enabled:
            print(f"│   Tools:         Enabled (Used: {self.tools_used_in_session})")
            
        print("└" + "─" * 50)

    def start_interaction(self):
        """
        Start tracking a new user interaction.
        Called at the beginning of each interaction to track timing.
        """
        self.last_interaction_time = datetime.now()


    def update_session_costs(self, input_tokens: int, output_tokens: int, 
                           is_tool_use: bool = False):
        """
        Update session costs with token usage information.
        
        Args:
            input_tokens: Number of input tokens in this interaction
            output_tokens: Number of output tokens in this interaction
            is_tool_use: Whether this interaction involved tool usage
        """
        try:
            # Check if session is active
            if not hasattr(self, 'session_active') or not self.session_active:
                # Auto-start session if not active
                if hasattr(self, 'start_session'):
                    self.start_session()
                else:
                    print("Warning: Session not active and no start_session method available")
                    
            # Determine which model to use for pricing
            model = self.tool_model if is_tool_use else self.query_model
            
            # Ensure we have valid token counts
            current_input = max(0, input_tokens)
            current_output = max(0, output_tokens)
            
            # Add tool definition overhead if tools are enabled
            tool_definition_overhead = 0
            if hasattr(self, 'tools_enabled') and self.tools_enabled and hasattr(self, 'TOOL_COSTS'):
                if isinstance(self.TOOL_COSTS, dict) and "definition_overhead" in self.TOOL_COSTS:
                    tool_definition_overhead = self.TOOL_COSTS["definition_overhead"]
                    current_input += tool_definition_overhead
                
            # Add tool usage overhead if applicable
            tool_usage_overhead = 0
            if is_tool_use and hasattr(self, 'TOOL_COSTS') and isinstance(self.TOOL_COSTS, dict):
                if "usage_overhead" in self.TOOL_COSTS and isinstance(self.TOOL_COSTS["usage_overhead"], dict):
                    if "tool" in self.TOOL_COSTS["usage_overhead"]:
                        tool_usage_overhead = self.TOOL_COSTS["usage_overhead"]["tool"]
                        current_input += tool_usage_overhead
            
            # Calculate costs (ensuring we get Decimals)
            update_input_cost = Decimal('0')
            update_output_cost = Decimal('0')
            
            if hasattr(self, 'calculate_token_cost'):
                update_input_cost = self.calculate_token_cost(model, "input", current_input)
                update_output_cost = self.calculate_token_cost(model, "output", current_output)
            
            # Ensure we have Decimal objects
            if not isinstance(update_input_cost, Decimal):
                update_input_cost = Decimal(str(update_input_cost))
            if not isinstance(update_output_cost, Decimal):
                update_output_cost = Decimal(str(update_output_cost))
            
            # Update class-level token counts
            if hasattr(self, 'total_input_tokens'):
                self.total_input_tokens += current_input
            if hasattr(self, 'total_output_tokens'):
                self.total_output_tokens += current_output
            
            # Update current_session dictionary
            if hasattr(self, 'current_session') and isinstance(self.current_session, dict):
                if 'input_tokens' in self.current_session:
                    self.current_session['input_tokens'] += input_tokens
                if 'output_tokens' in self.current_session:
                    self.current_session['output_tokens'] += output_tokens
                
                if is_tool_use and 'tool_tokens' in self.current_session:
                    self.current_session['tool_tokens'] += input_tokens + output_tokens
                    
                if 'total_cost' in self.current_session:
                    total_update_cost = update_input_cost + update_output_cost
                    self.current_session['total_cost'] += total_update_cost
            
            # Update tool/query cost tracking
            if is_tool_use and hasattr(self, 'total_tool_usage_costs'):
                self.total_tool_usage_costs += update_input_cost + update_output_cost
            elif hasattr(self, 'total_query_costs'):
                self.total_query_costs += update_input_cost + update_output_cost
                
            # Add tool definition costs if applicable
            if tool_definition_overhead > 0 and hasattr(self, 'total_tool_definition_costs'):
                tool_definition_cost = Decimal('0')
                if hasattr(self, 'calculate_token_cost'):
                    tool_definition_cost = self.calculate_token_cost(model, "input", tool_definition_overhead)
                self.total_tool_definition_costs += tool_definition_cost
                
            # Automatically display current usage statistics
            if hasattr(self, 'display_token_usage'):
                self.display_token_usage(input_tokens + output_tokens)
                
        except Exception as e:
            print(f"Error updating session costs: {e}")

    def calculate_token_cost(self, model: str, token_type: str, token_count: int) -> Decimal:
        """
        Calculate cost for given token count based on model and token type.
        
        Args:
            model: The model identifier (e.g., "claude-3-5-sonnet")
            token_type: Either "input" or "output"
            token_count: Number of tokens to calculate cost for
            
        Returns:
            Decimal cost for the tokens
        """
        try:
            # First handle negative token counts
            if token_count < 0:
                print(f"Warning: Negative token count ({token_count}) adjusted to 0")
                token_count = 0
            
            # Use fallback costs for unknown models
            if model not in self.MODEL_COSTS:
                print(f"Warning: Unknown model '{model}', using claude-3-5-sonnet-20241022 pricing")
                model = "claude-3-5-sonnet-20241022"  # Fall back to a known model
                
            # Use input costs if token_type is unknown
            if token_type not in self.MODEL_COSTS[model]:
                print(f"Warning: Unknown token type '{token_type}' for model '{model}', using 'input' pricing")
                token_type = "input"  # Fall back to input pricing
                
            # Get the per-million rate and ensure it's a Decimal
            per_million_rate = self.MODEL_COSTS[model][token_type]
            if not isinstance(per_million_rate, Decimal):
                per_million_rate = Decimal(str(per_million_rate))
                
            # Calculate the actual cost
            return (per_million_rate / Decimal('1000000')) * Decimal(str(token_count))
            
        except Exception as e:
            print(f"Error calculating token cost: {e}")
            return Decimal('0')  # Safe fallback

    def log_api_interaction(self, interaction_type: str, query_text: str):
        """
        Log details about an API interaction.
        
        Args:
            interaction_type: Type of interaction (e.g., "query", "tool_use")
            query_text: The text of the query
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"API Interaction [{timestamp}] - Type: {interaction_type}, Query: {query_text[:50]}...")

    def log_error(self, error_message: str):
        """
        Log an error that occurred during processing.
        
        Args:
            error_message: Description of the error
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Token Manager Error [{timestamp}]: {error_message}")
        
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the current session.
        
        Returns:
            Dictionary with session statistics
        """
        session_duration = datetime.now() - self.session_start_time
        minutes = session_duration.total_seconds() / 60
        
        total_cost = self.total_query_costs + self.total_tool_usage_costs + self.total_tool_definition_costs
        
        return {
            "session_duration_minutes": round(minutes, 2),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": float(total_cost),
            "tools_currently_enabled": self.tools_enabled,
            "tools_used_in_session": self.tools_used_in_session
        }

    def enable_tools(self, query: str = None) -> Dict[str, Any]:
        """
        Enable tools mode and reset the non-tool query counter.
        
        Args:
            query: Optional query text that triggered tool enabling
            
        Returns:
            Dictionary with state information and message
        """
        self.tools_enabled = True
        self.consecutive_non_tool_queries = 0
        self.last_state_change_time = datetime.now()
        
        # Debugging print statements
        print(f"DEBUG: Tools enabled at {self.last_state_change_time}")
        if query:
            print(f"DEBUG: Enabled by query: {query[:50]}...")
            
        # Return state information
        return {
            "state": "enabled",
            "message": "Tools are now enabled. What would you like help with?",
            "tools_active": True
        }

    def disable_tools(self, reason: str = "manual") -> Dict[str, Any]:
        """
        Disable tools mode and reset related counters.
        
        Args:
            reason: Reason for disabling tools ('manual', 'auto', 'timeout', etc.)
            
        Returns:
            Dictionary with state information and message
        """
        was_enabled = self.tools_enabled
        self.tools_enabled = False
        self.consecutive_non_tool_queries = 0
        self.last_state_change_time = datetime.now()
        
        # Debugging print statements
        print(f"DEBUG: Tools disabled at {self.last_state_change_time} (Reason: {reason})")
        
        message = "Tools are now disabled." if was_enabled else "Tools are already disabled."
        if reason == "auto":
            message = "Tools have been automatically disabled after 3 queries without tool usage."
            
        # Return state information
        return {
            "state": "disabled",
            "message": message,
            "tools_active": False
        }

    def record_tool_usage(self, tool_name: str) -> Dict[str, Any]:
        """
        Record that a tool was used in the current session and reset the counter.
        
        Args:
            tool_name: Name of the tool that was used
        
        Returns:
            Dict with status information
        """
        self.tools_used_in_session = True
        self.consecutive_non_tool_queries = 0
        self.last_tool_usage_time = datetime.now()
        
        print(f"Tool usage recorded: {tool_name}")
        print(f"Non-tool query counter reset to 0")
        
        return {
            "status": "recorded",
            "tool": tool_name,
            "reset_counter": True
        }

    def track_query_completion(self, used_tool: bool = False) -> Dict[str, Any]:
        """
        Track completion of a query and update tool state if needed.
        Should be called after each query is processed.
        
        Args:
            used_tool: Whether a tool was used in this query
                
        Returns:
            Dict with standardized keys:
                - state_changed (bool): Whether the tools state changed
                - tools_active (bool): Current state of tools
                - queries_remaining (int): Queries left before auto-disable
                - message (str, optional): Status message if state changed
        """
        # If tools are not enabled, return standardized structure
        if not self.tools_enabled:
            return {
                "state_changed": False,
                "tools_active": False,
                "queries_remaining": 0
            }
                
        # If a tool was used, we've already reset the counter in record_tool_usage()
        if used_tool:
            return {
                "state_changed": False,
                "tools_active": True,
                "queries_remaining": self.tool_disable_threshold
            }
                
        # No tool was used, increment counter
        self.consecutive_non_tool_queries += 1
        print(f"Non-tool query counter increased to {self.consecutive_non_tool_queries}")
            
        # Check if we should auto-disable tools
        if self.consecutive_non_tool_queries >= self.tool_disable_threshold:
            result = self.disable_tools(reason="auto")
            return {
                "state_changed": True, 
                "tools_active": False,
                "queries_remaining": 0,
                "message": result["message"]
            }
                
        # Tools still enabled but counter increased
        queries_remaining = self.tool_disable_threshold - self.consecutive_non_tool_queries
        return {
            "state_changed": False, 
            "tools_active": True,
            "queries_remaining": queries_remaining
        }

    def tools_are_active(self) -> bool:
        """
        Check if tools are currently active.
        
        Returns:
            Boolean indicating if tools are enabled
        """
        return self.tools_enabled

    def handle_tool_command(self, query: str) -> Tuple[bool, Optional[str]]:
        try:
            query_lower = query.lower()
            
            # Check for both command types
            has_enable_command = any(phrase in query_lower for phrase in self.TOOL_ENABLE_PHRASES)
            has_disable_command = any(phrase in query_lower for phrase in self.TOOL_DISABLE_PHRASES)
            
            # Handle the conflict case (mixed commands)
            if has_enable_command and has_disable_command:
                print(f"Mixed tool commands detected in: {query_lower}")
                
                # Find the position of the last occurring command using rfind
                last_enable_pos = max((query_lower.rfind(phrase) for phrase in self.TOOL_ENABLE_PHRASES), default=-1)
                last_disable_pos = max((query_lower.rfind(phrase) for phrase in self.TOOL_DISABLE_PHRASES), default=-1)
                
                # Determine which command came last
                if last_disable_pos > last_enable_pos:
                    print(f"Prioritizing disable command at position {last_disable_pos}")
                    result = self._handle_disable_command()
                    if isinstance(result, dict):
                        return True, result.get("message")
                    return True, str(result)  # Convert any non-dict result to string
                else:
                    print(f"Prioritizing enable command at position {last_enable_pos}")
                    result = self._handle_enable_command()
                    if isinstance(result, dict):
                        return True, result.get("message")
                    return True, str(result)  # Convert any non-dict result to string
            
            # Handle single command cases
            elif has_disable_command:
                result = self._handle_disable_command()
                if isinstance(result, dict):
                    return True, result.get("message")
                return True, str(result)  # Convert any non-dict result to string
            elif has_enable_command:
                result = self._handle_enable_command()
                if isinstance(result, dict):
                    return True, result.get("message")
                return True, str(result)  # Convert any non-dict result to string
                
            # Not a tool command
            return False, None
                
        except Exception as e:
            print(f"Error in handle_tool_command: {e}")
            return False, str(e)  # Always return a valid tuple even on error
        
    def _handle_enable_command(self) -> Dict[str, Any]:
        """
        Handle a command to enable tools.
        
        Returns:
            Dict containing:
                - success (bool): Whether command was handled successfully
                - message (str): Response message
                - state (str): Current state of tools
        """
        try:
            if not self.tools_enabled:
                result = self.enable_tools()
                return {
                    "success": True,
                    "message": result["message"],
                    "state": "enabled"
                }
            return {
                "success": True,
                "message": "Tools are already enabled.",
                "state": "enabled"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error enabling tools: {str(e)}",
                "state": "error"
            }

    def _handle_disable_command(self) -> Dict[str, Any]:
        """
        Handle a command to disable tools.
        
        Returns:
            Dict containing:
                - success (bool): Whether command was handled successfully
                - message (str): Response message
                - state (str): Current state of tools
        """
        try:
            if self.tools_enabled:
                result = self.disable_tools(reason="manual")
                if not isinstance(result, dict) or "message" not in result:
                    raise ValueError("Invalid result from disable_tools")
                return {
                    "success": True,
                    "message": result["message"],
                    "state": "disabled"
                }
            return {
                "success": True,
                "message": "Tools are already disabled.",
                "state": "disabled"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error disabling tools: {str(e)}",
                "state": "error"
            }

    def get_tools_for_query(self, query: str) -> Tuple[bool, List[dict]]:
        """
        Determine if tools are needed and which ones are relevant for a query.
        
        Args:
            query: User's voice query text
            
        Returns:
            Tuple containing:
                - bool: Whether any tools are needed for this query
                - List[dict]: List of relevant tool definitions
        """
        query_lower = query.lower()
        relevant_tools = []
        
        # Check if this is ONLY a tool command
        is_enable = any(phrase in query_lower for phrase in self.TOOL_ENABLE_PHRASES)
        is_disable = any(phrase in query_lower for phrase in self.TOOL_DISABLE_PHRASES)
        
        if is_enable or is_disable:
            # Remove the tool command part from query for further analysis
            for phrase in self.TOOL_ENABLE_PHRASES + self.TOOL_DISABLE_PHRASES:
                query_lower = query_lower.replace(phrase, '').strip()
                
            # If nothing left after removing tool command, return early
            if not query_lower:
                return False, []
        
        # Continue with tool analysis on remaining query
        for category, keywords in self.TOOL_CATEGORY_KEYWORDS.items():
            if any(word in query_lower for word in keywords):
                category_tools = get_tools_by_category(category)
                relevant_tools.extend(category_tools)
        
        # Remove duplicate tools while preserving order
        relevant_tools = list({tool['name']: tool for tool in relevant_tools}.values())
        
        # Debug logging
        print(f"\nTool Analysis:")
        print(f"Query: {query_lower[:50]}...")
        print(f"Contains Tool Command: {is_enable or is_disable}")
        print(f"Tools Found: {[tool['name'] for tool in relevant_tools]}")
        
        return bool(relevant_tools), relevant_tools
    
    def process_confirmation(self, response: str) -> Tuple[bool, bool, str]:
        """
        Process user's response to a confirmation prompt about using tools.
        
        Args:
            response: User's response text
            
        Returns:
            Tuple of (was_confirmation, is_affirmative, message)
        """
        response_lower = response.lower()
        
        # Check if this is a confirmation response
        confirmation_words = {'yes', 'yeah', 'correct', 'right', 'sure', 'okay', 'yep', 'yup'}
        rejection_words = {'no', 'nope', 'don\'t', 'do not', 'negative', 'cancel', 'stop'}
        
        is_confirmation = any(word in response_lower for word in confirmation_words) or \
                         any(word in response_lower for word in rejection_words)
        
        if not is_confirmation:
            return False, False, ""
            
        is_affirmative = any(word in response_lower for word in confirmation_words)
        
        # Update tool state based on response
        if is_affirmative:
            self.enable_tools()
            message = "Tools enabled. I'll use them to assist you."
        else:
            self.disable_tools(reason="declined")
            message = "I'll proceed without using tools."
            
        return True, is_affirmative, message



