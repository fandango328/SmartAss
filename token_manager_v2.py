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

from datetime import datetime
from decimal import Decimal
from typing import Dict, Tuple, Optional, Any, List
from config_transcription_v3 import ANTHROPIC_MODEL  # Direct import from config
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
        "tools activate", "launch toolkit", "begin assistance",
        "start tools", "enable assistant", "tools online",
        
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
        Initialize TokenManager with client using settings from config.

        Args:
            anthropic_client: Anthropic API client instance for API interactions

        Raises:
            ValueError: If ANTHROPIC_MODEL from config is not in MODEL_COSTS
            TypeError: If anthropic_client is None
        """
        if anthropic_client is None:
            raise TypeError("anthropic_client cannot be None")
    
        if ANTHROPIC_MODEL not in self.MODEL_COSTS:
            raise ValueError(f"Model {ANTHROPIC_MODEL} not found in supported models: {list(self.MODEL_COSTS.keys())}")

        # API Client
        self.client = anthropic_client
    
        # Model Configuration
        self.query_model = ANTHROPIC_MODEL
        self.tool_model = "claude-3-5-sonnet-20241022"
    
        # Session Tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.session_start_time = datetime.now()
        self.last_interaction_time = None
    
        # Cost Tracking (using Decimal for precision)
        self.total_tool_definition_costs = Decimal('0.00')
        self.total_tool_usage_costs = Decimal('0.00')
        self.total_query_costs = Decimal('0.00')
    
        # NEW: Simplified Tool State Management
        self.tools_enabled = False  # Binary state: enabled or disabled
        self.consecutive_non_tool_queries = 0  # Counter for auto-disable
        self.tool_disable_threshold = 3  # Disable after this many non-tool queries
        self.tools_used_in_session = False  # Track if any tool was used in current session
        
        # For logging and debugging
        self.last_state_change_time = datetime.now()
        self.last_tool_usage_time = None

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
        
        print(f"Tools enabled at {self.last_state_change_time}")
        if query:
            print(f"Enabled by query: {query[:50]}...")
            
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
        
        print(f"Tools disabled at {self.last_state_change_time} (Reason: {reason})")
        
        message = "Tools are now disabled." if was_enabled else "Tools are already disabled."
        if reason == "auto":
            message = "Tools have been automatically disabled after 3 queries without tool usage."
            
        return {
            "state": "disabled",
            "message": message,
            "tools_active": False
        }

    def record_tool_usage(self, tool_name: str):
        """
        Record that a tool was used in the current session and reset the counter.
        
        Args:
            tool_name: Name of the tool that was used
        """
        self.tools_used_in_session = True
        self.consecutive_non_tool_queries = 0  # Reset counter when tool is used
        self.last_tool_usage_time = datetime.now()
        print(f"Tool usage recorded: {tool_name}")
        print(f"Non-tool query counter reset to 0")

    def track_query_completion(self, used_tool: bool = False) -> Dict[str, Any]:
        """
        Track completion of a query and update tool state if needed.
        Should be called after each query is processed.
        
        Args:
            used_tool: Whether a tool was used in this query
            
        Returns:
            Dict with information about the state change, if any
        """
        # If tools are not enabled, nothing to track
        if not self.tools_enabled:
            return {"state_changed": False, "tools_active": False}
            
        # If a tool was used, we've already reset the counter in record_tool_usage()
        if used_tool:
            return {"state_changed": False, "tools_active": True}
            
        # No tool was used, increment counter
        self.consecutive_non_tool_queries += 1
        print(f"Non-tool query counter increased to {self.consecutive_non_tool_queries}")
        
        # Check if we should auto-disable tools
        if self.consecutive_non_tool_queries >= self.tool_disable_threshold:
            result = self.disable_tools(reason="auto")
            return {
                "state_changed": True, 
                "tools_active": False,
                "message": result["message"]
            }
            
        # No state change
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
        """
        Check if a query is a command to enable or disable tools,
        prioritizing the last command if multiple are present.
        
        Args:
            query: User's query text
            
        Returns:
            Tuple of (was_command, response_message)
        """
        query_lower = query.lower()
        
        # Check for both command types
        has_enable_command = any(phrase in query_lower for phrase in self.TOOL_ENABLE_PHRASES)
        has_disable_command = any(phrase in query_lower for phrase in self.TOOL_DISABLE_PHRASES)
        
        # Handle the conflict case (mixed commands)
        if has_enable_command and has_disable_command:
            print(f"Mixed tool commands detected in: {query_lower}")
            
            # Find the position of the last occurring command
            last_enable_pos = max([query_lower.find(phrase) for phrase in self.TOOL_ENABLE_PHRASES 
                                  if phrase in query_lower] or [-1])
            last_disable_pos = max([query_lower.find(phrase) for phrase in self.TOOL_DISABLE_PHRASES 
                                   if phrase in query_lower] or [-1])
            
            # Determine which command came last
            if last_disable_pos > last_enable_pos:
                print(f"Prioritizing disable command at position {last_disable_pos}")
                return self._handle_disable_command()
            else:
                print(f"Prioritizing enable command at position {last_enable_pos}")
                return self._handle_enable_command()
        
        # Handle single command cases
        elif has_disable_command:
            return self._handle_disable_command()
        elif has_enable_command:
            return self._handle_enable_command()
            
        # Not a tool command
        return False, None
        
    def _handle_enable_command(self) -> Tuple[bool, str]:
        """
        Handle a command to enable tools.
        
        Returns:
            Tuple of (was_handled, response_message)
        """
        if not self.tools_enabled:
            result = self.enable_tools()
            return True, result["message"]
        else:
            return True, "Tools are already enabled."
            
    def _handle_disable_command(self) -> Tuple[bool, str]:
        """
        Handle a command to disable tools.
        
        Returns:
            Tuple of (was_handled, response_message)
        """
        if self.tools_enabled:
            result = self.disable_tools(reason="manual")
            return True, result["message"]
        else:
            return True, "Tools are already disabled."

    def get_tools_for_query(self, query: str) -> Tuple[bool, List[dict], float]:
        """
        Analyze query to determine if tools are needed.
        
        Args:
            query: The user's query text
            
        Returns:
            Tuple of (needs_tools, relevant_tools, confidence)
        """
        query_lower = query.lower()
        
        relevant_tools = []
        confidence = 0.0
        
        # Check for explicit tool commands
        # Note: We don't return early here because we still want to collect relevant tools
        if any(phrase in query_lower for phrase in self.TOOL_ENABLE_PHRASES):
            confidence = max(confidence, 0.9)
        if any(phrase in query_lower for phrase in self.TOOL_DISABLE_PHRASES):
            # This is a request to disable tools, not use them
            return False, [], 0.9
            
        # Check for tool category keywords
        for category, keywords in self.TOOL_CATEGORY_KEYWORDS.items():
            if any(word in query_lower for word in keywords):
                category_tools = get_tools_by_category(category)
                relevant_tools.extend(category_tools)
                confidence = max(confidence, 0.8)
            
        # Remove duplicates while preserving order
        relevant_tools = list({tool['name']: tool for tool in relevant_tools}.values())
        
        # Tool need is determined by confidence threshold or finding relevant tools
        # If tools are already enabled, we have a lower bar for relevance
        needs_tools = confidence > 0.7 or bool(relevant_tools)
        
        print(f"Tool Analysis - Query: {query_lower[:50]}...")
        print(f"Tools Enabled: {self.tools_enabled}")
        print(f"Needs Tools: {needs_tools}")
        print(f"Confidence: {confidence}")
        print(f"Relevant Tools: {[tool['name'] for tool in relevant_tools]}")
        
        return needs_tools, relevant_tools, confidence

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

    def start_interaction(self):
        """
        Start tracking a new user interaction.
        Called at the beginning of each interaction to track timing.
        """
        self.last_interaction_time = datetime.now()

    def update_session_costs(self, input_tokens: int, output_tokens: int, 
                           is_tool_use: bool = False) -> Tuple[Decimal, Decimal]:
        """
        Update session costs with token usage information.
        
        Args:
            input_tokens: Number of input tokens in this interaction
            output_tokens: Number of output tokens in this interaction
            is_tool_use: Whether this interaction involved tool usage
            
        Returns:
            Tuple of (input_cost, output_cost)
        """
        model = self.tool_model if is_tool_use else self.query_model
        
        current_input = input_tokens
        current_output = output_tokens
        
        # Add tool definition overhead if tools are enabled
        if self.tools_enabled:
            current_input += self.TOOL_COSTS["definition_overhead"]
            
        # Add tool usage overhead if applicable
        if is_tool_use:
            current_input += self.TOOL_COSTS["usage_overhead"]["tool"]
            
        update_input_cost = self.calculate_token_cost(model, "input", current_input)
        update_output_cost = self.calculate_token_cost(model, "output", current_output)
        
        self.total_input_tokens += current_input
        self.total_output_tokens += current_output
        
        if is_tool_use:
            self.total_tool_usage_costs += update_input_cost + update_output_cost
        else:
            self.total_query_costs += update_input_cost + update_output_cost
            
        return update_input_cost, update_output_cost

    def calculate_token_cost(self, model: str, token_type: str, token_count: int) -> Decimal:
        """
        Calculate cost for given token count based on model and token type.
        
        Args:
            model: The model identifier (e.g., "claude-3-5-sonnet")
            token_type: Either "input" or "output"
            token_count: Number of tokens to calculate cost for
            
        Returns:
            Decimal cost for the tokens
            
        Raises:
            ValueError: If token_count is negative
            KeyError: If model or token_type is not valid
        """
        if token_count < 0:
            raise ValueError("Token count cannot be negative")
        if model not in self.MODEL_COSTS or token_type not in self.MODEL_COSTS[model]:
            raise KeyError(f"Invalid model ({model}) or token type ({token_type})")
        
        per_million_rate = Decimal(str(self.MODEL_COSTS[model][token_type]))
        return (per_million_rate / Decimal('1000000')) * Decimal(str(token_count))

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
        
    # Legacy method compatibility layer
    def check_authorization_state(self, query: str, voice_confidence: float) -> Dict[str, Any]:
        """
        Compatibility method for legacy code - redirects to the new implementation.
        
        Args:
            query: User query text
            voice_confidence: Voice recognition confidence (0-100)
            
        Returns:
            Dictionary with state information in the legacy format
        """
        # First handle any explicit tool commands
        was_command, message = self.handle_tool_command(query)
        if was_command:
            return {
                "state": "authorized" if self.tools_enabled else "none",
                "needs_confirmation": False,
                "message": message,
                "tools": []  # No specific tools to suggest
            }
        
        # Get tools that might be relevant to this query
        needs_tools, relevant_tools, tool_confidence = self.get_tools_for_query(query)
        
        # If tools are already enabled, just return their state
        if self.tools_enabled:
            return {
                "state": "authorized",
                "needs_confirmation": False,
                "message": None,
                "tools": [tool['name'] for tool in relevant_tools]
            }
            
        # If we need tools and have sufficient confidence
        if needs_tools and voice_confidence >= 90:
            return {
                "state": "pending",
                "needs_confirmation": True,
                "message": "Just to confirm, we're going to be enabling tools now, right?",
                "tools": [tool['name'] for tool in relevant_tools]
            }
        
        # Default case - no tools needed
        return {
            "state": "none",
            "needs_confirmation": False,
            "message": None,
            "tools": []
        }
    
    # Legacy method compatibility layer
    def request_authorization(self):
        """Legacy method - redirects to the new implementation."""
        self.enable_tools()
    
    # Legacy method compatibility layer
    def handle_confirmation(self, response: str, voice_confidence: float) -> Dict[str, Any]:
        """
        Compatibility method for legacy code - redirects to the new implementation.
        
        Args:
            response: User's response to confirmation prompt
            voice_confidence: Voice recognition confidence (0-100)
            
        Returns:
            Dictionary with state information in the legacy format
        """
        # Low confidence, don't process
        if voice_confidence < 80:
            return {
                "state": "none",
                "message": "I couldn't understand that clearly. Let's continue without tools.",
                "needs_input": False
            }
            
        was_confirmation, is_affirmative, message = self.process_confirmation(response)
        
        if not was_confirmation:
            # Not a clear yes/no response
            return {
                "state": "none",
                "message": "I'm not sure if you want to use tools. Let's continue without them for now.",
                "needs_input": False
            }
            
        if is_affirmative:
            return {
                "state": "authorized",
                "message": message,
                "needs_input": True
            }
        else:
            return {
                "state": "none",
                "message": message,
                "needs_input": False
            }
    
    # Legacy method compatibility layer
    def track_tool_attempt(self, successful: bool) -> Dict[str, Any]:
        """
        Compatibility method for legacy code - redirects to the new implementation.
        
        Args:
            successful: Whether the tool attempt was successful
            
        Returns:
            Dictionary with state information in the legacy format
        """
        if successful:
            self.record_tool_usage("unknown_tool")  # Record usage with generic name
            return {
                "state": "pending",
                "message": "Would you like to use another tool?",
                "continue_tools": True
            }
        
        # If unsuccessful and tools are enabled, we'll count this as a non-tool query
        if self.tools_enabled:
            result = self.track_query_completion(used_tool=False)
            
            # If tools were auto-disabled
            if result.get("state_changed", False):
                return {
                    "state": "none",
                    "message": result.get("message", "Continuing without tools."),
                    "continue_tools": False
                }
            
            # Tools still enabled but counter increased
            return {
                "state": "authorized",
                "message": f"Tools are still enabled. You have {result.get('queries_remaining', 'some')} more attempts.",
                "continue_tools": True
            }
        
        # Tools already disabled
        return {
            "state": "none",
            "message": "Continuing without tools.",
            "continue_tools": False
        }
