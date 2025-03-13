#!/usr/bin/env python3

"""
TokenManager for SmartAss Voice Assistant
Last Updated: 2025-03-12 23:14:18

Purpose:
    Manages token usage, costs, and authorization states for the SmartAss voice assistant,
    with specific handling of tool-related overheads and model-specific pricing.

Key Features:
    - Token counting and cost calculation for different Claude models
    - Tool usage and definition overhead tracking
    - Authorization state management for tool usage
    - Session-based cost accumulation
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Tuple, Optional, Any
from config_transcription_v3 import ANTHROPIC_MODEL  # Direct import from config
from laura_tools import AVAILABLE_TOOLS, get_tool_by_name, get_tools_by_category

class TokenManager:
    """
    Manages token counting, cost calculation, and tool usage authorization for SmartAss.
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

    # Define authorization phrases
    ACTION_INDICATORS = {
        # Direct requests
        "please", "can you", "could you", "would you", 
        "i want to", "i need to", "i'd like to", "let's",
        "i need you to", "i want you to", "i'd like you to",

        # Imperative forms
        "go ahead and", "just", "help me",

        # Polite forms
        "would you mind", "could you please", "would you please",
        "if you could", "if you would",

        # Direct commands
        "make", "create", "do", "start",

        # Intent indicators
        "i'm trying to", "i am trying to", "i would like to",
        "i'm looking to", "i am looking to"
    }

    TOOL_PHRASES = {
        # Tool enabling
        "enable tool use", "enable tools",
        "start using tools", "activate tools",
        "use tools", "turn on tools",
        "get tools ready", "prepare tools",

        # Email specific
        "draft an email", "send an email",
        "create an email", "compose an email",
        "write an email", "prepare an email",
        "make an email", "start an email",

        # Generic tool actions
        "use the tools", "work with tools",
        "access tools", "initialize tools"
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
        
            # Authorization State Management
            self.authorization_state = "none"
            self.pending_confirmation = {
                "type": None,
                "expiration_time": None
            }
            self.last_authorization_time = None
            self.tool_attempts_remaining = 0
            self.tools_used_in_session = False  # Add this line

    def check_authorization_state(self, query: str, voice_confidence: float) -> Dict[str, Any]:
        """
        Check if tools should be enabled based on query analysis and voice confidence.
        """
        # First get tool context analysis with the new implementation
        needs_tools, tool_names, tool_confidence = self.get_tools_for_query(query)
        
        # Rest of the method stays the same, but add more debug prints
        query_lower = query.lower()
        print(f"Authorization Check - Query: {query_lower}")
        print(f"Current Auth State: {self.authorization_state}")
        
        is_explicit_request = False
        for action in self.ACTION_INDICATORS:
            for phrase in self.TOOL_PHRASES:
                combined = f"{action} {phrase}"
                if combined in query_lower:
                    print(f"Found explicit request: {combined}")
                    is_explicit_request = True
                    self.authorization_state = "active"
                    print(f"Authorization state updated to: {self.authorization_state}")
                    break
            if is_explicit_request:
                break
                
        # Enhanced debug output for authorization decision
        print(f"Authorization Decision:")
        print(f"- Is Explicit Request: {is_explicit_request}")
        print(f"- Voice Confidence: {voice_confidence}")
        print(f"- Needs Tools: {needs_tools}")
        print(f"- Tool Confidence: {tool_confidence}")
        
        # Rest of the logic stays the same but with better feedback
        if is_explicit_request and voice_confidence >= 90:
            self.tool_attempts_remaining = 2
            return {
                "state": "authorized",
                "needs_confirmation": False,
                "message": "Tools enabled. What would you like help with?",
                "tools": tool_names
            }
            
        if needs_tools and voice_confidence >= 90:
            return {
                "state": "pending",
                "needs_confirmation": True,
                "message": "Just to confirm, we're going to be enabling tools now, right?",
                "tools": tool_names
            }
            
        return {
            "state": "none",
            "needs_confirmation": False,
            "message": None,
            "tools": []
        }
                
        # CASE 1: Explicit tool request with high confidence
        if is_explicit_request:
            if voice_confidence >= 90:
                self.tool_attempts_remaining = 2
                return {
                    "state": "authorized",
                    "needs_confirmation": False,
                    "message": "Tools enabled. What would you like help with?",
                    "tools": tool_names
                }
            return {
                "state": "none",
                "needs_confirmation": False,
                "message": "I couldn't understand that clearly. Could you repeat your request?",
                "tools": set()
            }
            
        # CASE 2: Tool context detected from query
        if needs_tools and voice_confidence >= 90:
            return {
                "state": "pending",
                "needs_confirmation": True,
                "message": "Just to confirm, we're going to be enabling tools now, right?",
                "tools": tool_names
            }
            
        # CASE 3: No tool needs detected
        return {
            "state": "none",
            "needs_confirmation": False,
            "message": None,
            "tools": set()
        }

    def request_continuation(self):
        """Ask if user wants to continue using tools."""
        print(f"Requesting continuation of tool use")
        # This method is called after successful tool use to potentially ask about continued tool usage
        pass

    def record_tool_usage(self, tool_name: str):
        """Record that a tool was used in the current session."""
        self.tools_used_in_session = True
        print(f"Tool usage recorded: {tool_name}")

    def tools_are_active(self) -> bool:
        """Check if tools are currently active."""
        active = self.authorization_state == "active"
        print(f"Checking if tools are active: {active}")
        return active

    def request_authorization(self):
        """Request authorization for tool use."""
        self.authorization_state = "active"  # Changed from "requested" to "active"
        print(f"Authorization requested. New state: {self.authorization_state}")

    def handle_confirmation(self, response: str, voice_confidence: float) -> Dict[str, Any]:
        """Process user's response to tool confirmation prompt."""
        if voice_confidence >= 90 and any(word in response.lower() 
                                        for word in {'yes', 'yeah', 'correct', 'right'}):
            self.tool_attempts_remaining = 2
            self.authorization_state = "active"  # Added this line to fix state inconsistency
            return {
                "state": "authorized",
                "message": "What would you like me to help you with using tools?",
                "needs_input": True
            }
        
        self.authorization_state = "none"
        return {
            "state": "none",
            "message": "Continuing without tools.",
            "needs_input": False
        }

    # Keeping placeholder methods for future implementation
    def get_tools_for_query(self, query: str) -> Tuple[bool, list, float]:
        """
        Analyze query to determine if tools are needed.
        Returns: (needs_tools, relevant_tools, confidence)
        """
        query_lower = query.lower()
    
        relevant_tools = []
        confidence = 0.0
        is_tool_request = False
    
        # Check for tool enabling without early return
        if any(phrase in query_lower for phrase in self.TOOL_PHRASES):
            if "enable tool" in query_lower or "use tool" in query_lower:
                is_tool_request = True
                confidence = max(confidence, 0.9)
    
        # Email-related patterns
        if any(word in query_lower for word in ['email', 'mail', 'send', 'write', 'compose', 'draft']):
            email_tools = get_tools_by_category('EMAIL')
            relevant_tools.extend(email_tools)
            confidence = max(confidence, 0.8)
            
        # Calendar-related patterns
        if any(word in query_lower for word in ['calendar', 'schedule', 'event', 'meeting']):
            calendar_tools = get_tools_by_category('CALENDAR')
            relevant_tools.extend(calendar_tools)
            confidence = max(confidence, 0.8)
            
        # Task-related patterns
        if any(word in query_lower for word in ['task', 'todo', 'reminder']):
            task_tools = get_tools_by_category('TASK')
            relevant_tools.extend(task_tools)
            confidence = max(confidence, 0.8)
            
        # Utility patterns
        if any(word in query_lower for word in ['time', 'location', 'calibrate', 'voice']):
            utility_tools = get_tools_by_category('UTILITY')
            relevant_tools.extend(utility_tools)
            confidence = max(confidence, 0.8)
            
        # Contact patterns
        if any(word in query_lower for word in ['contact', 'person', 'people']):
            contact_tools = get_tools_by_category('CONTACT')
            relevant_tools.extend(contact_tools)
            confidence = max(confidence, 0.8)
        
        # Remove duplicates while preserving order
        relevant_tools = list({tool['name']: tool for tool in relevant_tools}.values())
    
        # Tool need is determined by either explicit request or finding relevant tools
        needs_tools = is_tool_request or bool(relevant_tools) or confidence > 0.7
    
        print(f"Tool Analysis - Query: {query_lower}")
        print(f"Is Tool Request: {is_tool_request}")
        print(f"Needs Tools: {needs_tools}")
        print(f"Confidence: {confidence}")
        print(f"Relevant Tools: {[tool['name'] for tool in relevant_tools]}")
    
        return needs_tools, relevant_tools, confidence

    def handle_tool_command(self, query):
        return False, None

    def start_conversation_session(self):
        pass

    def end_conversation_session(self):
        pass

    def start_interaction(self):
        """Start tracking a new user interaction"""
        self.last_interaction_time = datetime.now()
        self.tools_used_in_session = False  # Reset tool usage tracking

    def authorization_requested(self):
        """Check if authorization is currently requested."""
        return self.authorization_state == "requested"

    def update_session_costs(self, input_tokens: int, output_tokens: int, 
                           is_tool_use: bool = False, tools_enabled: bool = False) -> Tuple[Decimal, Decimal]:
        """Update session costs with clear, separate calculation and accumulation steps."""
        model = self.tool_model if is_tool_use else self.query_model
        
        current_input = input_tokens
        current_output = output_tokens
        
        if tools_enabled:
            current_input += self.TOOL_COSTS["definition_overhead"]
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
        """Calculate cost for given token count based on model and token type."""
        if token_count < 0:
            raise ValueError("Token count cannot be negative")
        if model not in self.MODEL_COSTS or token_type not in self.MODEL_COSTS[model]:
            raise KeyError(f"Invalid model ({model}) or token type ({token_type})")
        
        per_million_rate = Decimal(str(self.MODEL_COSTS[model][token_type]))
        return (per_million_rate / Decimal('1000000')) * Decimal(str(token_count))

    def log_api_interaction(self, interaction_type, query_text):
        """Log details about an API interaction."""
        print(f"API Interaction - Type: {interaction_type}, Query: {query_text[:50]}...")

    def log_error(self, error_message):
        """Log an error that occurred during processing."""
        print(f"Token Manager Error: {error_message}")

    def track_tool_attempt(self, successful: bool) -> Dict[str, Any]:
        """Track tool usage attempt and manage remaining attempts."""
        if successful:
            return {
                "state": "pending",
                "message": "Would you like to use another tool?",
                "continue_tools": True
            }
    
        self.tool_attempts_remaining -= 1
    
        if self.tool_attempts_remaining <= 0:
            self.authorization_state = "none"
            return {
                "state": "none",
                "message": "Reverting to normal conversation mode.",
                "continue_tools": False
            }
    
        return {
            "state": "authorized",
            "message": "What tool would you like to use? You have one more attempt.",
            "continue_tools": True
        }