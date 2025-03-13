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
from typing import Dict, Tuple, Optional
from config_transcription_v3 import ANTHROPIC_MODEL  # Direct import from config

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
        self.query_model = ANTHROPIC_MODEL  # From config_transcription_v3
        self.tool_model = "claude-3-5-sonnet-20241022"  # Fixed model for tool operations
        
        # Session Tracking
        self.total_input_tokens = 0  # Cumulative input tokens
        self.total_output_tokens = 0  # Cumulative output tokens
        self.session_start_time = datetime.now()
        
        # Cost Tracking (using Decimal for precision)
        self.total_tool_definition_costs = Decimal('0.00')  # Tool definition overhead costs
        self.total_tool_usage_costs = Decimal('0.00')      # Tool usage costs
        self.total_query_costs = Decimal('0.00')           # Basic query costs
        
        # Authorization State Management
        self.authorization_state = "none"  # States: none, requested, active
        self.pending_confirmation = {
            "type": None,           # Type of tool access requested
            "expiration_time": None # When the authorization request expires
        }
        self.last_authorization_time = None

    def update_session_costs(self, input_tokens: int, output_tokens: int, 
                           is_tool_use: bool = False, tools_enabled: bool = False) -> Tuple[Decimal, Decimal]:
        """
        Update session costs with clear, separate calculation and accumulation steps.
        Returns the costs for this specific update.
        """
        # 1. Determine which model to use for pricing
        model = self.tool_model if is_tool_use else self.query_model
        
        # 2. Calculate this update's tokens (separating calculation from accumulation)
        current_input = input_tokens
        current_output = output_tokens
        
        # Add tool overheads if needed
        if tools_enabled:
            current_input += self.TOOL_COSTS["definition_overhead"]
        if is_tool_use:
            current_input += self.TOOL_COSTS["usage_overhead"]["tool"]
            
        # 3. Calculate costs for this update
        update_input_cost = self.calculate_token_cost(model, "input", current_input)
        update_output_cost = self.calculate_token_cost(model, "output", current_output)
        
        # 4. Update running totals
        self.total_input_tokens += current_input
        self.total_output_tokens += current_output
        
        if is_tool_use:
            self.total_tool_usage_costs += update_input_cost + update_output_cost
        else:
            self.total_query_costs += update_input_cost + update_output_cost
            
        return update_input_cost, update_output_cost

    def calculate_token_cost(self, model: str, token_type: str, token_count: int) -> Decimal:
        if token_count < 0:
            raise ValueError("Token count cannot be negative")
        if model not in self.MODEL_COSTS or token_type not in self.MODEL_COSTS[model]:
            raise KeyError(f"Invalid model ({model}) or token type ({token_type})")
        
        per_million_rate = Decimal(str(self.MODEL_COSTS[model][token_type]))
        return (per_million_rate / Decimal('1000000')) * Decimal(str(token_count))

def check_authorization_state(self, query: str, voice_confidence: float) -> Dict[str, Any]:
    """
    Check if tools should be enabled based on query analysis and voice confidence.
    
    Args:
        query: The user's voice query text
        voice_confidence: Voice recognition confidence (0-100)
        
    Returns:
        Dict containing:
            state: Current authorization state ('none', 'pending', 'authorized')
            needs_confirmation: Whether user confirmation is needed
            message: Next prompt if needed
            tools: Set of relevant tools if any
    """
    # First get tool context analysis
    needs_tools, tool_names, tool_confidence = get_tools_for_query(query)
    
    # Track what we're detecting for clarity in responses
    query_lower = query.lower()
    
    # These combinations require explicit confirmation
    action_indicators = {
        "please", "can you", "could you", "would you", 
        "i want to", "i need to", "i'd like to", "let's"
    }
    
    tool_phrases = {
        "enable tool use", "enable tools",
        "start using tools", "activate tools"
    }
    
    # Check for proper explicit requests by requiring action indicator + tool phrase
    is_explicit_request = False
    for action in action_indicators:
        for phrase in tool_phrases:
            if f"{action} {phrase}" in query_lower:
                is_explicit_request = True
                break
        if is_explicit_request:
            break
            
    # CASE 1: Explicit tool request with high confidence
    if is_explicit_request:
        if voice_confidence >= 99:
            self.tool_attempts_remaining = 2  # Reset tool attempts
            return {
                "state": "authorized",
                "needs_confirmation": False,
                "message": "Tools enabled. What would you like help with?",
                "tools": tool_names
            }
        # If voice confidence too low, ask for repeat
        return {
            "state": "none",
            "needs_confirmation": False,
            "message": "I couldn't understand that clearly. Could you repeat your request?",
            "tools": set()
        }
        
    # CASE 2: Tool context detected from query
    if needs_tools and voice_confidence >= 90:
        # Tool context detected, need confirmation
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

def handle_confirmation(self, response: str, voice_confidence: float) -> Dict[str, Any]:
    """
    Process user's response to tool confirmation prompt.
    
    Args:
        response: User's voice response to confirmation
        voice_confidence: Voice recognition confidence (0-100)
        
    Returns:
        Dict containing:
            state: New authorization state
            message: Next prompt or acknowledgment
            needs_input: Whether we need specific tool request
    """
    # Only accept clear confirmations
    if voice_confidence >= 90 and any(word in response.lower() 
                                    for word in {'yes', 'yeah', 'correct', 'right'}):
        self.tool_attempts_remaining = 2  # Give two attempts for tool use
        return {
            "state": "authorized",
            "message": "What would you like me to help you with using tools?",
            "needs_input": True
        }
    
    # Any unclear response reverts to no tools
    self.authorization_state = "none"
    return {
        "state": "none",
        "message": "Continuing without tools.",
        "needs_input": False
    }

    def track_tool_attempt(self, successful: bool) -> Dict[str, Any]:
        """
        Track tool usage attempt and manage remaining attempts.
    
        Args:
            successful: Whether the tool execution was successful
        
        Returns:
            Dict containing:
                state: Current state after attempt
                message: Next prompt if needed
                continue_tools: Whether tools remain available
        """
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