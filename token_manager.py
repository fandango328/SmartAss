#!/usr/bin/env python3

"""
Token Manager for LAURA Voice Assistant

Manages both tool state and token tracking with accurate overhead calculations.
Provides real-time and cumulative cost tracking for API usage.

Author: fandango328
Updated: 2025-03-12
"""

import json
import os
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any, Set
from dataclasses import dataclass
from pathlib import Path

# Tool toggle command keywords
TOOL_TOGGLE_COMMANDS = {
    "enable": ["enable tools", "turn on tools", "activate tools", "use tools"],
    "disable": ["disable tools", "turn off tools", "deactivate tools", "stop using tools"],
    "status": ["tool status", "are tools enabled", "check tool status"]
}

@dataclass
class TokenCosts:
    """Stores token cost information"""
    input_tokens: int = 0
    output_tokens: int = 0
    tool_overhead: int = 0
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.tool_overhead

class TokenManager:
    """Manages token tracking and tool state for LAURA"""
    
    # Token overhead by model and tool_choice based on official documentation
    TOOL_OVERHEAD = {
        "claude-3-7-sonnet-20250219": {
            "auto": 346,
            "any": 313,
            "tool": 313,
            "none": 0
        },
        "claude-3-5-sonnet-20241022": {  # October release
            "auto": 346,
            "any": 313,
            "tool": 313,
            "none": 0
        },
        "claude-3-opus-20240229": {
            "auto": 530,
            "any": 281,
            "tool": 281,
            "none": 0
        },
        "claude-3-sonnet": {
            "auto": 159,
            "any": 235,
            "tool": 235,
            "none": 0
        },
        "claude-3-haiku-20240307": {
            "auto": 264,
            "any": 340,
            "tool": 340,
            "none": 0
        },
        "claude-3-5-sonnet-20240620": {  # June release
            "auto": 294,
            "any": 261,
            "tool": 261,
            "none": 0
        }
    }

    def __init__(self, anthropic_client, default_model="claude-3-5-haiku-20241022", settings_file="laura_token_settings.json"):
        """Initialize TokenManager"""
        self.client = anthropic_client
        self.default_model = default_model
        self.model = default_model
        self.settings_file = settings_file
        
        # Tool authorization state
        self.authorization_state = "none"  # "none", "requested", "active"
        self.last_authorization_time = None
        self.tools_used_in_session = False
        self.last_tool_used = None
        
        # For tracking what the user is being asked about
        self.pending_confirmation = {
            "type": None,  # "initial", "continuation"
            "expiration_time": None
        }
        
        # Current session tracking
        self.current_costs = TokenCosts()
        self.interaction_history = []
        self.session_start = datetime.now()
        
        # Cumulative tracking
        self.cumulative_costs = TokenCosts()
        
        # Load previous cumulative data if exists
        self._load_cumulative_data()
    
    def _load_cumulative_data(self):
        """Load cumulative token usage from file"""
        try:
            data_file = Path("laura_token_usage.json")
            if data_file.exists():
                with open(data_file, "r") as f:
                    data = json.load(f)
                    self.cumulative_costs = TokenCosts(
                        input_tokens=data.get("input_tokens", 0),
                        output_tokens=data.get("output_tokens", 0),
                        tool_overhead=data.get("tool_overhead", 0)
                    )
        except Exception as e:
            print(f"Error loading cumulative data: {e}")
    
    def _save_cumulative_data(self):
        """Save cumulative token usage to file"""
        try:
            with open("laura_token_usage.json", "w") as f:
                json.dump({
                    "input_tokens": self.cumulative_costs.input_tokens,
                    "output_tokens": self.cumulative_costs.output_tokens,
                    "tool_overhead": self.cumulative_costs.tool_overhead,
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving cumulative data: {e}")
    
    def request_authorization(self) -> bool:
        """
        Request authorization to use tools based on high confidence detection.
        Called when tool analyzer confidence exceeds threshold.
        
        Returns:
            bool: True if authorization is requested, False if already active
        """
        if self.authorization_state == "active":
            return False  # Already authorized
            
        self.authorization_state = "requested"
        self.last_authorization_time = datetime.now()
        self.pending_confirmation = {
            "type": "initial",  
            "expiration_time": datetime.now().timestamp() + 30  # 30 second timeout
        }
        return True
    
    def activate_tools(self) -> bool:
        """
        Activate tool mode after user confirmation.
        
        Returns:
            bool: True if activation was successful
        """
        self.authorization_state = "active"
        self.last_authorization_time = datetime.now()
        self.tools_used_in_session = False
        self.pending_confirmation = {
            "type": None,
            "expiration_time": None
        }
        return True
    
    def deactivate_tools(self) -> bool:
        """
        Deactivate tool mode.
        
        Returns:
            bool: True if deactivation was successful
        """
        self.authorization_state = "none"
        self.last_authorization_time = None
        self.tools_used_in_session = False
        self.last_tool_used = None
        self.pending_confirmation = {
            "type": None,
            "expiration_time": None
        }
        return True
    
    def handle_tool_command(self, query: str) -> Tuple[bool, str]:
        """
        Handle explicit tool toggle voice commands.
        
        Returns:
            Tuple[bool, str]: (was_handled, response_message)
        """
        query_lower = query.lower()
        
        for enable_phrase in TOOL_TOGGLE_COMMANDS["enable"]:
            if enable_phrase in query_lower:
                if self.authorization_state == "active":
                    return True, "Tools are already enabled."
                self.activate_tools()
                return True, "Tools have been enabled. I can now help with tasks that require tool access."
        
        for disable_phrase in TOOL_TOGGLE_COMMANDS["disable"]:
            if disable_phrase in query_lower:
                if self.authorization_state == "none":
                    return True, "Tools are already disabled."
                self.deactivate_tools()
                return True, "Tools have been disabled."
                
        for status_phrase in TOOL_TOGGLE_COMMANDS["status"]:
            if status_phrase in query_lower:
                status = "enabled" if self.authorization_state == "active" else "disabled"
                return True, f"Tools are currently {status}."
                
        return False, ""
    
    def tools_are_active(self) -> bool:
        """Check if tools are currently authorized for use"""
        return self.authorization_state == "active"
    
    def authorization_requested(self) -> bool:
        """Check if tool authorization is currently being requested"""
        if self.authorization_state != "requested":
            return False
            
        # Check for timeout
        if (self.pending_confirmation["expiration_time"] and
            datetime.now().timestamp() > self.pending_confirmation["expiration_time"]):
            self.pending_confirmation = {
                "type": None,
                "expiration_time": None
            }
            return False
            
        return True
    
    def process_confirmation(self, response: str) -> Tuple[bool, bool, str]:
        """
        Process a confirmation response for tool usage.
        
        Args:
            response: User's response text
            
        Returns:
            Tuple[bool, bool, str]: (was_confirmation, is_affirmative, message)
                - was_confirmation: Whether this was a valid confirmation response
                - is_affirmative: Whether user confirmed/approved
                - message: Explanatory message
        """
        if self.authorization_state != "requested":
            return False, False, "No pending authorization request"
            
        # Check for timeout
        if (self.pending_confirmation["expiration_time"] and
            datetime.now().timestamp() > self.pending_confirmation["expiration_time"]):
            self.authorization_state = "none"
            self.pending_confirmation = {
                "type": None,
                "expiration_time": None
            }
            return True, False, "Authorization request timed out"
            
        response_lower = response.lower().strip()
        
        # Analyze response for confirmation signals
        affirmative = {
            "yes", "yeah", "sure", "absolutely", "definitely", "correct",
            "please do", "go ahead", "do it", "okay", "ok", "alright",
            "fine", "yep", "yup", "proceed", "true", "confirm", "confirmed"
        }
        
        negative = {
            "no", "nope", "don't", "do not", "negative", "never",
            "cancel", "stop", "abort", "wait", "hold on", "nevermind", 
            "never mind", "forget it", "false", "nah", "decline"
        }
        
        is_affirmative = any(word in response_lower for word in affirmative) or response_lower in affirmative
        is_negative = any(word in response_lower for word in negative) or response_lower in negative
        
        # If response is clear
        if is_affirmative:
            if self.pending_confirmation["type"] == "initial":
                self.activate_tools()
                return True, True, "Tool usage authorized"
            elif self.pending_confirmation["type"] == "continuation":
                self.activate_tools()  # Just ensure we stay in active state
                return True, True, "Tool usage will continue"
        elif is_negative:
            self.deactivate_tools()
            return True, False, "Tool usage declined"
            
        # Unclear response
        return False, False, "Unclear confirmation response"
    
    def record_tool_usage(self, tool_name: str) -> None:
        """Record that a tool was used in this session"""
        self.tools_used_in_session = True
        self.last_tool_used = tool_name
    
    def request_continuation(self) -> None:
        """
        Request confirmation to continue using tools.
        Called after tool use or if in tool mode but no tool used.
        """
        self.authorization_state = "requested"
        self.pending_confirmation = {
            "type": "continuation",
            "expiration_time": datetime.now().timestamp() + 30  # 30 second timeout
        }
    
    def check_session_timeout(self) -> bool:
        """
        Check if the tool authorization has timed out due to inactivity.
        
        Returns:
            bool: True if session has timed out
        """
        if self.authorization_state == "active" and self.last_authorization_time:
            timeout_minutes = 10  # Session timeout after 10 minutes of no tool use
            
            time_diff = (datetime.now() - self.last_authorization_time).total_seconds() / 60
            if time_diff > timeout_minutes:
                self.deactivate_tools()
                return True
                
        return False
    
    def count_tokens(self, messages: List[Dict], using_tools: bool = False, 
                    tool_choice: str = "auto") -> Tuple[int, int]:
        """Count tokens for messages"""
        try:
            count_request = {
                "model": self.model,
                "messages": messages
            }
            
            if using_tools:
                count_request["tools"] = self._get_tools()
                count_request["tool_choice"] = {"type": tool_choice}
            
            count_response = self.client.messages.count_tokens(**count_request)
            base_tokens = count_response.input_tokens
            
            # Calculate tool overhead
            overhead = 0
            if using_tools and tool_choice in self.TOOL_OVERHEAD[self.model]:
                overhead = self.TOOL_OVERHEAD[self.model][tool_choice]
            
            return base_tokens, overhead
            
        except Exception as e:
            print(f"Token counting error: {e}")
            return 0, 0
    
    def update_costs(self, input_tokens: int, output_tokens: int, 
                    tool_overhead: int = 0):
        """Update both current and cumulative token counts"""
        # Update current session
        self.current_costs.input_tokens += input_tokens
        self.current_costs.output_tokens += output_tokens
        self.current_costs.tool_overhead += tool_overhead
        
        # Update cumulative
        self.cumulative_costs.input_tokens += input_tokens
        self.cumulative_costs.output_tokens += output_tokens
        self.cumulative_costs.tool_overhead += tool_overhead
        
        # Save updated cumulative data
        self._save_cumulative_data()
    
    def calculate_cost(self, costs: TokenCosts) -> float:
        """Calculate cost for given token counts"""
        if "sonnet" in self.model.lower():
            input_cost_per_mtok = 3.00
            output_cost_per_mtok = 15.00
        else:  # Haiku pricing
            input_cost_per_mtok = 0.80
            output_cost_per_mtok = 4.00
        
        input_cost = (costs.input_tokens + costs.tool_overhead) * (input_cost_per_mtok / 1_000_000)
        output_cost = costs.output_tokens * (output_cost_per_mtok / 1_000_000)
        
        return input_cost + output_cost
    
    def print_metrics(self, context: str = "current query"):
        """Print token usage and cost metrics in horizontally aligned layout"""
        width = 68
        col_split = 27  # Adjusted for better column separation
        
        def pad_number(num: int, width: int = 8) -> str:
            return f"{num:,}".rjust(width)
        
        print("\n+" + "-" * width + "+")
        print(f"| Token Usage Report - {self.model}" + " " * (width - len(f"Token Usage Report - {self.model}") - 1) + "|")
        print(f"| Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " " * (width - len(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") - 1) + "|")
        print("|" + "-" * col_split + "+" + "-" * (width - col_split - 1) + "|")
        print("|       Current Query           |          Cumulative Usage            |")
        print("|" + "-" * col_split + "+" + "-" * (width - col_split - 1) + "|")
        
        # Adjusted spacing for consistent alignment
        def format_row(label: str, current: int, cumul: int, width: int = 15) -> str:
            return (f"| {label:<8} {pad_number(current, width)}  | "
                    f"{label:<8} {pad_number(cumul, width)} |")
        
        # Token rows with consistent spacing
        print(format_row("Input:", self.current_costs.input_tokens, self.cumulative_costs.input_tokens))
        print(format_row("Output:", self.current_costs.output_tokens, self.cumulative_costs.output_tokens))
        print(format_row("Overhead:", self.current_costs.tool_overhead, self.cumulative_costs.tool_overhead))
        print(format_row("Total:", self.current_costs.total_tokens, self.cumulative_costs.total_tokens))
        
        # Cost row with adjusted spacing
        current_cost = f"${self.calculate_cost(self.current_costs):.4f}".rjust(14)
        cumul_cost = f"${self.calculate_cost(self.cumulative_costs):.4f}".rjust(14)
        print(f"| Cost:   {current_cost}   | Total:   {cumul_cost}  |")
        
        print("|" + "-" * col_split + "+" + "-" * (width - col_split - 1) + "|")
        
        # Authorization state info
        auth_status = "None"
        if self.authorization_state == "active":
            auth_status = "Active"
        elif self.authorization_state == "requested":
            auth_status = "Requested"
        
        print(f"| Tool Status: {auth_status}" + " " * (width - 15 - len(auth_status)) + "|")
        
        if self.authorization_state == "active":
            overhead = f"| Current Overhead: auto={self.TOOL_OVERHEAD[self.model]['auto']}, any/tool={self.TOOL_OVERHEAD[self.model]['any']}"
            print(overhead + " " * (width - len(overhead)) + "|")
        
        print("+" + "-" * width + "+")
    
    def _get_tools(self) -> List[Dict]:
        """Get tools definition from global scope"""
        import sys
        main_module = sys.modules['__main__']
        return main_module.TOOLS if hasattr(main_module, 'TOOLS') else []
    
    def start_interaction(self):
        """Start tracking a new user interaction"""
        self.current_costs = TokenCosts()
        
        # Check for tool session timeout
        self.check_session_timeout()
    
    def end_interaction(self):
        """End the current interaction and record stats"""
        self.interaction_history.append({
            "timestamp": datetime.now().isoformat(),
            "input_tokens": self.current_costs.input_tokens,
            "output_tokens": self.current_costs.output_tokens,
            "tool_overhead": self.current_costs.tool_overhead,
            "tool_used": self.tools_used_in_session,
            "last_tool": self.last_tool_used
        })