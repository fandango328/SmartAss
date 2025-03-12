#!/usr/bin/env python3

"""
Token Manager for LAURA Voice Assistant

Manages both tool state and token tracking with accurate overhead calculations.
Provides real-time and cumulative cost tracking for API usage.

Author: fandango328
Created: 2025-03-12
"""

import json
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from pathlib import Path

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

    def __init__(self, anthropic_client, default_model="claude-3-5-haiku-20241022"):
        """Initialize TokenManager"""
        self.client = anthropic_client
        self.default_model = default_model
        self.model = default_model
        self.tools_enabled = False
        self.tool_use_authorized = False
        self.last_tool_used = None
        
        # Current session tracking
        self.current_costs = TokenCosts()
        
        # Cumulative tracking
        self.cumulative_costs = TokenCosts()
        self.session_start = datetime.now()
        
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
    
    def handle_tool_command(self, query: str) -> Tuple[bool, str]:
        """Handle tool toggle voice commands"""
        query_lower = query.lower()
        
        if "enable tools" in query_lower:
            self.tools_enabled = True
            # Always use Sonnet for tool operations
            self.model = "claude-3-5-sonnet-20241022"
            auto_overhead = self.TOOL_OVERHEAD[self.model]['auto']
            return True, (
                f"Tools enabled using {self.model} for reliable tool handling. "
                f"Tool overhead: {auto_overhead} tokens per use."
            )
        
        if "disable tools" in query_lower:
            self.tools_enabled = False
            self.tool_use_authorized = False
            self.model = self.default_model
            return True, f"Tools disabled. Reverted to {self.model}."
            
        return False, ""
    
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
    print(f"¦ Token Usage Report - {self.model}" + " " * (width - len(f"Token Usage Report - {self.model}") - 1) + "¦")
    print(f"¦ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " " * (width - len(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") - 1) + "¦")
    print("¦" + "-" * col_split + "-" + "-" * (width - col_split - 1) + "¦")
    print("¦       Current Query           ¦          Cumulative Usage            ¦")
    print("¦" + "-" * col_split + "+" + "-" * (width - col_split - 1) + "¦")
    
    # Adjusted spacing for consistent alignment
    def format_row(label: str, current: int, cumul: int, width: int = 15) -> str:
        return (f"¦ {label:<8} {pad_number(current, width)}  ¦ "
                f"{label:<8} {pad_number(cumul, width)} ¦")
    
    # Token rows with consistent spacing
    print(format_row("Input:", self.current_costs.input_tokens, self.cumulative_costs.input_tokens))
    print(format_row("Output:", self.current_costs.output_tokens, self.cumulative_costs.output_tokens))
    print(format_row("Overhead:", self.current_costs.tool_overhead, self.cumulative_costs.tool_overhead))
    print(format_row("Total:", self.current_costs.total_tokens, self.cumulative_costs.total_tokens))
    
    # Cost row with adjusted spacing
    current_cost = f"${self.calculate_cost(self.current_costs):.4f}".rjust(14)
    cumul_cost = f"${self.calculate_cost(self.cumulative_costs):.4f}".rjust(14)
    print(f"¦ Cost:   {current_cost}   ¦ Total:   {cumul_cost}  ¦")
    
    print("¦" + "-" * col_split + "-" + "-" * (width - col_split - 2) + "¦")
    
    if self.tools_enabled:
        print(f"¦ Tool Status: Enabled" + " " * (width - 23) + "¦")
        overhead = f"¦ Current Overhead: auto={self.TOOL_OVERHEAD[self.model]['auto']}, any/tool={self.TOOL_OVERHEAD[self.model]['any']}"
        print(overhead + " " * (width - len(overhead)) + "¦")
    else:
        print(f"¦ Tool Status: Disabled" + " " * (width - 22) + "¦")
    
    print("+" + "-" * width + "+")
    
    def can_use_tools(self, query: str) -> bool:
        """Determine if tools can be used for this query"""
        if not self.tools_enabled:
            return False
        if not self.tool_use_authorized:
            return False
        return True
    
    def authorize_tool_use(self):
        """Grant permission to use tools for this conversation"""
        self.tool_use_authorized = True
    
    def reset_authorization(self):
        """Reset tool authorization"""
        self.tool_use_authorized = False
        self.last_tool_used = None
    
    def _get_tools(self) -> List[Dict]:
        """Get tools definition from global scope"""
        import sys
        main_module = sys.modules['__main__']
        return main_module.TOOLS if hasattr(main_module, 'TOOLS') else []