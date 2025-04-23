#!/usr/bin/env python3

"""
Manages token counting, cost calculation, and tool usage authorization for SmartAss.

Features:
- Binary tool state (enabled/disabled)
- Phonetically optimized voice commands for tool state changes
- Token and cost tracking for API interactions
"""
import os
import json
import glob
import traceback
from config_cl import CHAT_LOG_MAX_TOKENS, CHAT_LOG_RECOVERY_TOKENS, CHAT_LOG_DIR, SYSTEM_PROMPT, ANTHROPIC_MODEL, SOUND_PATHS, SYSTEM_STATE_COMMANDS
from datetime import datetime
from decimal import Decimal
from typing import Dict, Tuple, Optional, Any, List, Union
from laura_tools import AVAILABLE_TOOLS, get_tool_by_name, get_tools_by_category
from audio_manager_vosk import AudioManager

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



    # Voice-optimized tool disabling phrases
    # Selected for clear phonetic distinction from enabling phrases and from
    # common conversation patterns to minimize false positives/negatives


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
        # Core client validation
        if anthropic_client is None:
            raise TypeError("anthropic_client cannot be None")
            
        # Core client and model settings
        self.anthropic_client = anthropic_client
        self.query_model = ANTHROPIC_MODEL
        self.tool_model = "claude-3-5-sonnet-20241022"
        
        # Session state tracking
        self.session_active = False
        self.session_start_time = None
        
        # Initialize empty tool handlers
        self.tool_handlers = {}
        
        # Token tracking per model
        self.haiku_tracker = {
            'input_tokens': 0,
            'output_tokens': 0,
            'cost': Decimal('0.00')
        }
        
        self.sonnet_tracker = {
            'input_tokens': 0,
            'output_tokens': 0,
            'tool_definition_tokens': 0,
            'cost': Decimal('0.00'),
            'tools_initialized': False
        }

        # Session state with model info
        self.session = {
            'current_model': 'claude-3-5-haiku-20241022',
            'history_tokens': 0,
            'total_cost': Decimal('0.00'),
            'tools_enabled': False,
        }
              
        # Tool state management
        self.tools_enabled = False
        self.tools_used_in_session = False

    def register_tool_handler(self, name, handler):
        """Register a single tool handler"""
        self.tool_handlers[name] = handler

    def start_interaction(self):
        """
        Start tracking a new interaction.
        Should be called at the beginning of each user interaction.
        """
        if not self.session_active:
            self.start_session()
            
        print("\n=== Token Log ===")
        return {
            "status": "started",
            "timestamp": datetime.now().isoformat(),
            "tools_enabled": self.tools_enabled,
            "session_active": self.session_active
        }

    def log_error(self, error_message: str):
        """
        Log an error that occurred during interaction.
        
        Args:
            error_message: The error message to log
            
        Returns:
            Dict with error logging status
        """
        try:
            timestamp = datetime.now().isoformat()
            error_log_dir = os.path.join(CHAT_LOG_DIR, "errors")
            os.makedirs(error_log_dir, exist_ok=True)
            
            log_file = os.path.join(error_log_dir, f"error_log_{datetime.now().strftime('%Y-%m-%d')}.json")
            
            error_entry = {
                "timestamp": timestamp,
                "error": error_message,
                "session_state": {
                    "tools_enabled": self.tools_enabled,
                    "tools_used": self.tools_used_in_session,
                }
            }
            
            # Load existing log if it exists
            existing_logs = []
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        existing_logs = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Error log file {log_file} was corrupted, starting fresh")
                    
            # Append new error and write back
            if not isinstance(existing_logs, list):
                existing_logs = []
            existing_logs.append(error_entry)
            
            # Write atomically using a temporary file
            temp_file = f"{log_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(existing_logs, f, indent=2)
            os.replace(temp_file, log_file)
            
            print(f"\nError logged: {error_message}")
            return {
                "status": "logged",
                "timestamp": timestamp,
                "location": log_file
            }
            
        except Exception as e:
            print(f"Failed to log error: {str(e)}")
            return {
                "status": "failed",
                "reason": str(e)
            }


    def prepare_messages_for_token_count(self, current_query: str, chat_log: list, system_prompt: str = None) -> tuple:
        formatted_messages = []

        # Handle chat log messages
        if chat_log and isinstance(chat_log, list):
            for msg in chat_log:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    formatted_messages.append({
                        "role": msg['role'],
                        "content": [{
                            "type": "text",
                            "text": str(msg['content'])
                        }]
                    })

        # Add current query
        formatted_messages.append({
            "role": "user",
            "content": [{"type": "text", "text": str(current_query)}]
        })

        system_content = (system_prompt or SYSTEM_PROMPT).strip()
        return formatted_messages, system_content

    def count_message_tokens(self, messages: Union[str, List[Dict], Dict], verbose: bool = False) -> int:
        """
        Count tokens for messages in a standardized way.
        
        Args:
            messages: Can be:
                - A single string (query)
                - A list of message dicts
                - A single message dict
            verbose: Whether to show detailed debug output (default: False)
                
        Returns:
            int: Token count
        """
        try:
            # Standardize input to list of clean message dicts
            clean_messages = []
            
            # Handle string input
            if isinstance(messages, str):
                clean_messages = [{"role": "user", "content": messages}]
                
            # Handle single dict
            elif isinstance(messages, dict):
                if 'role' in messages and 'content' in messages:
                    clean_messages = [{
                        "role": messages["role"],
                        "content": messages["content"]
                    }]
                    
            # Handle list of messages
            elif isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        if isinstance(msg['content'], list):
                            content = json.dumps(msg['content'])
                        else:
                            content = str(msg['content'])
                            
                        clean_messages.append({
                            "role": msg["role"],
                            "content": content
                        })

            # Debug logging only if verbose
            if verbose:
                print("\nToken counting debug:")
                for i, msg in enumerate(clean_messages):
                    print(f"Message {i+1}:")
                    print(f"Role: {msg['role']}")
                    print(f"Content preview: {str(msg['content'])[:50]}...")

            # Use Anthropic's token counter
            count_result = self.anthropic_client.messages.count_tokens(
                model=self.query_model,
                messages=clean_messages
            )
            
            token_count = count_result.input_tokens if hasattr(count_result, 'input_tokens') else 0
            
            # Only show final count if verbose
            if verbose:
                print(f"Final token count: {token_count}")
            
            return token_count

        except Exception as e:
            print(f"Error counting tokens: {e}")
            traceback.print_exc()
            return 0
        
    def count_output_tokens(self, response_text: str) -> int:
        """
        Estimates output tokens for cost tracking.
        
        Args:
            response_text: The text response from Claude
            
        Returns:
            int: Estimated token count for cost calculation
        """
        # Ensure response_text is a string
        if not isinstance(response_text, str):
            print(f"ERROR: Response text is not a string: {response_text}")
            return 0

        # Log the response text
        #print(f"DEBUG: Response text for output token estimation: {response_text}")

        char_count = len(response_text)
        estimated_tokens = char_count // 4

        print(f"\nOutput Token Estimation:")
        print(f"Characters: {char_count}")
        print(f"Estimated Tokens: {estimated_tokens}")

        return estimated_tokens

    def start_session(self):
        self.session_active = True
        self.session_start_time = datetime.now()
        
        self.haiku_tracker = {
            'input_tokens': 0,
            'output_tokens': 0,
            'cost': Decimal('0.00')
        }
        
        self.sonnet_tracker = {
            'input_tokens': 0,
            'output_tokens': 0,
            'tool_definition_tokens': 0,
            'cost': Decimal('0.00'),
            'tools_initialized': False
        }
        
        self.session = {
            'current_model': 'claude-3-5-haiku-20241022',
            'history_tokens': 0,
            'total_cost': Decimal('0.00'),
            'tools_enabled': False
        }
        
        self.tools_enabled = False
        self.tools_used_in_session = False
        
        print(f"Token tracking session started at {self.session_start_time}")
        return {
            "status": "started",
            "timestamp": self.session_start_time.isoformat()
        }

    def display_token_usage(self, query_tokens: int = None):
        print("\n┌─ Token Usage Report " + "─" * 40)
        
        if query_tokens:
            print(f"│ Current Query Tokens: {query_tokens:,}")
            print("│" + "─" * 50)
            
        print("│ Haiku Usage:")
        print(f"│   Input Tokens:  {self.haiku_tracker['input_tokens']:,}")
        print(f"│   Output Tokens: {self.haiku_tracker['output_tokens']:,}")
        print(f"│   Cost:         ${self.haiku_tracker['cost']:.4f}")
        
        print("│\n│ Sonnet Usage:")
        print(f"│   Input Tokens:  {self.sonnet_tracker['input_tokens']:,}")
        print(f"│   Output Tokens: {self.sonnet_tracker['output_tokens']:,}")
        print(f"│   Tool Def Tokens: {self.sonnet_tracker['tool_definition_tokens']:,}")
        print(f"│   Cost:         ${self.sonnet_tracker['cost']:.4f}")
        
        print("│\n│ Session Totals:")
        total_cost = self.haiku_tracker['cost'] + self.sonnet_tracker['cost']
        print(f"│   Total Cost:    ${total_cost:.4f}")
        
        if self.session_start_time:
            duration = datetime.now() - self.session_start_time
            print(f"│   Duration:      {duration.total_seconds():.1f}s")
        
        print("└" + "─" * 50)

    def update_session_costs(self, input_tokens: int, response_text: str, is_tool_use: bool = False):
        try:
            tracker = self.sonnet_tracker if is_tool_use else self.haiku_tracker
            model = self.tool_model if is_tool_use else self.query_model

            # Log the input tokens and response text
            print(f" Input tokens: {input_tokens}")
            print(f" Response text: {response_text}")

            output_tokens = self.count_output_tokens(response_text)  # Ensure correct response text

            print(f" Updating session costs - input_tokens: {input_tokens}, output_tokens: {output_tokens}")

            current_input = max(0, input_tokens)
            current_output = max(0, output_tokens)

            if is_tool_use and not self.sonnet_tracker['tools_initialized']:
                current_input += self.TOOL_COSTS['definition_overhead']
                self.sonnet_tracker['tool_definition_tokens'] = self.TOOL_COSTS['definition_overhead']
                self.sonnet_tracker['tools_initialized'] = True

            input_cost = self.calculate_token_cost(model, "input", current_input)
            output_cost = self.calculate_token_cost(model, "output", current_output)

            tracker['input_tokens'] += current_input
            tracker['output_tokens'] += current_output
            tracker['cost'] += input_cost + output_cost

            self.session['total_cost'] += input_cost + output_cost
            self.session['history_tokens'] += current_input + current_output

            print(f" Session costs updated - input_cost: {input_cost}, output_cost: {output_cost}")

            self.display_token_usage(current_input + current_output)

        except Exception as e:
            print(f"Error updating session costs: {e}")

    def calculate_token_cost(self, model: str, token_type: str, token_count: int) -> Decimal:
        try:
            if token_count < 0:
                print(f"Warning: Negative token count ({token_count}) adjusted to 0")
                token_count = 0
            
            if model not in self.MODEL_COSTS:
                print(f"Warning: Unknown model '{model}', using claude-3-5-sonnet-20241022 pricing")
                model = "claude-3-5-sonnet-20241022"
                
            if token_type not in self.MODEL_COSTS[model]:
                print(f"Warning: Unknown token type '{token_type}' for model '{model}', using 'input' pricing")
                token_type = "input"
                
            per_million_rate = self.MODEL_COSTS[model][token_type]
            if not isinstance(per_million_rate, Decimal):
                per_million_rate = Decimal(str(per_million_rate))
                
            return (per_million_rate / Decimal('1000000')) * Decimal(str(token_count))
            
        except Exception as e:
            print(f"Error calculating token cost: {e}")
            return Decimal('0')

    def get_session_summary(self) -> Dict[str, Any]:
        session_duration = datetime.now() - self.session_start_time
        minutes = session_duration.total_seconds() / 60
        
        haiku = self.haiku_tracker
        sonnet = self.sonnet_tracker
        
        return {
            "session_duration_minutes": round(minutes, 2),
            "haiku_stats": {
                "input_tokens": haiku['input_tokens'],
                "output_tokens": haiku['output_tokens'],
                "total_cost": float(haiku['cost'])
            },
            "sonnet_stats": {
                "input_tokens": sonnet['input_tokens'],
                "output_tokens": sonnet['output_tokens'],
                "tool_definition_tokens": sonnet['tool_definition_tokens'],
                "total_cost": float(sonnet['cost'])
            },
            "session_totals": {
                "total_tokens": (haiku['input_tokens'] + haiku['output_tokens'] + 
                               sonnet['input_tokens'] + sonnet['output_tokens']),
                "total_cost_usd": float(haiku['cost'] + sonnet['cost'])
            },
            "tools_currently_enabled": self.tools_enabled,
            "tools_used_in_session": self.tools_used_in_session
        }

    def enable_tools(self) -> bool:
        """Simple tool state enablement."""
        self.tools_enabled = True
        return True

    def disable_tools(self) -> bool:
        """Simple tool state disablement."""
        self.tools_enabled = False
        return True

    def record_tool_usage(self, tool_name: str) -> Dict[str, Any]:
        """Records that a tool was used."""
        self.tools_used_in_session = True
        print(f"Tool usage recorded: {tool_name}")
        return {
            "status": "recorded",
            "tool": tool_name
        }

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool and track its token usage"""
        try:
            tool = get_tool_by_name(tool_name)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")
                
            self.record_tool_usage(tool_name)
            result = tool['function'](**kwargs)
            
            if asyncio.iscoroutine(result):
                result = await result
                
            return result
            
        except Exception as e:
            print(f"Error executing tool {tool_name}: {e}")
            traceback.print_exc()
            raise

    def track_query_completion(self, used_tool: bool = False) -> Dict[str, Any]:
        """Only tracks tool usage for audio feedback purposes."""
        if not self.tools_enabled or not used_tool:
            return {
                "state_changed": False,
                "tools_active": self.tools_enabled,
                "mood": "casual"
            }
                    
        # Play tool use audio when a tool is actually used
        try:
            audio_file = random.choice(os.listdir(SOUND_PATHS['tool']['use']))
            AudioManager.play_audio(os.path.join(SOUND_PATHS['tool']['use'], audio_file))
        except Exception as e:
            print(f"Error playing tool use audio: {e}")
        
        return {
            "state_changed": False,
            "tools_active": True,
            "mood": "focused"
        }

    def tools_are_active(self) -> bool:
        return self.tools_enabled

    def handle_tool_command(self, query: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle tool enable/disable commands
        Returns (was_command, response_dict)
        """
        try:
            query_lower = query.lower().strip()
            
            # Use existing enable/disable commands from SYSTEM_STATE_COMMANDS
            enable_commands = SYSTEM_STATE_COMMANDS["tool"]["enable"]
            disable_commands = SYSTEM_STATE_COMMANDS["tool"]["disable"]
            
            has_enable = any(cmd.lower() in query_lower for cmd in SYSTEM_STATE_COMMANDS["tool"]["enable"])
            has_disable = any(cmd.lower() in query_lower for cmd in SYSTEM_STATE_COMMANDS["tool"]["disable"])
            
            if has_enable and has_disable:
                # Use last mentioned command
                last_enable_pos = max((query_lower.rfind(cmd.lower()) 
                                   for cmd in enable_commands), default=-1)
                last_disable_pos = max((query_lower.rfind(cmd.lower()) 
                                    for cmd in disable_commands), default=-1)
                
                return True, (self.disable_tools() if last_disable_pos > last_enable_pos 
                            else self.enable_tools())
                
            elif has_enable:
                return True, self.enable_tools()
            elif has_disable:
                return True, self.disable_tools()
                    
            return False, None
                    
        except Exception as e:
            print(f"Error in handle_tool_command: {e}")
            return False, {"success": False, "message": str(e), "state": "error"}

    def tools_are_active(self):
        """Simple check if tools are enabled - used by generate_response"""
        return self.tools_enabled


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
        is_enable = any(phrase in query_lower 
                       for phrase in SYSTEM_STATE_COMMANDS["tool"]["enable"])
        is_disable = any(phrase in query_lower 
                        for phrase in SYSTEM_STATE_COMMANDS["tool"]["disable"])
        
        if is_enable or is_disable:
            # Remove the tool command part from query for further analysis
            for phrase in (SYSTEM_STATE_COMMANDS["tool"]["enable"] + 
                          SYSTEM_STATE_COMMANDS["tool"]["disable"]):
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
    


    def track_query_completion(self, used_tool: bool = False) -> Dict[str, Any]:
        """Only tracks tool usage for audio feedback purposes."""
        if not self.tools_enabled or not used_tool:
            return {
                "state_changed": False,
                "tools_active": self.tools_enabled,
                "mood": "casual"
            }
                    
        # Play tool use audio when a tool is actually used
        try:
            audio_file = random.choice(os.listdir(SOUND_PATHS['tool']['use']))
            AudioManager.play_audio(os.path.join(SOUND_PATHS['tool']['use'], audio_file))
        except Exception as e:
            print(f"Error playing tool use audio: {e}")
        
        return {
            "state_changed": False,
            "tools_active": True,
            "mood": "focused"
        }

    def tools_are_active(self) -> bool:
        return self.tools_enabled

    def handle_tool_command(self, query: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle tool enable/disable commands
        Returns (was_command, response_dict)
        """
        try:
            query_lower = query.lower().strip()
            
            # Use existing enable/disable commands from SYSTEM_STATE_COMMANDS
            enable_commands = SYSTEM_STATE_COMMANDS["tool"]["enable"]
            disable_commands = SYSTEM_STATE_COMMANDS["tool"]["disable"]
            
            has_enable = any(cmd.lower() in query_lower for cmd in SYSTEM_STATE_COMMANDS["tool"]["enable"])
            has_disable = any(cmd.lower() in query_lower for cmd in SYSTEM_STATE_COMMANDS["tool"]["disable"])
            
            if has_enable and has_disable:
                # Use last mentioned command
                last_enable_pos = max((query_lower.rfind(cmd.lower()) 
                                   for cmd in enable_commands), default=-1)
                last_disable_pos = max((query_lower.rfind(cmd.lower()) 
                                    for cmd in disable_commands), default=-1)
                
                return True, (self.disable_tools() if last_disable_pos > last_enable_pos 
                            else self.enable_tools())
                
            elif has_enable:
                return True, self.enable_tools()
            elif has_disable:
                return True, self.disable_tools()
                    
            return False, None
                    
        except Exception as e:
            print(f"Error in handle_tool_command: {e}")
            return False, {"success": False, "message": str(e), "state": "error"}

    def tools_are_active(self):
        """Simple check if tools are enabled - used by generate_response"""
        return self.tools_enabled


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
        is_enable = any(phrase in query_lower 
                       for phrase in SYSTEM_STATE_COMMANDS["tool"]["enable"])
        is_disable = any(phrase in query_lower 
                        for phrase in SYSTEM_STATE_COMMANDS["tool"]["disable"])
        
        if is_enable or is_disable:
            # Remove the tool command part from query for further analysis
            for phrase in (SYSTEM_STATE_COMMANDS["tool"]["enable"] + 
                          SYSTEM_STATE_COMMANDS["tool"]["disable"]):
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
    
