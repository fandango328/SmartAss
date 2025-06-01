import asyncio
import re
from datetime import datetime
import traceback
from typing import Dict, Any, Tuple, Optional

import config # Assuming MOOD_MAPPINGS is in config.py
# Ensure save_to_log_file is imported correctly from your functions script
from function_definitions import save_to_log_file

class ResponseHandler:
    """
    Handles formatting and packaging of assistant responses for remote clients.
    
    IMMUTABLE RULE COMPLIANCE:
    - Server does NOT modify or strip mood tags from LLM text output
    - Text field in JSON response contains UNALTERED LLM output (including [mood] tags)
    - Mood field is populated by IDENTIFYING mood from text, not by modifying text
    - Client is responsible for any text processing/cleaning for TTS
    """

    def __init__(self):
        pass

    async def _process_raw_content(self, llm_text_output: Optional[str]) -> Tuple[str, str]:
        """
        Receives the direct text output from the LLM.
        Returns the LLM text output UNALTERED for the client's "text" field.
        Separately identifies the mood from this text (if tagged) for the client's "mood" field.
        
        Args:
            llm_text_output: Raw text string from LLM
            
        Returns:
            Tuple[str, str]: (unaltered_text_for_client, identified_mood_for_client)
        """
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] ResponseHandler._process_raw_content received: {repr(llm_text_output)[:200]}")
        
        # Handle None input gracefully
        if llm_text_output is None:
            llm_text_output = "I'm not sure how to respond to that."

        # The text that will be sent to the client - UNALTERED from LLM
        final_text_for_client_text_field = llm_text_output
        
        # Separately identify mood from this text for the "mood" field
        identified_mood_for_client_mood_field = "neutral"  # Default
        
        if isinstance(llm_text_output, str):
            # Look for mood tag at start of string
            mood_match_in_text = re.match(r'^\[(.*?)\]([\s\S]*)', llm_text_output, re.IGNORECASE)
            if mood_match_in_text:
                raw_mood_str = mood_match_in_text.group(1).strip()
                
                # Map mood using config if available
                if hasattr(config, 'MOOD_MAPPINGS') and isinstance(config.MOOD_MAPPINGS, dict):
                    identified_mood_for_client_mood_field = config.MOOD_MAPPINGS.get(raw_mood_str.lower(), raw_mood_str.lower())
                else:
                    identified_mood_for_client_mood_field = raw_mood_str.lower()
        
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] ResponseHandler: Text for client 'text' field (unaltered): {repr(final_text_for_client_text_field)[:200]}")
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] ResponseHandler: Mood for client 'mood' field: {identified_mood_for_client_mood_field}")

        return final_text_for_client_text_field, identified_mood_for_client_mood_field

    async def handle_response(
        self,
        assistant_content: Any,  # Expected to be the raw text string from MainLoop's LLM result
        chat_log: Any,  # Not directly used by this handler, but kept for signature compatibility
        session_capabilities: Dict[str, Any],  # Not directly used, but kept for signature compatibility
        session_id: str,
        active_persona: str
    ) -> Dict[str, Any]:
        """
        Processes assistant content and returns a response payload for the client.
        
        Args:
            assistant_content: Raw response from the LLM (should be string)
            chat_log: Conversation history (passed through but not used directly)
            session_capabilities: Client capabilities (not used in current implementation)
            session_id: Session identifier for logging and response
            active_persona: Current persona name
            
        Returns:
            Dict containing the response payload for the client
        """

        # Extract the raw LLM text string from assistant_content
        llm_text_output_string = ""
        
        if isinstance(assistant_content, str):
            llm_text_output_string = assistant_content
        elif assistant_content is None:
            llm_text_output_string = "My apologies, I seem to be without a response."  # Fallback
        else:
            # If it came wrapped in a dict by mistake, try to extract
            if isinstance(assistant_content, dict) and 'text' in assistant_content:
                llm_text_output_string = assistant_content['text']
            else:
                llm_text_output_string = str(assistant_content)  # Last resort
        
        # Process the raw content according to the immutable rule
        text_for_client_json, mood_for_client_json = await self._process_raw_content(llm_text_output_string)

        # Log the UNALTERED text (which now includes the mood tag if LLM sent it)
        message_to_log = {"role": "assistant", "content": text_for_client_json}
        try:
            save_to_log_file(message_to_log)
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] ResponseHandler: Assistant message saved to log for {session_id}")
        except Exception as e:
            print(f"[ERROR] ResponseHandler failed to save assistant message to log for {session_id}: {e}")

        # Prepare response payload for client
        response_payload_for_client = {
            "text": text_for_client_json,      # UNALTERED TEXT FROM LLM (TAG INCLUDED IF LLM SENT IT)
            "mood": mood_for_client_json,      # MOOD IDENTIFIED BY LOOKING AT THE TEXT, OR DEFAULT
            "session_id": session_id,
            "active_persona": active_persona,
        }
        
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] ResponseHandler: Payload for client for session {session_id}: {str(response_payload_for_client)[:200]}")
        return response_payload_for_client
