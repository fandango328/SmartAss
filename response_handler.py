import asyncio
from typing import Dict, Any

from core_functions import process_response_content
from function_definitions import save_to_log_file 
import system_manager
import display_manager
import audio_manager
import notification_manager
class ResponseHandler:
    """
    Handles formatting and packaging of assistant responses for remote clients.
    - Only includes text, mood, session_id, and active_persona.
    - Does NOT handle or include any TTS/audio fields.
    """

    def __init__(self):
        pass

    async def handle_response(
        self,
        assistant_content: Any,
        chat_log: Any,
        session_capabilities: Dict[str, Any],
        session_id: str,
        active_persona: str,
        system_manager: Any,    
        display_manager: Any,
        audio_manager: Any,
        notification_manager: Any
    ) -> Dict[str, Any]:
        """
        Processes assistant content and returns a response payload for the client.

        Args:
            assistant_content: The raw assistant response (string, dict, or block)
            chat_log: Conversation history (may be used by process_response_content)
            session_capabilities: Dict with output capabilities (ignored in this implementation)
            session_id: Unique identifier for the session (for logging, tracking, etc.)
            active_persona: The current persona (must always be included in response)

        Returns:
            Dict[str, Any]: Payload to be sent to the client device.
        """
        # Process the assistant response before delivery
        # process_response_content returns (text, formatted_text, mood)
        text, _, mood = await process_response_content(
            assistant_content,
            chat_log,
        )

        # Save assistant message to log file
        save_to_log_file(session_id, text, sender="assistant")

        # Always include text, mood, session_id, and active_persona
        response_payload = {
            "text": text,
            "mood": mood,
            "session_id": session_id,
            "active_persona": active_persona,
        }

        return response_payload
