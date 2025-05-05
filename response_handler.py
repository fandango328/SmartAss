import asyncio
import base64
from typing import Dict, Any, Optional

from core_functions import process_response_content, save_to_log_file
from tts_handler import TTSHandler

class ResponseHandler:
    """
    Handles formatting and packaging of assistant responses based on session/device output capabilities.
    - Prepares a response payload suitable for sending to a remote client (e.g., via WebSocket).
    - Only includes modalities requested by the client (text, audio, mood).
    - Does NOT handle any direct output (audio playback, display) locally.
    - Does NOT log user messages.
    """

    def __init__(
        self,
        tts_handler: Optional[TTSHandler] = None,
    ):
        self.tts_handler = tts_handler

    async def handle_response(
        self,
        assistant_content: Any,
        chat_log: Any,
        session_capabilities: Dict[str, Any],
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Processes assistant content and returns a response payload for the client.

        Args:
            assistant_content: The raw assistant response (string, dict, or block)
            chat_log: Conversation history (may be used by process_response_content)
            session_capabilities: Dict with output capabilities, e.g., {"output": ["text", "audio", "display"]}
            session_id: Unique identifier for the session (for logging, tracking, etc.)

        Returns:
            Dict[str, Any]: Payload to be sent to the client device via WebSocket/protocol.
        """
        output_modalities = session_capabilities.get("output", ["text"])
        # Always process the assistant response before delivery
        # process_response_content should return (text, formatted_text, mood)
        #   - text: display/response text for user
        #   - formatted_text: for TTS (may be same as text)
        #   - mood: mood tag or display metadata
        text, tts_text, mood = await process_response_content(
            assistant_content,
            chat_log,
        )

        # Save assistant message to log file
        save_to_log_file(session_id, text, sender="assistant")

        response_payload = {}
        if "text" in output_modalities:
            response_payload["text"] = text

        if "audio" in output_modalities and self.tts_handler:
            try:
                audio_data = self.tts_handler.generate_audio(tts_text)
                # Base64 encode the audio data for WebSocket transport
                response_payload["tts_audio"] = base64.b64encode(audio_data).decode("utf-8")
                response_payload["tts_text"] = tts_text
            except Exception as e:
                response_payload["audio_error"] = f"TTS generation failed: {str(e)}"

        if "display" in output_modalities and mood:
            # Only provide the mood/display tag for the client to interpret
            response_payload["mood"] = mood

        # Always include session_id for correlation
        response_payload["session_id"] = session_id

        return response_payload