import asyncio
from typing import Any, Dict, List, Optional, Callable

class InputOrchestrator:
    """
    Async orchestrator for managing user and system input.
    Routes input to the main_loop, attaches output mode preference (text/audio/both),
    and handles session state. Does NOT run tool/function calls itself.
    """

    def __init__(
        self,
        main_loop_handler: Callable[[Dict[str, Any]], None],
        *,
        default_output_mode: str = "text",
    ):
        """
        :param main_loop_handler: Function or coroutine to pass input event to main loop for processing.
        :param default_output_mode: Default output route if not specified per input (text, audio, both).
        """
        self.main_loop_handler = main_loop_handler
        self.default_output_mode = default_output_mode
        self.input_queue: asyncio.Queue = asyncio.Queue()
        self.running = False

    async def start(self):
        """Start the orchestrator event loop."""
        self.running = True
        print("Orchestrator started. Awaiting input...")
        while self.running:
            input_event = await self.get_next_input()
            if input_event is None:
                continue
            await self.process_input(input_event)

    async def get_next_input(self) -> Optional[Dict[str, Any]]:
        """Pop the next input event from the queue."""
        try:
            input_event = await self.input_queue.get()
            return input_event
        except Exception as e:
            print(f"[Orchestrator] Error fetching input: {e}")
            return None

    async def add_input(self, input_event: Dict[str, Any]):
        """
        Add new input event from any producer (text, voice, webhook).
        input_event should include:
            - type: "text" | "voice" | "webhook"
            - content: str
            - files: Optional[List[Union[str, bytes, ...]]]
            - meta: Optional[dict]
            - output_mode: Optional["text" | "audio" | "both"]
        """
        await self.input_queue.put(input_event)

    async def process_input(self, input_event: Dict[str, Any]):
        """
        Pass input event to main_loop handler with correct output mode.
        """
        # Determine output mode: explicit > meta > default
        output_mode = (
            input_event.get("output_mode") or
            (input_event.get("meta", {}) or {}).get("output_mode") or
            self.default_output_mode
        )
        input_event["output_mode"] = output_mode

        print(f"[Orchestrator] Routing input ({input_event.get('type')}) | Output: {output_mode}")
        await self.main_loop_handler(input_event)

    def stop(self):
        """Stop the orchestrator loop."""
        self.running = False

# Example producer: terminal/console
async def text_input_producer(orchestrator: InputOrchestrator):
    while orchestrator.running:
        user_input = input("You: ")
        if user_input.strip().lower() in ("exit", "quit"):
            orchestrator.stop()
            break
        await orchestrator.add_input({"type": "text", "content": user_input})

# Example webhook input producer (skeleton)
async def webhook_input_producer(orchestrator: InputOrchestrator):
    """
    This simulates a webhook HTTP handler.
    In a real system, use FastAPI and call orchestrator.add_input() for each request.
    """
    import random
    example_webhook_events = [
        {
            "type": "webhook",
            "content": "Summarize attached document.",
            "files": ["/tmp/example.pdf"],
            "meta": {"source": "webhook", "output_mode": "text"},
        },
        {
            "type": "webhook",
            "content": "Read this invoice aloud.",
            "files": ["/tmp/invoice.pdf"],
            "meta": {"source": "webhook", "output_mode": "audio"},
        },
        {
            "type": "webhook",
            "content": "Give me both text and audio summary.",
            "files": ["/tmp/summary.docx"],
            "meta": {"source": "webhook", "output_mode": "both"},
        }
    ]
    # Simulate webhook events
    while orchestrator.running:
        await asyncio.sleep(10)
        event = random.choice(example_webhook_events)
        print(f"[Webhook Producer] Received webhook event: {event['content']}")
        await orchestrator.add_input(event)

# Example main_loop handler (stub)
async def main_loop_handler(input_event: Dict[str, Any]):
    """
    Stub main_loop handler; in real use, this would:
      - Manage conversation state and context
      - Call LLM/tool handlers as needed
      - Route output to appropriate device(s) based on 'output_mode'
    """
    content = input_event.get("content", "")
    output_mode = input_event.get("output_mode", "text")
    files = input_event.get("files", [])
    print(f"\n[Main Loop] User input: {content}")
    if files:
        print(f"[Main Loop] Files: {files}")
    print(f"[Main Loop] Output mode: {output_mode}\n")
    # Simulated response routing
    if output_mode in ("text", "both"):
        print("[Main Loop] (Display) Assistant: Here is your text reply.")
    if output_mode in ("audio", "both"):
        print("[Main Loop] (TTS) Assistant: [Audio playback simulated]")

if __name__ == "__main__":
    orchestrator = InputOrchestrator(main_loop_handler, default_output_mode="text")

    async def main():
        await asyncio.gather(
            orchestrator.start(),
            text_input_producer(orchestrator),
            webhook_input_producer(orchestrator)
        )

    asyncio.run(main())