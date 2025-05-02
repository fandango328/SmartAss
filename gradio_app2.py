import gradio as gr
import asyncio
import numpy as np
import time
import os
from pathlib import Path
import sys
from display_manager import DisplayManager

display_manager = DisplayManager()

def refresh_avatar_image(state="idle", mood="casual"):
    display_manager.update_display(state, mood=mood)
    display_manager.aura.update()
    return display_manager.aura.get_surface_image()

class GradioInterface:
    def __init__(self):
        self.conversation_mode = True
        self.chat_history = []
        
    def build(self):
        css = """
        #laura-main-interface {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        #laura-main-container {
            min-height: 600px;
        }
        #laura-avatar {
            min-height: 400px;
        }
        #laura-file-drop {
            margin-top: 10px;
            padding: 10px;
            border: 2px dashed #ccc;
            border-radius: 5px;
            text-align: center;
        }
        .tab-content {
            padding: 20px;
        }
        footer {
            margin-top: 20px;
            text-align: center;
            color: #666;
        }
        """

        with gr.Blocks(css=css) as app:
            gr.Markdown("# LAURA - Language & Automation User Response Agent")

            with gr.Row(elem_id="laura-main-container"):
                with gr.Column(scale=1, elem_id="laura-main-interface"):
                    avatar_img = gr.Image(
                        value=display_manager.aura.get_surface_image(),
                        shape=(600, 600),
                        elem_id="laura-avatar"
                    )
                    file_drop = gr.File(
                        label="Drop files here for context",
                        file_types=["pdf", "txt", "docx"],
                        elem_id="laura-file-drop"
                    )
                    conversation_toggle = gr.Checkbox(
                        label="Enable conversation follow-ups",
                        value=True,
                        info="When enabled, LAURA will listen for follow-up questions"
                    )
                    text_input = gr.Textbox(
                        label="Type your query",
                        placeholder="Ask a question or give a command...",
                        lines=2
                    )
                    submit_btn = gr.Button("Send", variant="primary")

                with gr.Column(scale=1):
                    conversation_display = gr.Chatbot(
                        label="Conversation",
                        elem_id="laura-conversation",
                        height=400
                    )
                    with gr.Row():
                        context_display = gr.Markdown("Context: None")
                        status_display = gr.Markdown("Status: Ready 🟢")

            with gr.Row():
                recent_tools_md = gr.Markdown(
                    "Recent Tools: Calendar | Email | Tasks | Settings | Memory: No topics loaded",
                    elem_id="laura-tools-bar"
                )

            # ... (Tabs unchanged, as before) ...

            # Event handlers
            def handle_text_submit(message, history):
                if not message.strip():
                    return "", history, display_manager.aura.get_surface_image()
                history.append((message, None))
                # Here, insert backend LLM logic if available
                response = f"I received: {message}"
                history[-1] = (history[-1][0], response)
                # Simulate a mood/state change:
                state, mood = "speaking", "cheerful"
                display_manager.update_display(state, mood=mood)
                display_manager.aura.update()
                return "", history, display_manager.aura.get_surface_image()

            def handle_file_upload(file_path):
                if file_path is None:
                    return "Context: None"
                file_name = Path(file_path).name
                return f"Context: File loaded - {file_name}"

            text_input.submit(
                handle_text_submit,
                inputs=[text_input, conversation_display],
                outputs=[text_input, conversation_display, avatar_img]
            )
            submit_btn.click(
                handle_text_submit,
                inputs=[text_input, conversation_display],
                outputs=[text_input, conversation_display, avatar_img]
            )
            file_drop.upload(
                handle_file_upload,
                inputs=[file_drop],
                outputs=[context_display]
            )

            gr.HTML(
                """<footer>LAURA - Language & Automation User Response Agent | Version 1.0 | © 2023</footer>"""
            )

        return app

    async def process_query(self, query):
        await asyncio.sleep(1)
        return f"Response to: {query}"

if __name__ == "__main__":
    interface = GradioInterface()
    app = interface.build()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
