import gradio as gr
import numpy as np

# Placeholder functions – connect these to your real backend logic!
def process_text_or_audio(user_input, audio_input, history, persona, enabled_tools):
    if user_input:
        response_text = f"Assistant: (Text) {user_input}"
        response_audio = None  # Replace with TTS output
    elif audio_input is not None:
        response_text = "Assistant: (Audio) [Transcribed] Hello from audio input."
        response_audio = None  # Replace with TTS output
    else:
        response_text = "Please enter text or record audio."
        response_audio = None

    # Update history (simulate multi-turn)
    if history is None:
        history = []
    history.append((user_input if user_input else "[voice]", response_text))
    return history, response_audio, "awake", ["Meeting at 3pm", "Reminder: Take a break"]

def toggle_persona(persona):
    # This would set the persona in your backend
    return f"Persona set to {persona}"

def toggle_tool(tool, enabled):
    # This would enable/disable a tool in your backend
    return f"Tool {tool} {'enabled' if enabled else 'disabled'}"

with gr.Blocks() as demo:
    gr.Markdown("# 🗣️ Laura Smart Assistant Web UI")

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="Conversation History", height=400, type="messages")

            with gr.Row():
                text_input = gr.Textbox(label="Type your command", lines=1)
                audio_input = gr.Audio(type="numpy", label="Or speak to Laura")
            send_btn = gr.Button("Send")

            audio_output = gr.Audio(label="Assistant's TTS Response", interactive=False)
            system_state = gr.Textbox(label="System State", value="idle", interactive=False)
        with gr.Column(scale=1):
            gr.Markdown("### Persona / Mood")
            persona_choice = gr.Radio(["Laura", "Helper", "Professional"], label="Persona", value="Laura")
            persona_status = gr.Textbox(label="Current Persona", value="Laura", interactive=False)

            gr.Markdown("---")
            gr.Markdown("### Tools/Plugins")
            tool1 = gr.Checkbox(label="Calendar Tool", value=True)
            tool2 = gr.Checkbox(label="Email Tool", value=True)
            tool3 = gr.Checkbox(label="Document Tool", value=False)

            gr.Markdown("---")
            gr.Markdown("### Notifications & Events")
            notification_panel = gr.HighlightedText(label="Upcoming Events / Notifications",
                                                    value=[("Meeting at 3pm", "event")])

    # Callbacks and logic wiring
    def on_send(user_input, audio_input, chatbot_hist, persona, tool1, tool2, tool3):
        enabled_tools = []
        if tool1: enabled_tools.append("calendar")
        if tool2: enabled_tools.append("email")
        if tool3: enabled_tools.append("document")
        hist, audio, state, notifications = process_text_or_audio(user_input, audio_input, chatbot_hist, persona, enabled_tools)
        return hist, None, audio, state, notifications

    send_btn.click(
        on_send,
        inputs=[text_input, audio_input, chatbot, persona_choice, tool1, tool2, tool3],
        outputs=[chatbot, text_input, audio_output, system_state, notification_panel]
    )

    # Persona toggling
    persona_choice.change(
        lambda p: p,
        inputs=persona_choice,
        outputs=persona_status
    )

    # Tool toggles could wire to backend here as needed
    tool1.change(lambda x: None, inputs=tool1, outputs=None)
    tool2.change(lambda x: None, inputs=tool2, outputs=None)
    tool3.change(lambda x: None, inputs=tool3, outputs=None)

if __name__ == "__main__":
    demo.launch()
