import gradio as gr
import asyncio
import numpy as np
import time
import os
import json
from pathlib import Path
from datetime import datetime
import base64
import sys

# Add import for the LAURA main script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# We'll import LAURA components as needed - these would need to be adapted
# from your existing scripts
# from LAURA_gradio import generate_response, audio_manager, tts_handler, etc.

class SVGAvatarManager:
    """Manages SVG avatar animations and state transitions"""
    
    def __init__(self):
        self.moods = {
            "casual": {"eyes": "normal", "mouth": "smile", "eyebrows": "neutral"},
            "happy": {"eyes": "bright", "mouth": "big_smile", "eyebrows": "raised"},
            "serious": {"eyes": "focused", "mouth": "straight", "eyebrows": "lowered"},
            "thinking": {"eyes": "looking_up", "mouth": "slight_smile", "eyebrows": "raised"},
            "listening": {"eyes": "attentive", "mouth": "slightly_open", "eyebrows": "neutral"}
        }
        self.current_state = "idle"
        self.current_mood = "casual"
        self.is_speaking = False
        
    def get_avatar_html(self, state=None, mood=None, is_speaking=None):
        """
        Generate HTML with SVG avatar animation based on state and mood
        """
        # Update state attributes if provided
        if state is not None:
            self.current_state = state
        if mood is not None:
            self.current_mood = mood
        if is_speaking is not None:
            self.is_speaking = is_speaking
            
        # For now, using a placeholder SVG - this would be replaced with your character SVG
        svg_content = self._create_laura_svg()
        
        # Add animation script for mouth movements and state transitions
        js_animation = self._create_animation_script()
        
        # Create container with SVG and animation scripts
        html = f"""
        <div class="laura-avatar-container">
            <div class="laura-avatar">
                {svg_content}
            </div>
            <div class="laura-state-indicator">{self.current_state.capitalize()}</div>
            {js_animation}
        </div>
        <style>
            .laura-avatar-container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 400px;
                background: linear-gradient(180deg, #e6f7ff 0%, #c8eaff 100%);
                border-radius: 10px;
                position: relative;
                overflow: hidden;
            }}
            .laura-avatar {{
                width: 300px;
                height: 300px;
                position: relative;
            }}
            .laura-state-indicator {{
                margin-top: 10px;
                font-size: 18px;
                font-weight: bold;
                background-color: rgba(255, 255, 255, 0.7);
                padding: 5px 15px;
                border-radius: 20px;
            }}
        </style>
        """
        return html
        
    def _create_laura_svg(self):
        """Create SVG for LAURA character with separate paths for facial features"""
        # This is a simplified placeholder - you would replace this with your actual character SVG
        # with proper mouth, eyes, and eyebrow elements that can be animated
        
        # Using inline SVG for demonstration - this would be replaced with your SVG content
        # Important: Add proper IDs to the elements that need animation (mouth, eyes, eyebrows)
        svg = f"""
        <svg viewBox="0 0 300 400" xmlns="http://www.w3.org/2000/svg">
            <!-- Background/Face -->
            <rect width="300" height="400" fill="#e6f7ff" rx="5" ry="5"/>
            <circle cx="150" cy="150" r="100" fill="#fff0e6" stroke="#333" stroke-width="2"/>
            
            <!-- Hair -->
            <path d="M50,80 Q150,30 250,80 L250,170 Q150,200 50,170 Z" fill="#8B4513" id="laura-hair"/>
            
            <!-- Eyes -->
            <g id="laura-eyes">
                <circle cx="110" cy="130" r="15" fill="#fff" stroke="#000" stroke-width="2"/>
                <circle cx="110" cy="130" r="8" fill="#4991D4" id="laura-eye-left"/>
                
                <circle cx="190" cy="130" r="15" fill="#fff" stroke="#000" stroke-width="2"/>
                <circle cx="190" cy="130" r="8" fill="#4991D4" id="laura-eye-right"/>
            </g>
            
            <!-- Eyebrows -->
            <g id="laura-eyebrows">
                <path d="M90,110 Q110,100 130,110" stroke="#8B4513" stroke-width="3" fill="none" id="laura-eyebrow-left"/>
                <path d="M170,110 Q190,100 210,110" stroke="#8B4513" stroke-width="3" fill="none" id="laura-eyebrow-right"/>
            </g>
            
            <!-- Mouth - with shapes that can be animated -->
            <path d="M110,180 Q150,190 190,180" stroke="#000" stroke-width="2" fill="#fff0e6" id="laura-mouth"/>
            
            <!-- LAURA Text -->
            <text x="150" y="350" text-anchor="middle" font-size="24" font-weight="bold">L.A.U.R.A.</text>
        </svg>
        """
        return svg
    
    def _create_animation_script(self):
        """Create JavaScript for SVG animations based on current state and mood"""
        # Set speaking state for the animation
        speaking_state = "true" if self.is_speaking else "false"
        
        # This script handles the mouth animation and mood transitions
        script = f"""
        <script>
            // State variables
            const isSpeaking = {speaking_state};
            const currentMood = "{self.current_mood}";
            const currentState = "{self.current_state}";
            let mouthAnimation;
            
            // Initialize when document is loaded
            document.addEventListener('DOMContentLoaded', function() {{
                const mouth = document.getElementById('laura-mouth');
                const leftEye = document.getElementById('laura-eye-left');
                const rightEye = document.getElementById('laura-eye-right');
                const leftEyebrow = document.getElementById('laura-eyebrow-left');
                const rightEyebrow = document.getElementById('laura-eyebrow-right');
                
                if (!mouth || !leftEye || !rightEye || !leftEyebrow || !rightEyebrow) {{
                    console.error('Could not find all SVG elements needed for animation');
                    return;
                }}
                
                // Apply mood-based styling
                applyMoodStyling(currentMood);
                
                // Start speaking animation if needed
                if (isSpeaking) {{
                    startSpeakingAnimation();
                }}
                
                // Add state-specific animations
                if (currentState === "listening") {{
                    // Subtle eye movement for listening state
                    animateListening();
                }} else if (currentState === "thinking") {{
                    // Eye movement for thinking state
                    animateThinking();
                }}
            }});
            
            function startSpeakingAnimation() {{
                const mouth = document.getElementById('laura-mouth');
                if (!mouth) return;
                
                // Animate mouth during speech - alternating between closed and open
                let mouthOpen = false;
                mouthAnimation = setInterval(() => {{
                    if (mouthOpen) {{
                        // Closed mouth
                        mouth.setAttribute('d', 'M110,180 Q150,190 190,180');
                    }} else {{
                        // Open mouth
                        mouth.setAttribute('d', 'M110,180 Q150,200 190,180');
                    }}
                    mouthOpen = !mouthOpen;
                }}, 150);
                
                // Stop animation after 5 seconds unless we get updates from backend
                setTimeout(() => {{
                    clearInterval(mouthAnimation);
                    mouth.setAttribute('d', 'M110,180 Q150,190 190,180');
                }}, 5000);
            }}
            
            function applyMoodStyling(mood) {{
                const leftEye = document.getElementById('laura-eye-left');
                const rightEye = document.getElementById('laura-eye-right');
                const leftEyebrow = document.getElementById('laura-eyebrow-left');
                const rightEyebrow = document.getElementById('laura-eyebrow-right');
                const mouth = document.getElementById('laura-mouth');
                
                if (!leftEye || !rightEye || !leftEyebrow || !rightEyebrow || !mouth) return;
                
                switch(mood) {{
                    case 'happy':
                        leftEyebrow.setAttribute('d', 'M90,105 Q110,95 130,105');
                        rightEyebrow.setAttribute('d', 'M170,105 Q190,95 210,105');
                        mouth.setAttribute('d', 'M110,180 Q150,200 190,180');
                        break;
                    case 'serious':
                        leftEyebrow.setAttribute('d', 'M90,105 Q110,100 130,105');
                        rightEyebrow.setAttribute('d', 'M170,105 Q190,100 210,105');
                        mouth.setAttribute('d', 'M110,180 Q150,180 190,180');
                        break;
                    case 'thinking':
                        leftEyebrow.setAttribute('d', 'M90,100 Q110,90 130,100');
                        rightEyebrow.setAttribute('d', 'M170,100 Q190,90 210,100');
                        mouth.setAttribute('d', 'M110,180 Q150,185 190,180');
                        break;
                    default: // casual
                        leftEyebrow.setAttribute('d', 'M90,110 Q110,100 130,110');
                        rightEyebrow.setAttribute('d', 'M170,110 Q190,100 210,110');
                        mouth.setAttribute('d', 'M110,180 Q150,190 190,180');
                }}
            }}
            
            function animateListening() {{
                const leftEye = document.getElementById('laura-eye-left');
                const rightEye = document.getElementById('laura-eye-right');
                if (!leftEye || !rightEye) return;
                
                // Subtle eye movements to indicate active listening
                setInterval(() => {{
                    // Random small eye movements
                    const offsetX = Math.random() * 2 - 1;
                    const offsetY = Math.random() * 2 - 1;
                    
                    leftEye.setAttribute('cx', 110 + offsetX);
                    leftEye.setAttribute('cy', 130 + offsetY);
                    rightEye.setAttribute('cx', 190 + offsetX);
                    rightEye.setAttribute('cy', 130 + offsetY);
                }}, 500);
            }}
            
            function animateThinking() {{
                const leftEye = document.getElementById('laura-eye-left');
                const rightEye = document.getElementById('laura-eye-right');
                if (!leftEye || !rightEye) return;
                
                // Look up and around occasionally
                setInterval(() => {{
                    // More pronounced eye movements for thinking
                    leftEye.setAttribute('cy', 127);
                    rightEye.setAttribute('cy', 127);
                    
                    setTimeout(() => {{
                        leftEye.setAttribute('cy', 130);
                        rightEye.setAttribute('cy', 130);
                    }}, 1000);
                }}, 3000);
            }}
        </script>
        """
        return script

class GradioInterface:
    def __init__(self):
        self.avatar_manager = SVGAvatarManager()
        self.conversation_mode = True
        self.chat_history = []
        
    def build(self):
        """Build and return the Gradio interface"""
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
            
            # Main interface container with two columns
            with gr.Row(elem_id="laura-main-container"):
                # Left column - Avatar and controls
                with gr.Column(scale=1, elem_id="laura-main-interface"):
                    # Avatar display with initial state
                    avatar_html = gr.HTML(
                        self.avatar_manager.get_avatar_html(state="idle", mood="casual"),
                        elem_id="laura-avatar"
                    )
                    
                    # File upload area
                    file_drop = gr.File(
                        label="Drop files here for context",
                        file_types=["pdf", "txt", "docx"],
                        elem_id="laura-file-drop"
                    )
                    
                    # Conversation mode toggle
                    conversation_toggle = gr.Checkbox(
                        label="Enable conversation follow-ups",
                        value=True,
                        info="When enabled, LAURA will listen for follow-up questions"
                    )
                    
                    # Text input as alternative to voice
                    text_input = gr.Textbox(
                        label="Type your query",
                        placeholder="Ask a question or give a command...",
                        lines=2
                    )
                    
                    submit_btn = gr.Button("Send", variant="primary")
                
                # Right column - Conversation display
                with gr.Column(scale=1):
                    # Conversation display
                    conversation_display = gr.Chatbot(
                        label="Conversation",
                        elem_id="laura-conversation",
                        height=400
                    )
                    
                    # Status display
                    with gr.Row():
                        context_display = gr.Markdown("Context: None")
                        status_display = gr.Markdown("Status: Ready 🟢")
            
            # Bottom section - Recent tools and tabs
            with gr.Row():
                recent_tools_md = gr.Markdown(
                    "Recent Tools: Calendar | Email | Tasks | Settings | Memory: No topics loaded",
                    elem_id="laura-tools-bar"
                )
            
            # Tabs for additional functionality
            with gr.Tabs() as tabs:
                with gr.TabItem("Home"):
                    gr.Markdown("""
                    ### Quick Actions
                    - Check calendar
                    - Send email
                    - Manage tasks
                    - System settings
                    """)
                    
                with gr.TabItem("Configuration"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Voice Settings")
                            wake_word = gr.Textbox(label="Wake Word", value="Hey LAURA")
                            voice_sensitivity = gr.Slider(
                                label="Voice Detection Sensitivity",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.7,
                                step=0.05
                            )
                        with gr.Column():
                            gr.Markdown("### Interface Settings")
                            theme_dropdown = gr.Dropdown(
                                label="Theme",
                                choices=["Light", "Dark", "System Default"],
                                value="Light"
                            )
                            animation_toggle = gr.Checkbox(
                                label="Enable Animations",
                                value=True
                            )
                
                with gr.TabItem("Integrations"):
                    gr.Markdown("""
                    ### Connected Services
                    
                    Connect LAURA to your preferred services.
                    """)
                    
                    with gr.Row():
                        google_btn = gr.Button("Connect Google")
                        microsoft_btn = gr.Button("Connect Microsoft")
                        api_btn = gr.Button("Manage API Keys")
                
                with gr.TabItem("Persona"):
                    with gr.Column():
                        gr.Markdown("### Persona Settings")
                        
                        with gr.Row():
                            with gr.Column():
                                formality = gr.Slider(
                                    label="Formality",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.5,
                                    step=0.1
                                )
                                
                                verbosity = gr.Slider(
                                    label="Verbosity",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.6,
                                    step=0.1
                                )
                                
                            with gr.Column():
                                creativity = gr.Slider(
                                    label="Creativity",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.7,
                                    step=0.1
                                )
                                
                                helpfulness = gr.Slider(
                                    label="Helpfulness",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.9,
                                    step=0.1
                                )
                        
                        persona_profile = gr.Dropdown(
                            label="Saved Profiles",
                            choices=["Default", "Professional", "Casual", "Technical"],
                            value="Default"
                        )
                        
                        save_profile_btn = gr.Button("Save Current Profile")
                
                with gr.TabItem("User Profile"):
                    gr.Markdown("### User Information")
                    
                    with gr.Column():
                        user_name = gr.Textbox(label="Your Name", value="User")
                        user_email = gr.Textbox(label="Primary Email")
                        user_preferences = gr.CheckboxGroup(
                            label="Communication Preferences",
                            choices=["Email Notifications", "Calendar Alerts", "Task Reminders"],
                            value=["Calendar Alerts"]
                        )
                
                with gr.TabItem("Tasks & Projects"):
                    gr.Markdown("### Task Management")
                    
                    task_list = gr.Dataframe(
                        headers=["Task", "Due Date", "Priority", "Status"],
                        datatype=["str", "str", "str", "str"],
                        row_count=5,
                        col_count=(4, "fixed"),
                        value=[
                            ["Complete project report", "2023-05-01", "High", "In Progress"],
                            ["Review marketing materials", "2023-05-03", "Medium", "Not Started"],
                            ["Schedule team meeting", "2023-05-05", "Low", "Not Started"],
                            ["Update client presentation", "2023-05-10", "High", "Not Started"],
                            ["Research competitor products", "2023-05-15", "Medium", "In Progress"]
                        ]
                    )
                    
                    with gr.Row():
                        add_task_btn = gr.Button("Add Task")
                        delete_task_btn = gr.Button("Delete Selected")
                        
                    gr.Markdown("### Projects")
                    
                    project_list = gr.Dataframe(
                        headers=["Project", "Deadline", "Status", "Completion"],
                        datatype=["str", "str", "str", "str"],
                        row_count=3,
                        col_count=(4, "fixed"),
                        value=[
                            ["Website Redesign", "2023-06-30", "Active", "45%"],
                            ["Marketing Campaign", "2023-05-15", "Active", "80%"],
                            ["Product Launch", "2023-07-01", "Planning", "15%"]
                        ]
                    )
                
                with gr.TabItem("Multi-Device Management"):
                    gr.Markdown("### Connected Devices")
                    
                    device_list = gr.Dataframe(
                        headers=["Device Name", "Type", "Status", "Last Connected"],
                        datatype=["str", "str", "str", "str"],
                        row_count=3,
                        col_count=(4, "fixed"),
                        value=[
                            ["Home Office", "Desktop", "Online", "Now"],
                            ["Laptop", "Laptop", "Offline", "2 hours ago"],
                            ["Mobile Phone", "Mobile", "Online", "5 minutes ago"]
                        ]
                    )
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### API Access")
                            api_port = gr.Number(label="API Port", value=5000)
                            api_toggle = gr.Checkbox(label="Enable Remote API Access", value=False)
                        
                        with gr.Column():
                            gr.Markdown("### Device Authorization")
                            auth_token = gr.Textbox(
                                label="Authorization Token", 
                                value="••••••••••••••••",
                                type="password"
                            )
                            gen_token_btn = gr.Button("Generate New Token")
                
                with gr.TabItem("System Monitor"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### System Resources")
                            cpu_usage = gr.Label(label="CPU Usage", value="23%")
                            memory_usage = gr.Label(label="Memory Usage", value="1.2GB / 8GB")
                            storage = gr.Label(label="Storage", value="5.4GB / 32GB")
                        
                        with gr.Column():
                            gr.Markdown("### Performance")
                            response_time = gr.Label(label="Avg. Response Time", value="0.8s")
                            uptime = gr.Label(label="System Uptime", value="3d 5h 42m")
                    
                    with gr.Row():
                        gr.Markdown("### System Log")
                        system_log = gr.Textbox(
                            label="",
                            value="[INFO] System started\n[INFO] Voice detection initialized\n[INFO] All components loaded successfully",
                            lines=8,
                            max_lines=8,
                            interactive=False
                        )
            
            # Event handlers
            def handle_text_submit(message, history):
                """Handle text input submission"""
                if not message.strip():
                    return "", history
                
                # Add user message to history
                history.append((message, None))
                
                # This would connect to the LAURA backend in a real implementation
                # For now, just echo back a response
                response = f"I received: {message}"
                
                # Update history with assistant response
                history[-1] = (history[-1][0], response)
                
                return "", history
            
            def update_avatar_state(state="listening", mood="casual", is_speaking=False):
                """Update avatar state and return HTML"""
                return avatar_manager.get_avatar_html(state, mood, is_speaking)
            
            def handle_file_upload(file_path):
                """Handle file upload and extract context"""
                if file_path is None:
                    return "Context: None"
                
                file_name = Path(file_path).name
                return f"Context: File loaded - {file_name}"
            
            # Connect event handlers
            text_input.submit(
                handle_text_submit,
                inputs=[text_input, conversation_display],
                outputs=[text_input, conversation_display]
            ).then(
                update_avatar_state,
                inputs=[gr.State("thinking"), gr.State("casual"), gr.State(False)],
                outputs=[avatar_html]
            ).then(
                lambda: time.sleep(1), # Simulate processing time
                None,
                None
            ).then(
                update_avatar_state,
                inputs=[gr.State("speaking"), gr.State("casual"), gr.State(True)],
                outputs=[avatar_html]
            ).then(
                lambda: time.sleep(3), # Simulate speaking time
                None,
                None
            ).then(
                update_avatar_state,
                inputs=[gr.State("idle"), gr.State("casual"), gr.State(False)],
                outputs=[avatar_html]
            )
            
            submit_btn.click(
                handle_text_submit,
                inputs=[text_input, conversation_display],
                outputs=[text_input, conversation_display]
            ).then(
                update_avatar_state,
                inputs=[gr.State("thinking"), gr.State("casual"), gr.State(False)],
                outputs=[avatar_html]
            ).then(
                lambda: time.sleep(1), # Simulate processing time
                None,
                None
            ).then(
                update_avatar_state,
                inputs=[gr.State("speaking"), gr.State("casual"), gr.State(True)],
                outputs=[avatar_html]
            ).then(
                lambda: time.sleep(3), # Simulate speaking time
                None,
                None
            ).then(
                update_avatar_state,
                inputs=[gr.State("idle"), gr.State("casual"), gr.State(False)],
                outputs=[avatar_html]
            )
            
            file_drop.upload(
                handle_file_upload,
                inputs=[file_drop],
                outputs=[context_display]
            )
            
            # Add footer
            gr.HTML(
                """<footer>LAURA - Language & Automation User Response Agent | Version 1.0 | © 2023</footer>"""
            )
        
        return app

    async def process_query(self, query):
        """Process query using the LAURA backend (to be implemented)"""
        # This would connect to your existing LAURA script
        # For now, return a placeholder response
        await asyncio.sleep(1)  # Simulate processing time
        return f"Response to: {query}"

# Initialize and launch the interface
if __name__ == "__main__":
    interface = GradioInterface()
    app = interface.build()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
