    async def _handle_persona_command(self, action: str, arguments: str = None) -> bool:
        """
        Handle persona-related commands using pre-recorded audio files
        Returns True if command succeeded, False if it failed
        """
        # Initialize personas_data outside try block so it's accessible in catch-all error handler
        personas_data = None  
        
        try:
            # Step 1: Update display to 'tools' state to show we're processing a command
            try:
                await self.display_manager.update_display('tools')
            except Exception as e:
                print(f"Warning: Failed to update display to tools state: {e}")
            
            success = False
            
            # Step 2: Load or create personalities configuration
            persona_path = "personalities.json"
            try:
                with open(persona_path, 'r') as f:
                    personas_data = json.load(f)
            except FileNotFoundError:
                # Create default config if file doesn't exist
                personas_data = {
                    "personas": {
                        "laura": {
                            "voice": "L.A.U.R.A.",
                            "system_prompt": "You are Laura (Language & Automation User Response Agent), a professional and supportive AI-powered smart assistant."
                        }
                    },
                    "active_persona": "laura"
                }
                with open(persona_path, 'w') as f:
                    json.dump(personas_data, f, indent=2)
            
            # Step 3: Get available personas from config
            available_personas = personas_data.get("personas", {})
            
            # Step 4: Process the switch command
            if action == "switch":
                # Normalize input for case-insensitive matching
                normalized_input = arguments.strip().lower() if arguments else ""
                
                # Step 5: Find matching persona in available personas
                target_persona = None
                for key in available_personas:
                    if key.lower() in normalized_input:
                        target_persona = key
                        break
                
                # Step 6: If matching persona found, process the switch
                if target_persona:
                    # Step 7: Check for persona's audio files
                    audio_path = Path(f'/home/user/LAURA/sounds/{target_persona}/persona_sentences')
                    
                    if audio_path.exists():
                        # Get list of MP3 files in persona directory
                        audio_files = list(audio_path.glob('*.mp3'))
                        
                        if audio_files:
                            try:
                                # Step 8: Update display path first
                                new_base_path = str(Path(f'/home/user/LAURA/pygame/{target_persona.lower()}'))
                                await self.display_manager.update_display_path(new_base_path)
                                
                                # Step 9: Show tools state with new persona
                                await self.display_manager.update_display('tools')
                                
                                # Step 10: Update active persona in config and save
                                personas_data["active_persona"] = target_persona
                                with open(persona_path, 'w') as f:
                                    json.dump(personas_data, f, indent=2)
                                
                                # Step 11: Play switch announcement audio
                                chosen_audio = str(random.choice(audio_files))
                                await self.audio_manager.play_audio(chosen_audio)
                                await self.audio_manager.wait_for_audio_completion()
                                
                                # Update configuration and reinitialize TTS
                                import config
                                importlib.reload(config)
                                try:
                                    # Change the active persona name and data
                                    config.ACTIVE_PERSONA = target_persona
                                    config.ACTIVE_PERSONA_DATA = personas_data["personas"][target_persona]
                                    # Update the voice that will be used
                                    config.VOICE = personas_data["personas"][target_persona].get("voice", "L.A.U.R.A.")
                                    # Update the prompt that will be used
                                    new_prompt = personas_data["personas"][target_persona].get("system_prompt", "You are an AI assistant.")
                                    config.SYSTEM_PROMPT = f"{new_prompt}\n\n{config.UNIVERSAL_SYSTEM_PROMPT}"
                                    

                                    # Reinitialize TTS handler with new voice
                                    from secret import ELEVENLABS_KEY
                                    new_config = {
                                        "TTS_ENGINE": config.TTS_ENGINE,
                                        "ELEVENLABS_KEY": ELEVENLABS_KEY,
                                        "VOICE": config.VOICE,
                                        "ELEVENLABS_MODEL": config.ELEVENLABS_MODEL,
                                    }
                                    self.tts_handler = TTSHandler(new_config)
                                    
                                    print(f"Switched to persona: {target_persona}")
                                    print(f"Using voice: {config.VOICE}")
                                    print(f"TTS handler reinitialized with new voice")
                                    print(f"System prompt updated and reloaded")
                                except Exception as e:
                                    print(f"Error switching persona: {e}")
                                    success = False
                                
                            except Exception as e:
                                print(f"Error during persona switch: {e}")
                                success = False
                        else:
                            print(f"No audio files found in {audio_path}")
                            success = False
                    else:
                        print(f"Audio path not found: {audio_path}")
                        success = False
                else:
                    print(f"Persona '{arguments}' not found")
                    success = False
            
            # Step 11: Return to listening state
            try:
                await self.display_manager.update_display('listening')
            except Exception as e:
                print(f"Warning: Failed to update display to listening state: {e}")
            
            return success
            
