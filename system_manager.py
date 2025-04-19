async def _handle_persona_command(self, action: str, arguments: str = None) -> bool:
    """Handle persona-related commands with transition animations"""
    try:
        # Log the command details
        print(f"\nDEBUG: Handling persona command - Action: {action}, Arguments: {arguments}")
        
        # Import config module at function start
        import config as config_module
        
        # Step 1: Update to system state with current persona's "out" animation
        current_persona = config_module.ACTIVE_PERSONA.lower()
        out_path = f"/home/user/LAURA/pygame/{current_persona}/system/persona/out"
        default_image = "/home/user/LAURA/pygame/laura/system/persona/dont_touch_this_image.png"
        
        print(f"DEBUG: Transitioning from {current_persona} with path: {out_path}")
        
        # Check if persona out directory exists with images
        out_path_dir = Path(out_path)
        if out_path_dir.exists() and any(out_path_dir.glob('*.png')):
            # Use transition path to display persona-specific "out" animation
            await self.display_manager.update_display('system', transition_path=str(out_path_dir))
        else:
            # Fall back to default system state if no transition images
            print(f"Warning: No persona exit animations found at {out_path}")
            await self.display_manager.update_display('thinking')
        
        # Load personality configuration
        persona_path = "personalities.json"
        try:
            with open(persona_path, 'r') as f:
                personas_data = json.load(f)
        except FileNotFoundError:
            print("DEBUG: Creating default personalities configuration")
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
        
        # Find target persona
        if action == "switch":
            # If arguments is empty or None, handle it gracefully
            if not arguments:
                print("DEBUG: No persona specified in switch command")
                return False
                
            normalized_input = arguments.strip().lower()
            target_persona = None
            
            # Check if arguments exactly matches a persona key
            if normalized_input in personas_data.get("personas", {}):
                target_persona = normalized_input
            else:
                # Try to find a matching persona
                for key in personas_data.get("personas", {}):
                    if key.lower() == normalized_input:
                        target_persona = key
                        break
            
            if target_persona:
                print(f"DEBUG: Switching to persona: {target_persona}")
                
                # Step 2: Update the base display path
                new_base_path = str(Path(f'/home/user/LAURA/pygame/{target_persona.lower()}'))
                await self.display_manager.update_display_path(new_base_path)
                
                # Step 3: Show incoming animation
                in_path = f"/home/user/LAURA/pygame/{target_persona.lower()}/system/persona/in"
                print(f"DEBUG: Loading incoming animation from: {in_path}")
                
                # Check if persona in directory exists with images
                in_path_dir = Path(in_path)
                if in_path_dir.exists() and any(in_path_dir.glob('*.png')):
                    # Use transition path to display persona-specific "in" animation
                    await self.display_manager.update_display('system', transition_path=str(in_path_dir))
                    # Add a small delay to show the animation
                    await asyncio.sleep(0.5)
                else:
                    print(f"Warning: No persona entry animations found at {in_path}")
                
                # Step 4: Update configuration
                try:
                    # Update active persona
                    personas_data["active_persona"] = target_persona
                    with open(persona_path, 'w') as f:
                        json.dump(personas_data, f, indent=2)
                    
                    # Reload config and update system
                    importlib.reload(config_module)
                    
                    config_module.ACTIVE_PERSONA = target_persona
                    config_module.ACTIVE_PERSONA_DATA = personas_data["personas"][target_persona]
                    config_module.VOICE = personas_data["personas"][target_persona].get("voice", "L.A.U.R.A.")
                    new_prompt = personas_data["personas"][target_persona].get("system_prompt", "You are an AI assistant.")
                    config_module.SYSTEM_PROMPT = f"{new_prompt}\n\n{config_module.UNIVERSAL_SYSTEM_PROMPT}"
                    
                    # Reinitialize TTS handler
                    from secret import ELEVENLABS_KEY
                    new_config = {
                        "TTS_ENGINE": config_module.TTS_ENGINE,
                        "ELEVENLABS_KEY": ELEVENLABS_KEY,
                        "VOICE": config_module.VOICE,
                        "ELEVENLABS_MODEL": config_module.ELEVENLABS_MODEL,
                    }
                    self.tts_handler = TTSHandler(new_config)
                    
                    print(f"DEBUG: Successfully switched to persona: {target_persona}")
                    print(f"DEBUG: Using voice: {config_module.VOICE}")
                    
                    # Step 5: Transition to listening state
                    await asyncio.sleep(0.1)  # Small buffer for state change
                    await self.display_manager.update_display('listening')
                    
                    return True
                    
                except Exception as e:
                    print(f"ERROR: Failed to update configuration: {e}")
                    traceback.print_exc()
                    return False
            else:
                print(f"ERROR: Persona '{arguments}' not found")
                return False
        
        return False
        
    except Exception as e:
        print(f"ERROR: Persona command failed: {e}")
        traceback.print_exc()
        return False
