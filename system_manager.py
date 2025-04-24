async def generate_response(query: str) -> str:
    """
    Generate a response using the LLM with proper tool coordination and state management.
    
    Args:
        query: User's input query
        
    Returns:
        str: Processed response ready for TTS
    """
    global chat_log, last_interaction, last_interaction_check

    # ✓ Updated: Use tool registry for manager access
    system_manager = tool_registry.get_manager('system')
    if not system_manager:
        raise RuntimeError("System manager not registered with tool registry")
        
    display_manager = tool_registry.get_manager('display')
    audio_manager = tool_registry.get_manager('audio')
    token_manager = tool_registry.get_manager('token')
    notification_manager = tool_registry.get_manager('notification')

    # Update interaction timestamps
    now = datetime.now()
    last_interaction = now
    last_interaction_check = now
    token_manager.start_interaction()

    try:
        # Handle tool commands (preserved logic with registry access)
        was_command, command_response = token_manager.handle_tool_command(query)
        if was_command:
            success = command_response.get('success', False)
            mood = command_response.get('mood', 'casual')
            
            await display_manager.update_display('speaking', mood=mood if success else 'excited')
            
            if command_response.get('state') in ['enabled', 'disabled']:
                status_type = command_response['state'].lower()
                audio_file = get_random_audio('tool', f'status/{status_type}')
                if audio_file:
                    await display_manager.update_display('tools', specific_image=status_type)
                    await audio_manager.queue_audio(audio_file=audio_file, priority=True)
                    await audio_manager.wait_for_queue_empty()
            return "[CONTINUE]"

        # Log the user's message to chat history
        try:
            chat_message = {
                "role": "user",
                "content": query
            }
            save_to_log_file(chat_message)
        except Exception as e:
            print(f"Warning: Failed to save chat message to log: {e}")
            traceback.print_exc()
        
        already_added = (
            len(chat_log) > 0 and 
            chat_log[-1]["role"] == "user" and
            chat_log[-1]["content"] == query
        )
        
        if not already_added:
            chat_log.append(storage_message)

        # ✓ Updated: Tool execution check with registry
        use_tools = token_manager.tools_are_active()
        relevant_tools = []
