async def play_queued_notifications():
    """
    Play queued notifications and handle notification state management.
    Supports both calendar events and configurable notifications.
    """
    global NOTIFICATION_STATES, PENDING_NOTIFICATIONS
    
    if notification_queue.empty() and not PENDING_NOTIFICATIONS:
        return

    previous_state = display_manager.current_state
    previous_mood = display_manager.current_mood
    current_time = datetime.now()

    # First handle any immediate notifications in the queue
    while not notification_queue.empty():
        notification_data = await notification_queue.get()
        
        # Expected format: dict with type, id, message, requires_clear, reminder_interval
        notification_type = notification_data.get('type', 'calendar')
        notification_id = notification_data.get('id')
        message = notification_data.get('message', '')
        requires_clear = notification_data.get('requires_clear', False)
        reminder_interval = notification_data.get('reminder_interval', 10)  # Default 10 minutes

        # Add to pending if requires clear
        if requires_clear:
            PENDING_NOTIFICATIONS[notification_id] = {
                "type": notification_type,
                "message": message,
                "created": current_time,
                "requires_clear": True,
                "last_reminder": current_time,
                "reminder_interval": reminder_interval
            }

        try:
            await audio_manager.wait_for_audio_completion()
            await display_manager.update_display('speaking', mood='casual')
            
            audio = tts_handler.generate_audio(message)
            with open("notification.mp3", "wb") as f:
                f.write(audio)
            
            await audio_manager.play_audio("notification.mp3")
            await audio_manager.wait_for_audio_completion()
            
        except Exception as e:
            print(f"Error playing notification: {e}")

    # Handle pending notifications that need reminders
    pending_to_remove = set()
    for notification_id, notification_data in PENDING_NOTIFICATIONS.items():
        if not notification_data['requires_clear']:
            pending_to_remove.add(notification_id)
            continue

        time_since_last = (current_time - notification_data['last_reminder']).total_seconds() / 60
        if time_since_last >= notification_data['reminder_interval']:
            reminder_message = f"Reminder: {notification_data['message']}"
            
            try:
                await audio_manager.wait_for_audio_completion()
                await display_manager.update_display('speaking', mood='casual')
                
                audio = tts_handler.generate_audio(reminder_message)
                with open("notification.mp3", "wb") as f:
                    f.write(audio)
                
                await audio_manager.play_audio("notification.mp3")
                await audio_manager.wait_for_audio_completion()
                
                # Update last reminder time
                PENDING_NOTIFICATIONS[notification_id]['last_reminder'] = current_time
                
            except Exception as e:
                print(f"Error playing reminder: {e}")

    # Remove cleared notifications
    for notification_id in pending_to_remove:
        PENDING_NOTIFICATIONS.pop(notification_id, None)

    # Restore previous display state
    await display_manager.update_display(previous_state, mood=previous_mood)
