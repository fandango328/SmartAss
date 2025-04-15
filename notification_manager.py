#!/usr/bin/env python3

import asyncio
from datetime import datetime
from pathlib import Path
import json
from colorama import Fore
import random
import os

class NotificationManager:
    def __init__(self, audio_manager, config_path="config/notifications.json"):
        self.audio_manager = audio_manager
        self.last_calendar_check = datetime.now()
        self.notification_queue = asyncio.Queue()
        self.active_reminders = {}
        self.pending_notifications = {}
        
        # Load notification configuration
        self.config = self._load_notification_config(config_path)
        
        # Set up notification paths
        self.notification_base_dir = Path("sounds/notifications")
        self.is_processing = False
        
    def _load_notification_config(self, config_path):
        """Load notification configuration from JSON file"""
        default_config = {
            "notification_types": {
                "daily_medicine": {
                    "intervals": [10, 20, 30],
                    "over30_interval": 15,
                    "requires_clear": True,
                    "audio_path": "daily_medicine",
                    "default_schedule": {
                        "time": "07:30",
                        "days": "all"
                    }
                },
                "calendar": {
                    "intervals": [5, 15, 30],
                    "requires_clear": False,
                    "audio_path": "calendar",
                    "max_reminders": 3
                }
            },
            "default_settings": {
                "check_interval": 60,
                "reminder_check_interval": 30
            }
        }
        
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
            return default_config
        except Exception as e:
            print(f"Error loading notification config: {e}")
            return default_config
            
    async def start(self):
        """Start the notification manager background tasks"""
        self.is_processing = True
        asyncio.create_task(self._process_notification_queue())
        
    async def stop(self):
        """Stop the notification manager"""
        self.is_processing = False
        
    def queue_notification(self, text, priority=1, sound_file=None):
        """Add a notification to the queue"""
        notification = {
            'text': text,
            'priority': priority,
            'timestamp': datetime.now(),
            'sound_file': sound_file
        }
        self.notification_queue.put_nowait(notification)
        
    async def _process_notification_queue(self):
        """Background task to process queued notifications"""
        while self.is_processing:
            try:
                # Check if we can process notifications
                if not self.audio_manager.is_speaking and not self.audio_manager.is_listening:
                    # Get notification if available
                    try:
                        notification = self.notification_queue.get_nowait()
                        await self._play_notification(notification)
                    except asyncio.QueueEmpty:
                        pass
                        
                # Short sleep to prevent CPU spinning
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing notification queue: {e}")
                await asyncio.sleep(1)
                
    async def _play_notification(self, notification):
        """Play a single notification"""
        try:
            print(f"{Fore.CYAN}Playing notification: {notification['text']}{Fore.WHITE}")
            
            # Use provided sound file or select default
            sound_file = notification.get('sound_file')
            if not sound_file:
                sound_file = self._get_random_notification_sound()
                
            if sound_file and Path(sound_file).exists():
                await self.audio_manager.play_audio(sound_file)
                await self.audio_manager.wait_for_audio_completion()
            
        except Exception as e:
            print(f"Error playing notification: {e}")
            
    def _get_random_notification_sound(self):
        """Get a random notification sound file"""
        sound_files = list(self.notification_sounds_dir.glob("*.mp3"))
        return str(random.choice(sound_files)) if sound_files else None
        
    async def check_calendar_events(self, calendar_manager):
        """Check for upcoming calendar events"""
        now = datetime.now()
        
        # Only check calendar every minute
        if (now - self.last_calendar_check).total_seconds() < self.calendar_check_interval:
            return
            
        self.last_calendar_check = now
        
        try:
            upcoming_events = calendar_manager.get_upcoming_events()
            if upcoming_events:
                for event in upcoming_events:
                    notification_text = f"Upcoming event: {event['summary']} at {event['start_time']}"
                    self.queue_notification(
                        text=notification_text,
                        priority=2,
                        sound_file="calendar_notification.mp3"
                    )
        except Exception as e:
            print(f"Error checking calendar events: {e}")
            
    async def has_pending_notifications(self):
        """Check if there are any pending notifications"""
        return not self.notification_queue.empty()
        
    async def get_next_notification(self):
        """Get the next notification if available"""
        try:
            return self.notification_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
            
    async def process_pending_notifications(self):
        """Process all pending notifications"""
        while not self.notification_queue.empty():
            notification = await self.get_next_notification()
            if notification:
                await self._play_notification(notification)
