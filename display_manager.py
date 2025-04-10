import pygame
import asyncio
import random
import time
from pathlib import Path

class DisplayManager:
    def __init__(self):
        pygame.init()
        self.base_path = Path('/home/user/LAURA/pygame')
        self.screen = None
        self.image_cache = {}
        self.current_state = 'sleep'
        self.current_mood = 'casual'
        self.last_state = None
        self.last_mood = None
        self.current_image = None
        self.last_image_change = None
        self.state_entry_time = None
        self.initialized = False
        self.state_lock = asyncio.Lock()
        self.moods = [
            "caring", "casual", "cheerful", "concerned", "confused", "curious",
            "disappointed", "embarrassed", "sassy", "surprised", "suspicious", "thoughtful"
        ]
        self.states = {
            'listening': str(self.base_path / 'listening'),
            'idle': str(self.base_path / 'idle'),
            'sleep': str(self.base_path / 'sleep'),
            'speaking': str(self.base_path / 'speaking'),
            'thinking': str(self.base_path / 'thinking'),
            'tools': str(self.base_path / 'tools'),
            'wake': str(self.base_path / 'wake')
        }
        self.setup_display()
        self.load_image_directories()
        
        # Do proper initial display
        if 'sleep' in self.image_cache:
            self.current_image = random.choice(self.image_cache['sleep'])
            self.screen.blit(self.current_image, (0, 0))
            pygame.display.flip()
            # Set timings AFTER first image is displayed
            self.last_image_change = time.time()
            self.state_entry_time = time.time()
            self.initialized = True
        else:
            raise RuntimeError("Failed to load sleep images")
        
    def setup_display(self):
        self.screen = pygame.display.set_mode((512, 512))
        pygame.display.set_caption("LAURA")
        
    def load_image_directories(self):
        #print("\nLoading image directories...")
        for state, directory in self.states.items():
            #print(f"\nChecking state: {state}")
            #print(f"Path: {directory}")
            
            if state == 'speaking':
                self.image_cache[state] = {}
                for mood in self.moods:
                    mood_path = Path(directory) / mood
                    print(f"Checking mood: {mood}")
                    #print(f"Path: {mood_path}")
                    if mood_path.exists():
                        png_files = list(mood_path.glob('*.png'))
                        #print(f"Found {len(png_files)} images")
                        if png_files:
                            self.image_cache[state][mood] = [
                                pygame.transform.scale(pygame.image.load(str(img)), (512, 512))
                                for img in png_files
                            ]
            else:
                state_path = Path(directory)
                if state_path.exists():
                    png_files = list(state_path.glob('*.png'))
                    #print(f"Found {len(png_files)} images")
                    if png_files:
                        self.image_cache[state] = [
                            pygame.transform.scale(pygame.image.load(str(img)), (512, 512))
                            for img in png_files
                        ]

    async def update_display(self, state, mood=None):
        async with self.state_lock:
            if mood is None:
                mood = self.current_mood

            if state == self.current_state and mood == self.current_mood:
                return  # No change needed
                
            try:
                self.last_state = self.current_state
                self.current_state = state
                self.current_mood = mood
                
                if state == 'speaking':
                    if mood not in self.image_cache[state]:
                        print(f"Warning: Invalid mood '{mood}', using casual mood")
                        mood = 'casual'
                    self.current_image = random.choice(self.image_cache[state][mood])
                elif state in self.image_cache:
                    self.current_image = random.choice(self.image_cache[state])
                else:
                    print(f"Error: Invalid state '{state}'")
                    return
                    
                self.screen.blit(self.current_image, (0, 0))
                pygame.display.flip()
                
                self.state_entry_time = time.time()
                if state in ['idle', 'sleep']:
                    self.last_image_change = self.state_entry_time
                    
            except Exception as e:
                print(f"Error updating display: {e}")

    async def rotate_background(self):
        while not self.initialized:
            await asyncio.sleep(0.1)
            
        while True:
            try:
                current_time = time.time()
                
                # Only proceed with rotation if in stable idle/sleep state
                if self.current_state in ['idle', 'sleep']:
                    # Check if we're in a state transition
                    if current_time - self.state_entry_time < 1.0:
                        await asyncio.sleep(0.1)
                        continue
                        
                    time_diff = current_time - self.last_image_change
                    if time_diff >= 15:
                        available_images = self.image_cache[self.current_state]
                        if len(available_images) > 1:
                            # Lock the rotation process
                            async with self.state_lock:
                                current_options = [img for img in available_images 
                                                if img != self.current_image]
                                if current_options:
                                    new_image = random.choice(current_options)
                                    self.current_image = new_image
                                    self.screen.blit(self.current_image, (0, 0))
                                    pygame.display.flip()
                                    self.last_image_change = current_time
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"Error in rotate_background: {e}")
                await asyncio.sleep(1)
            
    def cleanup(self):
        pygame.quit()
