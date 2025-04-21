import pygame
import json
import asyncio
import random
import time
from pathlib import Path

class DisplayManager:
    def __init__(self):
        pygame.init()
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
        
        # Define moods list
        self.moods = [
            "amused", "annoyed", "caring", "casual", "cheerful", "concerned",
            "confused", "curious", "disappointed", "embarrassed", "excited",
            "frustrated", "interested", "sassy", "scared", "surprised",
            "suspicious", "thoughtful"
        ]
        
        # Load active persona from personalities.json
        try:
            with open("personalities.json", 'r') as f:
                personas_data = json.load(f)
                active_persona = personas_data.get("active_persona", "laura")  # fallback to laura if not found
            self.base_path = Path(f'/home/user/LAURA/pygame/{active_persona}')
        except Exception as e:
            print(f"Warning: Could not load active persona from personalities.json: {e}")
            # Fallback to laura if there's any issue
            self.base_path = Path('/home/user/LAURA/pygame/laura')
        
        # Setup state paths
        self.states = {
            'listening': str(self.base_path / 'listening'),
            'idle': str(self.base_path / 'idle'),
            'sleep': str(self.base_path / 'sleep'),
            'speaking': str(self.base_path / 'speaking'),
            'thinking': str(self.base_path / 'thinking'),
            'tools': str(self.base_path / 'tools'),
            'wake': str(self.base_path / 'wake'),
            'system': str(self.base_path / 'system')
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
        """Load and cache images for all states and moods"""
        print(f"\nLoading images from: {self.base_path}")
        
        for state, directory in self.states.items():
            try:
                if state == 'speaking':
                    self.image_cache[state] = {}
                    for mood in self.moods:
                        mood_path = Path(directory) / mood
                        if mood_path.exists():
                            png_files = list(mood_path.glob('*.png'))
                            if png_files:
                                #print(f"Loading {len(png_files)} images for {state}/{mood}")
                                self.image_cache[state][mood] = [
                                    pygame.transform.scale(
                                        pygame.image.load(str(img)), 
                                        (512, 512)
                                    )
                                    for img in png_files
                                ]
                            else:
                                print(f"Warning: No PNG files found in {mood_path}")
                        else:
                            print(f"Warning: Mood directory not found: {mood_path}")
                else:
                    state_path = Path(directory)
                    if state_path.exists():
                        png_files = list(state_path.glob('*.png'))
                        if png_files:
                            #print(f"Loading {len(png_files)} images for {state}")
                            self.image_cache[state] = [
                                pygame.transform.scale(
                                    pygame.image.load(str(img)), 
                                    (512, 512)
                                )
                                for img in png_files
                            ]
                        else:
                            print(f"Warning: No PNG files found in {state_path}")
                    else:
                        print(f"Warning: State directory not found: {state_path}")
            
            except Exception as e:
                print(f"Error loading images for {state}: {e}")
        
        # Verify required states are loaded
        required_states = ['sleep', 'idle', 'speaking']
        missing_states = [state for state in required_states if state not in self.image_cache]
        
        if missing_states:
            raise RuntimeError(f"Failed to load required states: {', '.join(missing_states)}")

    def _get_image_path(self, state_type, specific_type=None):
        """
        Centralized image path resolution with fallbacks
        
        Args:
            state_type: Primary category (tools, calibration, document)
            specific_type: Subcategory (enabled, disabled, load, etc.)
        
        Returns:
            Path object to the appropriate image or directory
        """
        # First try persona-specific path
        if specific_type:
            primary_path = Path(f"{self.base_path}/system/{state_type}/{specific_type}")
        else:
            primary_path = Path(f"{self.base_path}/{state_type}")
        
        # Check if primary path exists with images
        if primary_path.exists() and any(primary_path.glob('*.png')):
            return primary_path
        
        # Fall back to Laura's resources
        if specific_type:
            fallback_path = Path(f"/home/user/LAURA/pygame/laura/system/{state_type}/{specific_type}")
        else:
            fallback_path = Path(f"/home/user/LAURA/pygame/laura/{state_type}")
        
        if fallback_path.exists() and any(fallback_path.glob('*.png')):
            return fallback_path
        
        # If all else fails, return the thinking state as last resort
        return Path(f"{self.base_path}/thinking")
        
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
    
    async def update_display_path(self, new_base_path: str):
        """
        Update the base path and reload all image directories for a new persona
        Args:
            new_base_path: Path to the new persona's pygame directory
        """
        async with self.state_lock:
            try:
                print(f"Updating display path to: {new_base_path}")
                self.base_path = Path(new_base_path)
                
                # Update state paths
                self.states = {
                    'listening': str(self.base_path / 'listening'),
                    'idle': str(self.base_path / 'idle'),
                    'sleep': str(self.base_path / 'sleep'),
                    'speaking': str(self.base_path / 'speaking'),
                    'thinking': str(self.base_path / 'thinking'),
                    'tools': str(self.base_path / 'tools'),
                    'wake': str(self.base_path / 'wake'),
                    'system': str(self.base_path / 'system')
                }
                
                # Clear existing cache
                self.image_cache.clear()
                
                # Reload all images
                self.load_image_directories()
                
                # Reset to sleep state with new images
                if 'sleep' in self.image_cache:
                    self.current_state = 'sleep'
                    self.current_mood = 'casual'
                    self.current_image = random.choice(self.image_cache['sleep'])
                    self.screen.blit(self.current_image, (0, 0))
                    pygame.display.flip()
                    self.last_image_change = time.time()
                    self.state_entry_time = time.time()
                    return True
                else:
                    print("Warning: Failed to load sleep images for new persona")
                    return False
            
            except Exception as e:
                print(f"Error updating display path: {e}")
                return False
             
    async def update_display(self, state, mood=None, transition_path=None, specific_image=None):
        async with self.state_lock:
            if mood is None:
                mood = self.current_mood

            # Skip no-change updates unless it's a transition with a specific path
            if state == self.current_state and mood == self.current_mood and transition_path is None and specific_image is None:
                return  # No change needed
            
            try:
                self.last_state = self.current_state
                self.current_state = state
                self.current_mood = mood
                
                # Handle special case for tools (enabled/disabled/use)
                if state == 'tools':
                    # Tool state should be either 'enabled', 'disabled', or 'use' if specified
                    tool_state = specific_image if specific_image in ['enabled', 'disabled', 'use'] else None
                    
                    if tool_state:
                        # Use centralized path resolution
                        tool_path = self._get_image_path('tools', tool_state)
                        
                        # Load first PNG from the directory
                        png_files = list(tool_path.glob('*.png'))
                        if png_files:
                            tool_image = pygame.transform.scale(
                                pygame.image.load(str(png_files[0])), 
                                (512, 512)
                            )
                            self.current_image = tool_image
                            self.screen.blit(self.current_image, (0, 0))
                            pygame.display.flip()
                            self.state_entry_time = time.time()
                            return
                    
                    # If we get here, there's no valid tool state image, use thinking state from current persona
                    print(f"No valid tool state image found, using thinking state for {state}/{specific_image}")
                    thinking_path = Path(f"{self.base_path}/thinking")
                    if thinking_path.exists() and any(thinking_path.glob('*.png')):
                        print(f"Using persona's thinking state as fallback for tools")
                        thinking_image = pygame.transform.scale(
                            pygame.image.load(str(list(thinking_path.glob('*.png'))[0])), 
                            (512, 512)
                        )
                        self.current_image = thinking_image
                        self.screen.blit(self.current_image, (0, 0))
                        pygame.display.flip()
                        self.state_entry_time = time.time()
                        return
                
                # Handle special case for calibration
                if state == 'calibration':
                    # Use centralized path resolution
                    calib_path = self._get_image_path('calibration')
                    
                    # Load first PNG from the directory
                    png_files = list(calib_path.glob('*.png'))
                    if png_files:
                        calib_image = pygame.transform.scale(
                            pygame.image.load(str(png_files[0])), 
                            (512, 512)
                        )
                        self.current_image = calib_image
                        self.screen.blit(self.current_image, (0, 0))
                        pygame.display.flip()
                        self.state_entry_time = time.time()
                        return
                
                # Handle special case for document operations
                if state == 'document':
                    # doctype should be either 'load' or 'unload'
                    doctype = specific_image if specific_image in ['load', 'unload'] else 'load'
                    
                    # Use centralized path resolution
                    doc_path = self._get_image_path('document', doctype)
                    
                    # Load first PNG from the directory
                    png_files = list(doc_path.glob('*.png'))
                    if png_files:
                        doc_image = pygame.transform.scale(
                            pygame.image.load(str(png_files[0])), 
                            (512, 512)
                        )
                        self.current_image = doc_image
                        self.screen.blit(self.current_image, (0, 0))
                        pygame.display.flip()
                        self.state_entry_time = time.time()
                        return
                        
                # Handle special case for persona transitions
                if transition_path is not None:
                    # If a specific image is provided, use that directly
                    if specific_image is not None:
                        specific_path = Path(specific_image)
                        if specific_path.exists() and specific_path.is_file():
                            try:
                                # Load and display the specific image
                                transition_image = pygame.transform.scale(
                                    pygame.image.load(str(specific_path)), 
                                    (512, 512)
                                )
                                self.current_image = transition_image
                                self.screen.blit(self.current_image, (0, 0))
                                pygame.display.flip()
                                self.state_entry_time = time.time()
                                return
                            except Exception as e:
                                print(f"Error loading specific image {specific_image}: {e}")
                                # Continue with directory handling
                    
                    # Load and display transition image(s) directly from the path
                    transition_path = Path(transition_path)
                    if transition_path.exists():
                        png_files = list(transition_path.glob('*.png'))
                        if png_files:
                            # If we have multiple images, use the first one for now
                            transition_image = pygame.transform.scale(
                                pygame.image.load(str(png_files[0])), 
                                (512, 512)
                            )
                            self.current_image = transition_image
                            self.screen.blit(self.current_image, (0, 0))
                            pygame.display.flip()
                            self.state_entry_time = time.time()
                            return
                        else:
                            print(f"Warning: No PNG files found in transition path: {transition_path}")
                    else:
                        print(f"Warning: Transition path not found: {transition_path}")
                        # Continue with normal state handling
                
                # Normal state handling (unchanged)
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

    async def play_transition_sequence(self, transition_path, frame_delay=0.1):
        """
        Play a sequence of transition images from a directory
        Args:
            transition_path: Path to directory containing transition frames
            frame_delay: Delay between frames in seconds
        """
        async with self.state_lock:
            try:
                transition_path = Path(transition_path)
                if not transition_path.exists():
                    print(f"Warning: Transition path not found: {transition_path}")
                    return False
                    
                png_files = sorted(list(transition_path.glob('*.png')))
                if not png_files:
                    print(f"Warning: No PNG files found in transition path: {transition_path}")
                    return False
                    
                # Play the sequence
                for frame_file in png_files:
                    frame_image = pygame.transform.scale(
                        pygame.image.load(str(frame_file)), 
                        (512, 512)
                    )
                    self.current_image = frame_image
                    self.screen.blit(self.current_image, (0, 0))
                    pygame.display.flip()
                    await asyncio.sleep(frame_delay)
                    
                return True
                    
            except Exception as e:
                print(f"Error playing transition sequence: {e}")
                return False
