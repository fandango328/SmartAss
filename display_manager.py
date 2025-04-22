import pygame
import json
import asyncio
import random
import time
from pathlib import Path

class DisplayManager:
    """
    Manages the visual display system for LAURA with persona-specific resource handling.
    
    The display system follows a hierarchical structure:
    /pygame/{persona}/
    ├── system/                    # System-level states and transitions
    │   ├── tools/                # Tool-related states
    │   │   ├── enabled/          # Tool enabled state images
    │   │   ├── disabled/         # Tool disabled state images
    │   │   └── use/             # Tool use action images
    │   ├── calibration/          # Calibration-related images
    │   ├── document/             # Document operation images
    │   │   ├── load/            # Document loading state
    │   │   └── unload/          # Document unloading state
    │   └── persona/             # Persona transition animations
    │       ├── in/              # Persona activation animations
    │       └── out/             # Persona deactivation animations
    ├── speaking/                 # Speech state with mood variations
    │   ├── casual/              # Default mood
    │   ├── excited/             # Elevated mood
    │   └── .../                 # Other mood states
    └── {other_states}/          # Basic state images
    
    The system implements a fallback chain:
    1. Try persona-specific resources first
    2. Fall back to Laura's resources if not found
    3. Ultimate fallback to thinking state
    """

    def __init__(self):
        """
        Initialize the display system with proper state management and resource loading.
        
        The initialization process:
        1. Sets up the pygame display system
        2. Loads active persona configuration
        3. Establishes state paths and validation rules
        4. Initializes the image cache
        5. Sets up the initial display state
        
        Raises:
            RuntimeError: If critical resources (like sleep state images) can't be loaded
        """
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
        
        # Define valid moods for speaking state
        # These must match subdirectory names in the speaking/ directory
        self.moods = [
            "amused", "annoyed", "caring", "casual", "cheerful", "concerned",
            "confused", "curious", "disappointed", "embarrassed", "excited",
            "frustrated", "interested", "sassy", "scared", "surprised",
            "suspicious", "thoughtful"
        ]
        
        # Load active persona configuration
        # This determines which resource set to use as primary
        try:
            with open("personalities.json", 'r') as f:
                personas_data = json.load(f)
                active_persona = personas_data.get("active_persona", "laura")
            self.base_path = Path(f'/home/user/LAURA/pygame/{active_persona}')
        except Exception as e:
            print(f"Warning: Could not load active persona from personalities.json: {e}")
            self.base_path = Path('/home/user/LAURA/pygame/laura')
        
        # Define state paths and validation rules
        # This ensures consistent path resolution and state validation
        self.states = {
            # Base states - Direct paths under persona directory
            'listening': str(self.base_path / 'listening'),
            'idle': str(self.base_path / 'idle'),
            'sleep': str(self.base_path / 'sleep'),
            'speaking': str(self.base_path / 'speaking'),
            'thinking': str(self.base_path / 'thinking'),
            'wake': str(self.base_path / 'wake'),
            
            # System states - Under system/ directory
            'tools': str(self.base_path / 'system' / 'tools'),
            'calibration': str(self.base_path / 'system' / 'calibration'),
            'document': str(self.base_path / 'system' / 'document'),
            'persona': str(self.base_path / 'system' / 'persona'),
            'system': str(self.base_path / 'system')
        }
        
        # Define valid subtypes for system states
        # This prevents invalid state/subtype combinations
        self.system_subtypes = {
            'tools': ['enabled', 'disabled', 'use'],
            'document': ['load', 'unload'],
            'persona': ['in', 'out'],
            'calibration': []  # No subtypes for calibration
        }
        
        self.setup_display()
        self.load_image_directories()
        
        # Initialize display with sleep state
        # This is critical for proper system startup
        if 'sleep' in self.image_cache:
            self.current_image = random.choice(self.image_cache['sleep'])
            self.screen.blit(self.current_image, (0, 0))
            pygame.display.flip()
            self.last_image_change = time.time()
            self.state_entry_time = time.time()
            self.initialized = True
        else:
            raise RuntimeError("Failed to load sleep images - critical resource missing")
    
    def setup_display(self):
        """
        Configure the pygame display surface.
        All images are scaled to 512x512 for consistency.
        """
        self.screen = pygame.display.set_mode((512, 512))
        pygame.display.set_caption("LAURA")
    
    def load_image_directories(self):
        """
        Load and cache all state images with proper mood handling for speaking state.
        
        The loading process:
        1. For speaking state:
           - Creates mood-specific subdictionaries
           - Loads images for each mood variation
        2. For other states:
           - Loads all PNG files from state directory
           - Scales images to display size
        
        Images are cached to prevent repeated disk access and ensure smooth transitions.
        """
        print(f"\nLoading images from: {self.base_path}")
        
        for state, directory in self.states.items():
            try:
                if state == 'speaking':
                    # Speaking state requires mood-specific image sets
                    self.image_cache[state] = {}
                    for mood in self.moods:
                        mood_path = Path(directory) / mood
                        if mood_path.exists():
                            png_files = list(mood_path.glob('*.png'))
                            if png_files:
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
                    # Other states have direct image sets
                    state_path = Path(directory)
                    if state_path.exists():
                        png_files = list(state_path.glob('*.png'))
                        if png_files:
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
        
        # Verify critical states are available
        required_states = ['sleep', 'idle', 'speaking']
        missing_states = [state for state in required_states if state not in self.image_cache]
        
        if missing_states:
            raise RuntimeError(f"Failed to load required states: {', '.join(missing_states)}")

    def _get_system_image_path(self, state_type: str, subtype: str = None, specific_file: str = None) -> Path:
        """
        Resolve system state image paths with proper fallback chain.
        
        The resolution process:
        1. Try persona-specific path first:
           /pygame/{persona}/system/{state_type}/{subtype}
        2. Fall back to Laura's resources:
           /pygame/laura/system/{state_type}/{subtype}
        3. Ultimate fallback to thinking state if nothing else works
        
        Args:
            state_type: Primary category (tools, calibration, document)
            subtype: Optional subcategory (enabled, disabled, load, etc.)
            specific_file: Optional specific file to look for
            
        Returns:
            Path: Object pointing to appropriate image/directory
            
        Raises:
            RuntimeError: If no valid image path can be found in fallback chain
        """
        paths_to_try = []
        
        # Build primary path (persona-specific)
        if subtype:
            primary = self.base_path / 'system' / state_type / subtype
        else:
            primary = self.base_path / 'system' / state_type
        paths_to_try.append(primary)
        
        # Add Laura fallback paths
        laura_base = Path('/home/user/LAURA/pygame/laura')
        if subtype:
            laura_path = laura_base / 'system' / state_type / subtype
        else:
            laura_path = laura_base / 'system' / state_type
        paths_to_try.append(laura_path)
        
        # Try specific file first if provided
        if specific_file:
            for base_path in paths_to_try:
                specific_path = base_path / specific_file
                if specific_path.exists():
                    return specific_path
        
        # Otherwise look for any valid PNG in the directories
        for path in paths_to_try:
            if path.exists():
                png_files = list(path.glob('*.png'))
                if png_files:
                    return path
        
        # Ultimate fallback - thinking state
        thinking_path = self.base_path / 'thinking'
        if thinking_path.exists() and any(thinking_path.glob('*.png')):
            return thinking_path
            
        # If even thinking isn't available, try Laura's thinking
        laura_thinking = laura_base / 'thinking'
        if laura_thinking.exists() and any(laura_thinking.glob('*.png')):
            return laura_thinking
            
        raise RuntimeError(f"No valid image path found for {state_type}/{subtype}")
        
    async def rotate_background(self):
        """
        Periodically rotate background images for idle/sleep states.
        
        The rotation process:
        1. Only activates for idle/sleep states
        2. Ensures minimum display time for each image
        3. Randomly selects new images from available set
        4. Uses state lock to prevent conflicts with state changes
        """
        while not self.initialized:
            await asyncio.sleep(0.1)
        
        while True:
            try:
                current_time = time.time()
                
                # Only rotate in stable idle/sleep states
                if self.current_state in ['idle', 'sleep']:
                    # Prevent rotation during state transitions
                    if current_time - self.state_entry_time < 1.0:
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Check if it's time for rotation
                    time_diff = current_time - self.last_image_change
                    if time_diff >= 15:  # Rotate every 15 seconds
                        available_images = self.image_cache[self.current_state]
                        if len(available_images) > 1:
                            # Ensure thread safety during rotation
                            async with self.state_lock:
                                # Avoid repeating the current image
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
        """
        Clean up pygame resources on shutdown.
        """
        pygame.quit()
    
    async def update_display_path(self, new_base_path: str):
        """
        Update resource paths and reload images for persona changes.
        
        The update process:
        1. Updates base path to new persona
        2. Reconstructs all state paths
        3. Clears existing image cache
        4. Reloads all images for new persona
        5. Resets to sleep state with new images
        
        Args:
            new_base_path: Path to new persona's pygame directory
            
        Returns:
            bool: Success status of the update
        """
        async with self.state_lock:
            try:
                print(f"Updating display path to: {new_base_path}")
                self.base_path = Path(new_base_path)
                
                # Update all state paths for new persona
                self.states = {
                    # Base states
                    'listening': str(self.base_path / 'listening'),
                    'idle': str(self.base_path / 'idle'),
                    'sleep': str(self.base_path / 'sleep'),
                    'speaking': str(self.base_path / 'speaking'),
                    'thinking': str(self.base_path / 'thinking'),
                    'wake': str(self.base_path / 'wake'),
                    
                    # System states
                    'tools': str(self.base_path / 'system' / 'tools'),
                    'calibration': str(self.base_path / 'system' / 'calibration'),
                    'document': str(self.base_path / 'system' / 'document'),
                    'persona': str(self.base_path / 'system' / 'persona'),
                    'system': str(self.base_path / 'system')
                }
                
                # Reload all images for new persona
                self.image_cache.clear()
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
        """
        Update display state with comprehensive state and transition handling.
        
        The update process:
        1. Validates and normalizes input parameters
        2. Handles special system states (tools, calibration, document)
        3. Manages persona transitions
        4. Provides fallback mechanisms for missing resources
        
        Args:
            state: Target display state
            mood: Optional mood for speaking state
            transition_path: Optional path for transition animations
            specific_image: Optional specific image to display
        """
        async with self.state_lock:
            if mood is None:
                mood = self.current_mood

            # Skip redundant updates unless transitioning
            if (state == self.current_state and 
                mood == self.current_mood and 
                transition_path is None and 
                specific_image is None):
                return

            try:
                self.last_state = self.current_state
                self.current_state = state
                self.current_mood = mood

                # Handle system states with proper subtype validation
                if state in ['tools', 'calibration', 'document']:
                    try:
                        # Validate and determine system state parameters
                        if state == 'tools':
                            state_type = 'tools'
                            subtype = specific_image if specific_image in self.system_subtypes['tools'] else None
                        elif state == 'calibration':
                            state_type = 'calibration'
                            subtype = None
                        else:  # document
                            state_type = 'document'
                            subtype = specific_image if specific_image in self.system_subtypes['document'] else None

                        # Get path with fallback chain
                        image_path = self._get_system_image_path(state_type, subtype)
                        
                        # Load and display appropriate image
                        if image_path.is_file():
                            system_image = pygame.transform.scale(
                                pygame.image.load(str(image_path)),
                                (512, 512)
                            )
                        else:
                            # Handle directory of images
                            png_files = list(image_path.glob('*.png'))
                            if not png_files:
                                raise FileNotFoundError(f"No PNG files found in {image_path}")
                            system_image = pygame.transform.scale(
                                pygame.image.load(str(png_files[0])),
                                (512, 512)
                            )
                        
                        self.current_image = system_image
                        self.screen.blit(self.current_image, (0, 0))
                        pygame.display.flip()
                        self.state_entry_time = time.time()
                        return
                        
                    except Exception as e:
                        print(f"Error handling system state {state}: {e}")
                        # Fall through to normal state handling
                
                # Handle persona transitions with proper sequencing
                if transition_path is not None:
                    try:
                        transition_path = Path(transition_path)
                        if transition_path.exists():
                            # Handle specific transition image if provided
                            if specific_image:
                                specific_path = Path(specific_image)
                                if specific_path.exists() and specific_path.is_file():
                                    transition_image = pygame.transform.scale(
                                        pygame.image.load(str(specific_image)),
                                        (512, 512)
                                    )
                                    self.current_image = transition_image
                                    self.screen.blit(self.current_image, (0, 0))
                                    pygame.display.flip()
                                    self.state_entry_time = time.time()
                                    return
                            
                            # Use first image from transition sequence
                            png_files = list(transition_path.glob('*.png'))
                            if png_files:
                                transition_image = pygame.transform.scale(
                                    pygame.image.load(str(png_files[0])),
                                    (512, 512)
                                )
                                self.current_image = transition_image
                                self.screen.blit(self.current_image, (0, 0))
                                pygame.display.flip()
                                self.state_entry_time = time.time()
                                return
                    except Exception as e:
                        print(f"Error handling transition: {e}")
                        # Fall through to normal state handling

                # Handle normal states with mood variations
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
                traceback.print_exc()
                # Try to recover to thinking state
                try:
                    if 'thinking' in self.image_cache:
                        self.current_image = random.choice(self.image_cache['thinking'])
                        self.screen.blit(self.current_image, (0, 0))
                        pygame.display.flip()
                except:
                    pass

    async def play_transition_sequence(self, transition_path, frame_delay=0.1):
        """
        Play animated transition sequences with proper timing.
        
        The sequence process:
        1. Validates transition path and available frames
        2. Plays frames in sequence with specified delay
        3. Maintains thread safety during playback
        
        Args:
            transition_path: Path to directory containing transition frames
            frame_delay: Delay between frames in seconds
            
        Returns:
            bool: Success status of transition playback
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
                    
                # Play sequence with consistent timing
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
