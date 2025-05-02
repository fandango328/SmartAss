import pygame
import json
import asyncio
import random
import config
import cairosvg
import io
import numpy as np
import time
from pathlib import Path

class DisplayManager:
    """
    Manages the visual display system for LAURA with persona-specific and fallback resource handling.

    Folder structure (example for persona "laura"):
    /pygame/{persona}/
    ├── tool_use/                 # Persona-specific tool use images (preferred, new standard)
    ├── system/                   # System-level states and transitions (legacy/fallback)
    │   ├── tools_state/
    │   │   ├── enabled/
    │   │   └── disabled/
    │   ├── calibration/
    │   ├── document/
    │   │   ├── load/
    │   │   └── unload/
    │   └── persona/
    │       ├── in/
    │       └── out/
    ├── speaking/
    │   ├── casual/
    │   ├── excited/
    │   └── .../
    └── {other_states}/

    Fallback order for images:
    1. Persona's own directories (e.g., /pygame/{persona}/tool_use/)
    2. Laura's fallback directories (e.g., /pygame/laura/tool_use/)
    3. Ultimate fallback to thinking state (e.g., /pygame/laura/thinking/)

    Use the 'tool_use' folder for immediate tool-use feedback (stop_reason = tool_use), bridging to TTS playback.
    System/persona transitions (in/out), calibration, and document states are handled in their respective folders.
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
        self.moods = list(config.MOOD_COLORS.keys())
        
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
            'tool_use': str(self.base_path / 'tool_use'),
            
            # System states - Under system/ directory
            'tools_state': str(self.base_path / 'system' / 'tools_state'),
            'calibration': str(self.base_path / 'system' / 'calibration'),
            'document': str(self.base_path / 'system' / 'document'),
            'persona': str(self.base_path / 'system' / 'persona'),
            'system': str(self.base_path / 'system')
        }
        
        # Define valid subtypes for system states
        # This prevents invalid state/subtype combinations
        self.system_subtypes = {
            'tools_state': ['enabled', 'disabled'],
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
        self.screen = pygame.Surface((512, 512))
        
    def get_surface_image(self):
        """
        Return the current display surface as a NumPy array suitable for embedding in a web UI.
        This enables seamless integration with Gradio's image component.
        """
        import numpy as np
        arr = pygame.surfarray.array3d(self.screen)
        arr = np.transpose(arr, (1, 0, 2))  # Convert from (width, height, channels) to (height, width, channels)
        return arr        
    
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
        for state, directory in self.states.items():
            if state == 'speaking':
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
                    'tool_use': str(self.base_path / 'tool_use'),
                    
                    # System states
                    'tools_state': str(self.base_path / 'system' / 'tools_state'),
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
                    self.last_image_change = time.time()
                    self.state_entry_time = time.time()
                    return True
                else:
                    print("Warning: Failed to load sleep images for new persona")
                    return False
            
            except Exception as e:
                print(f"Error updating display path: {e}")
                return False
             
    async def update_display(self, state, mood=None, transition_path=None, tool_name=None):
        """
        Update display state with support for persona/fallback tool_use state.
        For 'tool_use', loads from persona/tool_use/, falls back to laura/tool_use/ if needed.
        All other logic remains as before.
        """
        async with self.state_lock:
            if mood is None:
                mood = self.current_mood
            mood = config.map_mood(mood)

            # Skip redundant updates unless transitioning
            if (state == self.current_state and 
                mood == self.current_mood and 
                transition_path is None and 
                tool_name is None):
                return

            try:
                self.last_state = self.current_state
                self.current_state = state
                self.current_mood = mood

                if state == "tool_use":
                    persona_path = self.base_path / 'tool_use'
                    laura_path = Path('/home/user/LAURA/pygame/laura/tool_use')
                    # Prefer tool-specific image if available
                    if tool_name:
                        for base in [persona_path, laura_path]:
                            tool_img = base / f"{tool_name}.png"
                            if tool_img.exists():
                                display_img = pygame.transform.scale(
                                    pygame.image.load(str(tool_img)), (512, 512)
                                )
                                self.current_image = display_img
                                self.screen.blit(self.current_image, (0, 0))
                                self.state_entry_time = time.time()
                                return
                    # Otherwise, randomly pick any PNG from persona or Laura fallback dir
                    pngs = []
                    for base in [persona_path, laura_path]:
                        if base.exists():
                            pngs.extend(list(base.glob("*.png")))
                    if pngs:
                        display_img = pygame.transform.scale(
                            pygame.image.load(str(random.choice(pngs))), (512, 512)
                        )
                        self.current_image = display_img
                        self.screen.blit(self.current_image, (0, 0))
                        self.state_entry_time = time.time()
                        return
                    # If nothing is found, fall through to normal state handling

                # Handle system states with updated tool state logic
                if state in ['tools_state', 'calibration', 'document']:
                    try:
                        if state == 'tools_state':
                            state_type = 'tools_state'
                            subtype = tool_name if tool_name in self.system_subtypes['tools_state'] else None
                        elif state == 'calibration':
                            state_type = 'calibration'
                            subtype = None
                        else:  # document
                            state_type = 'document'
                            subtype = tool_name if tool_name in self.system_subtypes['document'] else None
                        image_path = self._get_system_image_path(state_type, subtype)
                        if image_path.is_file():
                            system_image = pygame.transform.scale(
                                pygame.image.load(str(image_path)),
                                (512, 512)
                            )
                        else:
                            png_files = list(image_path.glob('*.png'))
                            if not png_files:
                                raise FileNotFoundError(f"No PNG files found in {image_path}")
                            system_image = pygame.transform.scale(
                                pygame.image.load(str(png_files[0])),
                                (512, 512)
                            )
                        self.current_image = system_image
                        self.screen.blit(self.current_image, (0, 0))
                        self.state_entry_time = time.time()
                        return
                    except Exception as e:
                        print(f"Error handling system state {state}: {e}")

                # Handle persona transitions with prior logic (unchanged)
                if transition_path is not None:
                    try:
                        transition_path = Path(transition_path)
                        if transition_path.exists():
                            if tool_name:
                                specific_path = Path(tool_name)
                                if specific_path.exists() and specific_path.is_file():
                                    transition_image = pygame.transform.scale(
                                        pygame.image.load(str(tool_name)),
                                        (512, 512)
                                    )
                                    self.current_image = transition_image
                                    self.screen.blit(self.current_image, (0, 0))
                                    self.state_entry_time = time.time()
                                    return
                            png_files = list(transition_path.glob('*.png'))
                            if png_files:
                                transition_image = pygame.transform.scale(
                                    pygame.image.load(str(png_files[0])),
                                    (512, 512)
                                )
                                self.current_image = transition_image
                                self.screen.blit(self.current_image, (0, 0))
                                self.state_entry_time = time.time()
                                return
                    except Exception as e:
                        print(f"Error handling transition: {e}")

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
                self.state_entry_time = time.time()
                if state in ['idle', 'sleep']:
                    self.last_image_change = self.state_entry_time
            except Exception as e:
                print(f"Error updating display: {e}")
                import traceback; traceback.print_exc()
                try:
                    if 'thinking' in self.image_cache:
                        self.current_image = random.choice(self.image_cache['thinking'])
                        self.screen.blit(self.current_image, (0, 0))
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
                    await asyncio.sleep(frame_delay)
                    
                return True
                    
            except Exception as e:
                print(f"Error playing transition sequence: {e}")
                return False

class AuraVisualizer:
    def __init__(
        self,
        svg_path,
        window_size=600,
        fps=30,
        mood="casual",
        base_radius_active=None,
        radius_variation_active=None,
        base_radius_idle=None,
        radius_variation_idle=None,
        idle_energy=0.3,
        aura_center_y_factor=0.45
    ):
        self.window_size = window_size
        self.fps = fps
        self.mood = mood
        self.gradient_colors = config.get_mood_color(mood)
        self.base_radius_active = base_radius_active or int(window_size * 0.5)
        self.radius_variation_active = radius_variation_active or int(window_size * 0.35)
        self.base_radius_idle = base_radius_idle or int(window_size * 0.4)
        self.radius_variation_idle = radius_variation_idle or int(window_size * 0.35)
        self.idle_energy = idle_energy
        self.aura_center_y_factor = aura_center_y_factor

        pygame.init()
        self.surface = pygame.Surface((window_size, window_size), pygame.SRCALPHA)

        self.svg_height = window_size
        self.svg_width = int(self.svg_height * 0.8)
        self.silhouette = self.svg_to_surface(svg_path, self.svg_width, self.svg_height)
        self.bounding_rect = self.silhouette.get_bounding_rect()
        self.y_offset = window_size - self.bounding_rect.bottom
        self.silhouette_rect = self.silhouette.get_rect()
        self.silhouette_rect.left = (window_size - self.svg_width) // 2
        self.silhouette_rect.top = self.y_offset

        self.current_energy = 0.0
        self.target_energy = 0.0
        self.audio_playing = False
        self.idle_start_time = None
        self.last_update_time = time.time()
        self.idle_frame = 0

    def svg_to_surface(self, svg_path, width, height):
        png_bytes = cairosvg.svg2png(url=str(svg_path), output_width=width, output_height=height)
        image = pygame.image.load(io.BytesIO(png_bytes)).convert_alpha()
        return image

    def set_mood(self, mood):
        """Update the current mood and corresponding gradient colors."""
        self.mood = mood
        self.gradient_colors = config.get_mood_color(mood)

    def set_energy(self, energy, active=True):
        self.target_energy = np.clip(energy, 0.0, 1.0)
        self.audio_playing = active
        if not active and self.idle_start_time is None:
            self.idle_start_time = time.time()

    def update(self):
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now

        if self.audio_playing:
            base_radius = self.base_radius_active
            radius_variation = self.radius_variation_active
            self.current_energy += (self.target_energy - self.current_energy) * 0.35
            self.idle_start_time = None
        else:
            if self.idle_start_time is None:
                self.idle_start_time = now
            t = now - self.idle_start_time
            base_radius = self.base_radius_idle
            radius_variation = self.radius_variation_idle
            pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)
            self.target_energy = self.idle_energy + (pulse - 0.5) * 0.05
            self.target_energy = np.clip(self.target_energy, self.idle_energy - 0.025, self.idle_energy + 0.025)
            self.current_energy += (self.target_energy - self.current_energy) * 0.25
            self.idle_frame += 1

        self.surface.fill((0, 0, 0, 0))
        aura_cx = self.silhouette_rect.centerx
        aura_cy = self.silhouette_rect.top + int(self.svg_height * self.aura_center_y_factor)
        # Use self.gradient_colors, which are now mood-dependent
        self.draw_audio_gradient(
            self.surface, (aura_cx, aura_cy), base_radius, radius_variation, self.current_energy, self.gradient_colors
        )
        self.surface.blit(self.silhouette, self.silhouette_rect)

    def draw_audio_gradient(self, surface, center, base_radius, variation, energy, colors):
        w, h = surface.get_size()
        cx, cy = center
        max_radius = base_radius + energy * variation
        pink_portion = 0.45
        for r in range(int(max_radius), 0, -2):
            t_linear = r / max_radius
            if t_linear < pink_portion:
                t = 0
            else:
                t = (t_linear - pink_portion) / (1 - pink_portion)
            color = [
                int(colors[0][i] * (1 - t) + colors[1][i] * t)
                for i in range(3)
            ]
            alpha = int(220 * (1 - t_linear) ** 2)
            s = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), center, r)
            surface.blit(s, (0, 0))

    def get_surface_image(self):
        arr = pygame.surfarray.array3d(self.surface)
        arr = np.transpose(arr, (1, 0, 2))
        return arr

    def reset(self):
        self.idle_start_time = None
        self.idle_frame = 0
