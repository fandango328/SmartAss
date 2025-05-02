import pygame
import asyncio
import config
import time
from pathlib import Path

class DisplayManager:
    """
    Manages the visual display system for LAURA using AuraVisualizer for all visuals.
    Static PNG logic is fully retired. Shows a boot image immediately, then hands off to AuraVisualizer.
    """

    def __init__(self):
        pygame.init()
        self.screen = pygame.Surface((600, 600))

        # Display a bootup image immediately
        self.boot_img_path = "/home/user/LAURA/pygame/laura/speaking/interested/interested01.png"
        self.boot_img = None
        self.booting = True
        try:
            boot_img = pygame.image.load(self.boot_img_path).convert_alpha()
            boot_img = pygame.transform.scale(boot_img, self.screen.get_size())
            self.screen.blit(boot_img, (0, 0))
            self.boot_img = boot_img
        except Exception as e:
            print(f"WARNING: Could not load bootup image {self.boot_img_path}: {e}")        

        self.current_state = 'sleep'
        self.current_mood = 'casual'
        self.last_state = None
        self.last_mood = None
        self.state_entry_time = time.time()
        self.initialized = True
        self.state_lock = asyncio.Lock()

        # Instantiate AuraVisualizer with default persona SVG
        svg_path = "/home/user/LAURA/svg files/silhouette.svg"  # Adjust if persona-specific
        self.aura = AuraVisualizer(svg_path=svg_path, window_size=600)
        self.aura.set_mood('casual')
        self.aura.reset()

    def finish_boot(self):
        """
        Call this once all initialization is complete and you want to show the aura visual.
        """
        self.booting = False

    def get_aura_image(self):
        """
        Return the current display as a numpy array.
        Shows boot image until booting is finished, then shows aura.
        """
        import numpy as np
        if getattr(self, "booting", False) and self.boot_img is not None:
            arr = pygame.surfarray.array3d(self.boot_img)
            arr = np.transpose(arr, (1, 0, 2))
            return arr
        else:
            self.aura.update()
            return self.aura.get_surface_image()

    async def update_display_path(self, new_base_path: str):
        """
        Update the AuraVisualizer SVG for persona changes.
        """
        async with self.state_lock:
            try:
                print(f"Updating display path to: {new_base_path}")
                # Persona folder expected to be /home/user/LAURA/pygame/{persona}
                persona_dir = Path(new_base_path)
                svg_path = persona_dir.parent / "svg files" / "silhouette.svg"
                self.aura = AuraVisualizer(svg_path=str(svg_path), window_size=600)
                self.aura.set_mood('casual')
                self.aura.reset()
                self.current_state = 'sleep'
                self.current_mood = 'casual'
                self.state_entry_time = time.time()
                return True
            except Exception as e:
                print(f"Error updating display path: {e}")
                return False

    async def update_display(self, state, mood=None, **kwargs):
        """
        Update display state and mood for AuraVisualizer-based visuals.
        All main states map to 'casual' (light blue & pink) unless overridden.
        """
        async with self.state_lock:
            # Map all standard states to the 'casual' mood (light blue & pink)
            STATE_TO_MOOD = {
                'sleep': 'casual',
                'wake': 'casual',
                'idle': 'casual',
                'thinking': 'casual',
                'tool_use': 'casual',
                'calibration': 'casual',
                'document': 'casual',
                'persona': 'casual',
                'system': 'casual',
                # You can add more mappings or make this dynamic later
            }
            # Allow explicit mood override, otherwise use mapped mood
            if mood is None:
                mood = STATE_TO_MOOD.get(state, 'casual')
            mood = config.map_mood(mood)
            self.aura.set_mood(mood)
            self.last_state = self.current_state
            self.current_state = state
            self.current_mood = mood
            self.state_entry_time = time.time()
            # Optionally, set energy or animation state here for future expansion

    def cleanup(self):
        pygame.quit()

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
        import cairosvg
        import io
        png_bytes = cairosvg.svg2png(url=str(svg_path), output_width=width, output_height=height)
        image = pygame.image.load(io.BytesIO(png_bytes)).convert_alpha()
        return image

    def set_mood(self, mood):
        self.mood = mood
        self.gradient_colors = config.get_mood_color(mood)

    def set_energy(self, energy, active=True):
        import numpy as np
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
            import numpy as np
            pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)
            self.target_energy = self.idle_energy + (pulse - 0.5) * 0.05
            self.target_energy = np.clip(self.target_energy, self.idle_energy - 0.025, self.idle_energy + 0.025)
            self.current_energy += (self.target_energy - self.current_energy) * 0.25
            self.idle_frame += 1

        self.surface.fill((0, 0, 0, 0))
        aura_cx = self.silhouette_rect.centerx
        aura_cy = self.silhouette_rect.top + int(self.svg_height * self.aura_center_y_factor)
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
        import numpy as np
        arr = pygame.surfarray.array3d(self.surface)
        arr = np.transpose(arr, (1, 0, 2))
        return arr

    def reset(self):
        self.idle_start_time = None
        self.idle_frame = 0
