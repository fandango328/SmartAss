import os
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

    def __init__(self, svg_path, boot_img_path, window_size):
        pygame.init()
        self.display_surface = pygame.display.set_mode((window_size, window_size))
        self.screen = pygame.Surface((window_size, window_size))

        self.boot_img_path = boot_img_path
        self.boot_img = None
        self.booting = True
        try:
            boot_img = pygame.image.load(self.boot_img_path).convert_alpha()
            boot_img = pygame.transform.scale(boot_img, self.screen.get_size())
            self.screen.blit(boot_img, (0, 0))
            self.boot_img = boot_img

            # Blit boot image to display and update window
            self.display_surface.blit(self.boot_img, (0, 0))
            pygame.display.flip()
        except Exception as e:
            print(f"WARNING: Could not load bootup image {self.boot_img_path}: {e}")        

        self.current_state = 'sleep'
        self.current_mood = 'casual'
        self.last_state = None
        self.last_mood = None
        self.state_entry_time = time.time()
        self.initialized = True
        self.state_lock = asyncio.Lock()

        self.aura = AuraVisualizer(svg_path=svg_path, window_size=window_size)
        self.aura.set_mood('casual')
        self.aura.reset()
        self.audio_active = False

    def finish_boot(self):
        self.booting = False

    def get_aura_image(self):
        import numpy as np
        if getattr(self, "booting", False) and self.boot_img is not None:
            arr = pygame.surfarray.array3d(self.boot_img)
            arr = np.transpose(arr, (1, 0, 2))
            self.display_surface.blit(self.boot_img, (0, 0))
            pygame.display.flip()
            return arr
        else:
            self.aura.update()
            self.display_surface.fill((0, 0, 0))
            self.display_surface.blit(self.aura.surface, (0, 0))
            pygame.display.flip()
            return self.aura.get_surface_image()

    async def update_display_path(self, new_base_path: str):
        async with self.state_lock:
            try:
                print(f"Updating display path to: {new_base_path}")
                persona_dir = Path(new_base_path)
                svg_path = persona_dir.parent / "svg files" / "silhouette.svg"
                self.aura = AuraVisualizer(svg_path=str(svg_path), window_size=self.screen.get_width())
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
        async with self.state_lock:
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
            }
            if mood is None:
                mood = STATE_TO_MOOD.get(state, 'casual')
            mood = config.map_mood(mood)
            self.aura.set_mood(mood)
            self.last_state = self.current_state
            self.current_state = state
            self.current_mood = mood
            self.state_entry_time = time.time()
            # Clear background before rendering aura
            self.aura.update()
            self.display_surface.fill((0, 0, 0))
            self.display_surface.blit(self.aura.surface, (0, 0))
            pygame.display.flip()

    def set_audio_energy(self, energy, active=True):
        """Call this from your audio playback loop to update the aura in real time with audio energy."""
        self.audio_active = active
        self.aura.set_energy(energy, active=active)

    def cleanup(self):
        pygame.quit()

class AuraVisualizer:
    def __init__(
        self,
        svg_path,
        window_size=512,
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
