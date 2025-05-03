import pygame
import asyncio
import threading
import numpy as np
import time
from pathlib import Path
import config
import pyaudio
import queue

class DisplayManager:
    """
    Manages the visual display system for LAURA using AuraVisualizer.
    Implements the full feathered, pulsing, audio-reactive aura effect as in the test script.
    """
    def __init__(self, svg_path="/home/user/LAURA/svg files/silhouette.svg", boot_img_path="/home/user/LAURA/pygame/laura/speaking/interested/interested01.png", window_size=512):
        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("LAURA Aura Visualizer")

        self.boot_img_path = boot_img_path
        self.boot_img = None
        self._booting_lock = threading.Lock()
        self.booting = True
        try:
            boot_img = pygame.image.load(self.boot_img_path).convert_alpha()
            boot_img = pygame.transform.scale(boot_img, (window_size, window_size))
            self.boot_img = boot_img
            self.screen.blit(boot_img, (0, 0))
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

        # Instantiate AuraVisualizer
        self.aura = AuraVisualizer(svg_path=svg_path, window_size=window_size)
        self.aura.set_mood('casual')
        self.aura.reset()

        # Setup audio input
        self.audio_stream = AudioInputStream()
        self.audio_stream.start()
        self.audio_energy = 0.0
        self.audio_active = True
        self.audio_idle_time = None

        self._running = True
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        self._current_energy = 0.0
        self._target_energy = 0.0
        self._audio_playing = True
        self._idle_start_time = None
        self._idle_frame = 0

    def _display_loop(self):
        clock = pygame.time.Clock()
        last_booting = True
        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
            with self._booting_lock:
                booting_now = self.booting
            if booting_now and self.boot_img is not None:
                self.screen.blit(self.boot_img, (0, 0))
                pygame.display.flip()
                last_booting = True
            else:
                if last_booting:
                    self.screen.fill((0, 0, 0))
                    last_booting = False
                # Audio-driven or idle pulsing aura
                energy, audio_playing = self.get_current_energy()
                self.aura.update(energy, audio_playing)
                surface_img = self.aura.get_surface_image()
                pygame.surfarray.blit_array(self.screen, np.transpose(surface_img, (1, 0, 2)))
                pygame.display.flip()
            clock.tick(self.aura.fps)

    def get_current_energy(self):
        """
        Computes and returns the current aura energy and audio playing state.
        """
        energy = self.audio_stream.get_latest_energy()
        if energy > 0.01:
            self._audio_playing = True
            self._idle_start_time = None
            self._target_energy = energy
            base_radius = self.aura.base_radius_active
            radius_variation = self.aura.radius_variation_active
        else:
            # Idle/pulse when no audio
            if self._audio_playing:
                self._idle_start_time = time.time()
                self._audio_playing = False
            t = (time.time() - (self._idle_start_time or time.time())) if self._idle_start_time else 0
            base_radius = self.aura.base_radius_idle
            radius_variation = self.aura.radius_variation_idle
            pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)
            self._target_energy = self.aura.idle_energy + (pulse - 0.5) * 0.05
            self._target_energy = np.clip(self._target_energy, self.aura.idle_energy - 0.025, self.aura.idle_energy + 0.025)
            self._idle_frame += 1
        # Smooth interpolation
        self._current_energy += (self._target_energy - self._current_energy) * 0.35
        return self._current_energy, self._audio_playing

    async def start_async_tasks(self):
        pass  # No longer needed, audio handled in display thread

    def finish_boot(self):
        with self._booting_lock:
            self.booting = False
        # Force a redraw immediately after boot
        self.aura.update(self._current_energy, self._audio_playing)
        surface_img = self.aura.get_surface_image()
        pygame.surfarray.blit_array(self.screen, np.transpose(surface_img, (1, 0, 2)))
        pygame.display.flip()

    def get_aura_image(self):
        if getattr(self, "booting", False) and self.boot_img is not None:
            arr = pygame.surfarray.array3d(self.boot_img)
            arr = np.transpose(arr, (1, 0, 2))
            return arr
        else:
            self.aura.update(self._current_energy, self._audio_playing)
            return self.aura.get_surface_image()

    async def update_display_path(self, new_base_path: str):
        async with self.state_lock:
            try:
                print(f"Updating display path to: {new_base_path}")
                persona_dir = Path(new_base_path)
                svg_path = persona_dir.parent / "svg files" / "silhouette.svg"
                self.aura = AuraVisualizer(svg_path=str(svg_path), window_size=self.window_size)
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

    def cleanup(self):
        self._running = False
        self.audio_stream.stop()
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
        if not pygame.display.get_init() or not pygame.display.get_surface():
            pygame.display.set_mode((1, 1), pygame.HIDDEN)
        self.surface = pygame.Surface((window_size, window_size), pygame.SRCALPHA)

        self.svg_height = window_size
        self.svg_width = int(self.svg_height * 0.8)
        self.silhouette = self.svg_to_surface(svg_path, self.svg_width, self.svg_height)
        self.bounding_rect = self.silhouette.get_bounding_rect()
        self.y_offset = window_size - self.bounding_rect.bottom
        self.silhouette_rect = self.silhouette.get_rect()
        self.silhouette_rect.left = (window_size - self.svg_width) // 2
        self.silhouette_rect.top = self.y_offset

    def svg_to_surface(self, svg_path, width, height):
        import cairosvg
        import io
        png_bytes = cairosvg.svg2png(url=str(svg_path), output_width=width, output_height=height)
        image = pygame.image.load(io.BytesIO(png_bytes)).convert_alpha()
        return image

    def set_mood(self, mood):
        self.mood = mood
        self.gradient_colors = config.get_mood_color(mood)

    def update(self, energy, audio_playing):
        now = time.time()
        self.surface.fill((0, 0, 0, 0))
        # Aura center lower in window, as in test script
        aura_cx = self.silhouette_rect.centerx
        aura_cy = self.silhouette_rect.top + int(self.svg_height * self.aura_center_y_factor)
        if audio_playing:
            base_radius = self.base_radius_active
            radius_variation = self.radius_variation_active
        else:
            base_radius = self.base_radius_idle
            radius_variation = self.radius_variation_idle
        self.draw_audio_gradient(
            self.surface, (aura_cx, aura_cy), base_radius, radius_variation, energy, self.gradient_colors
        )
        self.surface.blit(self.silhouette, self.silhouette_rect)

    def draw_audio_gradient(self, surface, center, base_radius, variation, energy, colors):
        w, h = surface.get_size()
        cx, cy = center
        max_radius = base_radius + energy * variation
        pink_portion = 0.45  # 45% pink before blue transition
        for r in range(int(max_radius), 0, -2):
            t_linear = r / max_radius
            if t_linear < pink_portion:
                t = 0  # Full pink (or center color)
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
        pass 
class AudioInputStream:
    """
    Captures audio input from the default device and computes normalized RMS amplitude in real time.
    """
    def __init__(self, sample_rate=16000, chunk_size=600):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_interface = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue()
        self.running = False
        self.max_recent = 100
        self.energy_history = []

    def start(self):
        if self.stream is not None:
            return
        self.running = True
        self.stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.callback,
        )
        self.stream.start_stream()

    def stop(self):
        self.running = False
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.audio_interface.terminate()

    def callback(self, in_data, frame_count, time_info, status):
        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2))
        self.energy_history.append(rms)
        if len(self.energy_history) > self.max_recent:
            self.energy_history.pop(0)
        # Normalize based on recent history for smooth response
        min_rms = min(self.energy_history) if self.energy_history else 0
        max_rms = max(self.energy_history) if self.energy_history else 1
        norm = (rms - min_rms) / (max_rms - min_rms + 1e-6)
        self.audio_queue.put(norm)
        return (None, pyaudio.paContinue)

    def get_latest_energy(self):
        # Return the latest energy value from the audio queue, or 0 if none available
        val = 0.0
        while not self.audio_queue.empty():
            val = self.audio_queue.get_nowait()
        return val
