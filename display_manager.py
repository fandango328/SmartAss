import pygame
import asyncio
import threading
import numpy as np
import time
from pathlib import Path
import config
import pyaudio
import queue
import cairosvg
import io

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

        # Initialize AuraVisualizer
        self.aura = AuraVisualizer(svg_path=svg_path, window_size=window_size)
        self.aura.set_mood('casual')
        self.aura.reset()

        # Setup audio input stream
        self.audio_stream = AudioInputStream()
        self.audio_stream.start()
        self.audio_energy = 0.0
        self.audio_active = True
        self.audio_idle_time = None

        self._running = True
        self._last_boot_time = time.time()
        # Call finish_boot after 2 seconds in tick() if needed

    def tick(self):
        # Should be called every frame from main thread
        now = time.time()
        # Handle boot transition
        with self._booting_lock:
            if self.booting and now - self._last_boot_time > 2.0:
                self.booting = False
                print("Boot phase complete, aura visualizer now active.")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False

        if self.booting and self.boot_img is not None:
            self.screen.blit(self.boot_img, (0, 0))
            pygame.display.flip()
        else:
            self.screen.fill((0, 0, 0))  # Clear background
            energy, audio_playing = self.get_current_energy()
            base_radius = self.aura.base_radius_active if audio_playing else self.aura.base_radius_idle
            radius_variation = self.aura.radius_variation_active if audio_playing else self.aura.radius_variation_idle
            aura_cx, aura_cy = self.aura.aura_center
            draw_audio_gradient(
                self.screen,
                (aura_cx, aura_cy),
                base_radius,
                radius_variation,
                energy,
                self.aura.gradient_colors
            )
            self.screen.blit(self.aura.silhouette, self.aura.silhouette_rect)
            pygame.display.flip()
    def get_current_energy(self):
        """
        Computes and returns the current aura energy and audio playing state.
        Implements blending and pulsing logic from the test script.
        """
        energy, audio_is_active = self.audio_stream.get_latest_energy_blended()
        if audio_is_active:
            self._audio_playing = True
            self._idle_start_time = None
            self._target_energy = energy
        else:
            # Idle/pulse when no audio
            if self._audio_playing:
                self._idle_start_time = time.time()
                self._audio_playing = False
            t = (time.time() - (self._idle_start_time or time.time())) if self._idle_start_time else 0
            pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)  # 0..1
            self._target_energy = self.aura.idle_energy + (pulse - 0.5) * 0.05  # Only 5% pulse
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
        print("Boot phase complete, aura visualizer now active.")

    def get_aura_image(self):
        # Not used as rendering is direct to screen
        pass

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
    ):
        self.window_size = window_size
        self.fps = fps
        self.mood = mood
        # Match test script gradient exactly: Pink center, light blue outside
        self.gradient_colors = [(255, 105, 180), (0, 206, 209)]
        self.base_radius_active = base_radius_active or int(window_size * 0.5)
        self.radius_variation_active = radius_variation_active or int(window_size * 0.35)
        self.base_radius_idle = base_radius_idle or int(window_size * 0.4)
        self.radius_variation_idle = radius_variation_idle or int(window_size * 0.35)
        self.idle_energy = idle_energy

        pygame.init()
        if not pygame.display.get_init() or not pygame.display.get_surface():
            pygame.display.set_mode((1, 1), pygame.HIDDEN)

        self.svg_height = window_size
        self.svg_width = int(self.svg_height * 0.8)
        self.silhouette = self.svg_to_surface(svg_path, self.svg_width, self.svg_height)
        self.bounding_rect = self.silhouette.get_bounding_rect()
        self.y_offset = window_size - self.bounding_rect.bottom
        self.silhouette_rect = self.silhouette.get_rect()
        self.silhouette_rect.left = (window_size - self.svg_width) // 2
        self.silhouette_rect.top = self.y_offset

    def svg_to_surface(self, svg_path, width, height):
        png_bytes = cairosvg.svg2png(url=str(svg_path), output_width=width, output_height=height)
        image = pygame.image.load(io.BytesIO(png_bytes)).convert_alpha()
        return image

    def set_mood(self, mood):
        self.mood = mood
        # Keep test script's gradient for now (ignore mood)
        self.gradient_colors = [(255, 105, 180), (0, 206, 209)]

    @property
    def aura_center(self):
        # Drop aura center lower, as in test script
        return (
            self.silhouette_rect.centerx,
            self.silhouette_rect.top + int(self.svg_height * 0.45)
        )

    def reset(self):
        pass

class AudioInputStream:
    """
    Captures audio input from the default device and computes normalized RMS and peak amplitude in real time.
    Blends both for lively, test-script-like visuals.
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
        self.peak_history = []

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
        peak = np.max(np.abs(samples))
        self.energy_history.append(rms)
        self.peak_history.append(peak)
        if len(self.energy_history) > self.max_recent:
            self.energy_history.pop(0)
        if len(self.peak_history) > self.max_recent:
            self.peak_history.pop(0)
        min_rms = min(self.energy_history) if self.energy_history else 0
        max_rms = max(self.energy_history) if self.energy_history else 1
        min_peak = min(self.peak_history) if self.peak_history else 0
        max_peak = max(self.peak_history) if self.peak_history else 1
        # Silence threshold logic
        silence_thresh = 2
        if rms < silence_thresh and peak < silence_thresh:
            norm_rms = 0.0
            norm_peak = 0.0
        else:
            norm_rms = (rms - min_rms) / (max_rms - min_rms + 1e-6) if max_rms > min_rms else 0.0
            norm_peak = (peak - min_peak) / (max_peak - min_peak + 1e-6) if max_peak > min_peak else 0.0
        # Blend peak and rms as in test script
        blended = 0.65 * norm_peak + 0.35 * norm_rms
        self.audio_queue.put((blended, not (norm_rms == 0.0 and norm_peak == 0.0)))
        return (None, pyaudio.paContinue)

    def get_latest_energy_blended(self):
        # Return the latest (energy, is_active) tuple from the audio queue, or (0, False) if none available
        val = (0.0, False)
        while not self.audio_queue.empty():
            val = self.audio_queue.get_nowait()
        return val

def draw_audio_gradient(surface, center, base_radius, variation, energy, colors):
    """Draws a circular gradient with radius modulated by audio energy, with more pink visible."""
    w, h = surface.get_size()
    cx, cy = center
    max_radius = base_radius + energy * variation
    pink_portion = 0.45  # 45% pink before blue transition
    for r in range(int(max_radius), 0, -2):
        t_linear = r / max_radius
        # Compress blue transition into outer 30% band
        if t_linear < pink_portion:
            t = 0  # Full pink
        else:
            # Map [pink_portion, 1] -> [0, 1] for transition
            t = (t_linear - pink_portion) / (1 - pink_portion)
        color = [
            int(colors[0][i] * (1 - t) + colors[1][i] * t)
            for i in range(3)
        ]
        alpha = int(220 * (1 - t_linear) ** 2)
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, alpha), center, r)
        surface.blit(s, (0, 0))
