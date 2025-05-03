import pygame
import time
import numpy as np
import cairosvg
import io
import sys

class AuraVisualizer:
    def __init__(self, svg_path, window_size=512, aura_colors=None):
        self.window_size = window_size
        self.surface = pygame.Surface((window_size, window_size), pygame.SRCALPHA)
        self.svg_surface = self.svg_to_surface(svg_path, window_size, window_size)
        self.aura_colors = aura_colors or [(120, 180, 255), (255, 120, 200)]
        self.start_time = time.time()

    def svg_to_surface(self, svg_path, width, height):
        try:
            png_bytes = cairosvg.svg2png(url=svg_path, output_width=width, output_height=height)
            image = pygame.image.load(io.BytesIO(png_bytes)).convert_alpha()
            return image
        except Exception as e:
            print(f"ERROR: Failed to load SVG: {e}")
            surf = pygame.Surface((width, height), pygame.SRCALPHA)
            font = pygame.font.SysFont("Arial", 36)
            txt = font.render("SVG Error", True, (255, 0, 0))
            surf.blit(txt, (width // 4, height // 2))
            return surf

    def update(self):
        t = (time.time() - self.start_time)
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * (0.5) * t)
        base_radius = int(self.window_size * (0.4 + 0.1 * pulse))
        variation = int(self.window_size * (0.25 + 0.05 * pulse))
        center = (self.window_size // 2, int(self.window_size * 0.5))
        self.draw_gradient(center, base_radius, variation)
        self.surface.blit(self.svg_surface, (0, 0))

    def draw_gradient(self, center, base_radius, variation):
        self.surface.fill((0, 0, 0, 0))
        max_radius = base_radius + variation
        for r in range(int(max_radius), 0, -2):
            t = r / max_radius
            color = [
                int(self.aura_colors[0][i] * (1 - t) + self.aura_colors[1][i] * t)
                for i in range(3)
            ]
            alpha = int(180 * (1 - t) ** 2)
            temp_surf = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*color, alpha), center, r)
            self.surface.blit(temp_surf, (0, 0))

class DisplayManager:
    def __init__(self, svg_path, boot_img_path, window_size=512):
        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("LAURAᕙ(  •̀ ᗜ •́  )ᕗ")
        self.boot_img = self.load_boot_image(boot_img_path)
        self.booting = True
        self.aura = AuraVisualizer(svg_path, window_size)
        self._running = True

    def load_boot_image(self, path):
        try:
            img = pygame.image.load(path).convert_alpha()
            img = pygame.transform.scale(img, (self.window_size, self.window_size))
            return img
        except Exception as e:
            print(f"WARNING: Could not load bootup image {path}: {e}")
            surf = pygame.Surface((self.window_size, self.window_size))
            surf.fill((32, 32, 32))
            font = pygame.font.SysFont("Arial", 36)
            txt = font.render("Booting...", True, (255, 255, 255))
            surf.blit(txt, (self.window_size // 4, self.window_size // 2))
            return surf

    def finish_boot(self):
        self.booting = False

    def update_display_loop(self):
        clock = pygame.time.Clock()
        boot_start = time.time()
        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False

            if self.booting:
                self.screen.blit(self.boot_img, (0, 0))
                pygame.display.update()
                if time.time() - boot_start > 2:
                    self.finish_boot()
            else:
                self.aura.update()
                self.screen.blit(self.aura.surface, (0, 0))
                pygame.display.update()
            clock.tick(30)

        pygame.quit()
        sys.exit()

    def stop(self):
        self._running = False

if __name__ == "__main__":
    # Update these paths to your actual files!
    SVG_PATH = "/home/user/LAURA/svg files/silhouette.svg"
    BOOT_IMG_PATH = "/home/user/LAURA/pygame/laura/speaking/interested/interested01.png"
    display_manager = DisplayManager(SVG_PATH, BOOT_IMG_PATH, window_size=512)
    display_manager.update_display_loop()
