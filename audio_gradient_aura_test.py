import pygame
import cairosvg
import io
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
import time

# ---- CONFIG ----
SVG_PATH = "/home/user/LAURA/svg files/silhouette.svg"
AUDIO_PATH = "/home/user/Downloads/audiofileforgradientcolor.mp3"
WINDOW_SIZE = 600  # Set square canvas: 512x512 or change to 768 for larger
FPS = 30
GRADIENT_COLORS = [(255, 105, 180), (0, 206, 209)]  # Pink center, light blue outside

# Main animation sizes for aura
BASE_RADIUS_ACTIVE = int(WINDOW_SIZE * 0.5)
RADIUS_VARIATION_ACTIVE = int(WINDOW_SIZE * 0.35)

# Idle (post-audio) sizes for aura
BASE_RADIUS_IDLE = int(WINDOW_SIZE * 0.4)
RADIUS_VARIATION_IDLE = int(WINDOW_SIZE * 0.35)  # Only 1% pulsing allowed

IDLE_ENERGY = 0.3  # Target energy after audio is done

def svg_to_surface(svg_path, width, height):
    """Rasterize SVG to PNG and load as pygame.Surface, preserving alpha."""
    png_bytes = cairosvg.svg2png(url=svg_path, output_width=width, output_height=height)
    image = pygame.image.load(io.BytesIO(png_bytes)).convert_alpha()
    return image

def get_audio_chunks(audio_path, chunk_ms=10, silence_thresh=2):
    """
    Yield chunks of audio and a blended peak/RMS amplitude for visualization,
    treating near-silence as zero. Also print raw RMS, peak, and normalized values for debugging.
    """
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_sample_width(2)
    chunks = make_chunks(audio, chunk_ms)
    peak_values = []
    rms_values = []
    for chunk in chunks:
        samples = np.array(chunk.get_array_of_samples())
        rms = np.sqrt(np.mean(samples.astype(float)**2))
        peak = np.max(np.abs(samples))
        rms_values.append(rms)
        peak_values.append(peak)
    min_rms, max_rms = min(rms_values), max(rms_values)
    min_peak, max_peak = min(peak_values), max(peak_values)
    norm_rms = []
    norm_peak = []
    for idx, (r, p) in enumerate(zip(rms_values, peak_values)):
        # Silence threshold logic:
        if r < silence_thresh and p < silence_thresh:
            norm_rms.append(0.0)
            norm_peak.append(0.0)
        else:
            norm_rms.append(
                (r - min_rms) / (max_rms - min_rms + 1e-6) if max_rms > min_rms else 0.0
            )
            norm_peak.append(
                (p - min_peak) / (max_peak - min_peak + 1e-6) if max_peak > min_peak else 0.0
            )
    blended = [0.65 * p + 0.35 * r for p, r in zip(norm_peak, norm_rms)]
    chunks_per_frame = int((1000 / FPS) // chunk_ms)
    if chunks_per_frame < 1:
        chunks_per_frame = 1
    output = []
    for i in range(0, len(chunks), chunks_per_frame):
        frame_rms = rms_values[i:i+chunks_per_frame]
        frame_peak = peak_values[i:i+chunks_per_frame]
        frame_norm = blended[i:i+chunks_per_frame]
        val = max(frame_norm) if frame_norm else 0.0
        print(
            f"Frame {i//chunks_per_frame}: "
            f"RMS={frame_rms}, Peak={frame_peak}, NormEnergy={frame_norm}"
        )
        output.append(val)
    for val in output:
        yield None, val

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

def main():
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Audio Reactive Gradient Aura Test")
    clock = pygame.time.Clock()

    # Prepare SVG silhouette, render at full window height for true bottom alignment
    svg_height = WINDOW_SIZE
    svg_width = int(svg_height * 0.8)
    silhouette = svg_to_surface(SVG_PATH, svg_width, svg_height)

    # Find the tight bounding box of the opaque silhouette
    bounding_rect = silhouette.get_bounding_rect()
    y_offset = WINDOW_SIZE - bounding_rect.bottom
    silhouette_rect = silhouette.get_rect()
    silhouette_rect.left = (WINDOW_SIZE - svg_width) // 2
    silhouette_rect.top = y_offset

    # Load audio and prepare chunks
    audio_chunks = list(get_audio_chunks(AUDIO_PATH))  # Now auto-downsamples to FPS
    start_time = time.time()

    # Energy diagnostic: print min, max, mean
    energies = [e for _, e in audio_chunks]
    print(f"Energy stats: min={min(energies):.3f}, max={max(energies):.3f}, mean={np.mean(energies):.3f}")

    pygame.mixer.music.load(AUDIO_PATH)
    pygame.mixer.music.play()

    current_energy = 0.0
    audio_playing = True
    idle_start_time = None
    idle_frame = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Calculate frame index based on elapsed time and FPS
        elapsed = time.time() - start_time
        audio_index = int(elapsed * FPS)

        if audio_index < len(audio_chunks):
            _, target_energy = audio_chunks[audio_index]
            print(f"Frame {audio_index}: Energy={target_energy:.3f}")
            audio_playing = True
            idle_start_time = None
            base_radius = BASE_RADIUS_ACTIVE
            radius_variation = RADIUS_VARIATION_ACTIVE
        else:
            # Audio is done; transition to tiny idle aura
            if audio_playing:
                idle_start_time = time.time()
                audio_playing = False
                print("Audio finished. Entering idle aura mode.")
            t = (time.time() - (idle_start_time or time.time())) if idle_start_time else 0
            base_radius = BASE_RADIUS_IDLE
            radius_variation = RADIUS_VARIATION_IDLE
            pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)  # 0..1
            target_energy = IDLE_ENERGY + (pulse - 0.5) * 0.05  # Only 5% pulse
            target_energy = np.clip(target_energy, IDLE_ENERGY - 0.025, IDLE_ENERGY + 0.025)
            if idle_frame % 3 == 0:
                print(f"Idle Aura - Frame {idle_frame}: Energy={target_energy:.3f}, BaseR={base_radius}, Var={radius_variation}")
            idle_frame += 1

        # Smooth interpolation for nice animation
        current_energy += (target_energy - current_energy) * 0.35

        screen.fill((0, 0, 0))

        # Drop aura center lower, for example to 0.45 of svg_height
        aura_cx = silhouette_rect.centerx
        aura_cy = silhouette_rect.top + int(svg_height * 0.45)
        draw_audio_gradient(
            screen,
            (aura_cx, aura_cy),
            base_radius,
            radius_variation,
            current_energy,
            GRADIENT_COLORS
        )

        screen.blit(silhouette, silhouette_rect)

        pygame.display.flip()
        clock.tick(FPS)

        if not audio_playing and not pygame.mixer.music.get_busy():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

    pygame.quit()

if __name__ == "__main__":
    main()
