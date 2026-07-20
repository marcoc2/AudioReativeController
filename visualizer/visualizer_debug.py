import pygame
import sys
import argparse
import time
import os
import subprocess
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import our custom modules
from core.feature_extractor import AudioFeatureExtractor
from visualizer.visualizer_pygame import DARK_BG, PINK, BLUE, WHITE

class DebugDashboard:
    """An educational visualizer that shows raw audio features and data points."""
    def __init__(self, screen):
        self.screen = screen
        self.width, self.height = screen.get_size()
        self.font_small = pygame.font.SysFont("monospace", 14)
        self.font_large = pygame.font.SysFont("monospace", 20, bold=True)
        self.history_len = 200
        self.bass_history = [0.0] * self.history_len
        self.pulse_history = [0.0] * self.history_len

    def draw_text(self, text, pos, color=WHITE, bold=False):
        font = self.font_large if bold else self.font_small
        surf = font.render(text, True, color)
        self.screen.blit(surf, pos)

    def draw_bar(self, label, value, rect_data, color):
        rect = pygame.Rect(rect_data)
        pygame.draw.rect(self.screen, (40, 40, 40), rect, border_radius=4)
        fill_w = int(rect.width * value)
        if fill_w > 0:
            pygame.draw.rect(self.screen, color, pygame.Rect(rect.left, rect.top, fill_w, rect.height), border_radius=4)
        self.draw_text(f"{label}: {value:.2f}", (rect.left, rect.top - 18))

    def render(self, features, playback_sec, dt):
        sidebar_x = self.width - 250
        pygame.draw.rect(self.screen, (20, 20, 20), (sidebar_x, 0, 250, self.height))
        self.draw_text("DEBUG CONSOLE", (sidebar_x + 10, 20), color=PINK, bold=True)
        self.draw_text(f"Time: {playback_sec:.2f}s", (sidebar_x + 10, 60))
        self.draw_text(f"FrameIdx: {features['frame_idx']}", (sidebar_x + 10, 80))
        self.draw_text(f"Pulse: {features['pulse']:.3f}", (sidebar_x + 10, 100))
        
        meter_w, start_y = 170, 150
        bands = features['bands']
        num_bands = len(bands)
        bar_h = min(20, (300 // num_bands) - 5)
        for i, val in enumerate(bands):
            rect = (sidebar_x + 10, start_y + i * (bar_h + 8), meter_w, bar_h)
            color_factor = i / max(1, num_bands - 1)
            color = (int(PINK[0] * (1 - color_factor) + BLUE[0] * color_factor), int(PINK[1] * (1 - color_factor) + BLUE[1] * color_factor), int(PINK[2] * (1 - color_factor) + BLUE[2] * color_factor))
            label = f"B{i+1}" if num_bands > 5 else ["BASS", "MID", "HIGH", "TREBLE", "PRES"][i] if i < 5 else f"B{i+1}"
            self.draw_bar(label, val, rect, color)
            sphere_x, sphere_center = sidebar_x + 10 + meter_w + 25, (sidebar_x + 10 + meter_w + 25, rect[1] + bar_h // 2)
            max_r, current_r = min(20, bar_h + 4), int(val * min(20, bar_h + 4))
            pygame.draw.circle(self.screen, (max(0, color[0]-120), max(0, color[1]-120), max(0, color[2]-120)), sphere_center, max_r, 1)
            if current_r > 0: pygame.draw.circle(self.screen, color, sphere_center, current_r)
        
        stem_y_start = start_y + num_bands * (bar_h + 8) + 20
        self.draw_text("AI STEMS ANALYZER", (sidebar_x + 10, stem_y_start - 25), color=(0, 255, 200), bold=True)
        stems = features['stems']
        stem_colors = {"vocals": (255, 180, 50), "drums": (255, 80, 80), "bass": (255, 100, 200), "guitar": (100, 255, 100), "piano": (100, 200, 255), "other": (180, 180, 180)}
        for i, s_type in enumerate(["vocals", "drums", "bass", "guitar", "piano", "other"]):
            val, color = stems[s_type], stem_colors.get(s_type, WHITE)
            s_rect = (sidebar_x + 10, stem_y_start + i * 33, meter_w, 20)
            self.draw_bar(s_type.upper(), val, s_rect, color)
            v_sphere_center, v_max_r = (sidebar_x + 10 + meter_w + 25, s_rect[1] + 10), 15
            v_current_r = int(val * v_max_r)
            pygame.draw.circle(self.screen, (max(0, color[0]-150), max(0, color[1]-150), max(0, color[2]-150)), v_sphere_center, v_max_r, 1)
            if v_current_r > 0: pygame.draw.circle(self.screen, color, v_sphere_center, v_current_r)

        fft_rect = pygame.Rect(40, 60, sidebar_x - 80, 240)
        pygame.draw.rect(self.screen, (15, 15, 15), fft_rect, border_radius=8)
        self.draw_text("FFT SPECTROGRAM (RAW BINS)", (40, 40))
        spectrum = features['spectrum']
        bin_count = len(spectrum[:512])
        bin_w = fft_rect.width / bin_count 
        for i, val in enumerate(spectrum[:512]):
            h, rect_x = int(val * fft_rect.height), int(fft_rect.left + i*bin_w)
            pygame.draw.rect(self.screen, (int(max(0, 255 - i/2)), int(min(255, i/2)), 200), (rect_x, int(fft_rect.bottom - h), int(max(1, bin_w)), h))

        graph_rect = pygame.Rect(40, 360, sidebar_x - 80, 240)
        pygame.draw.rect(self.screen, (15, 15, 15), graph_rect, border_radius=8)
        self.draw_text("ENERGY HISTORY (PINK=BASS, GOLD=VOCALS)", (40, 340))
        self.bass_history.pop(0); self.bass_history.append(features['bass'])
        if not hasattr(self, 'vocal_history'): self.vocal_history = [0.0] * self.history_len
        self.vocal_history.pop(0); self.vocal_history.append(features['vocal'])
        b_points, v_points, step_x = [], [], graph_rect.width / self.history_len
        for i in range(self.history_len):
            px = int(graph_rect.left + i * step_x)
            b_points.append((px, int(graph_rect.bottom - (self.bass_history[i] * graph_rect.height))))
            v_points.append((px, int(graph_rect.bottom - (self.vocal_history[i] * graph_rect.height))))
        if len(b_points) > 1: pygame.draw.lines(self.screen, PINK, False, b_points, 2)
        if len(v_points) > 1: pygame.draw.lines(self.screen, (255, 180, 50), False, v_points, 2)

        self.draw_text("DEVELOPER INFO:", (40, 650), color=(150, 150, 150))
        self.draw_text("Extraction Rate: 60Hz | FFT Window: 2048 samples | Normalization: Amplitude-to-DB -> [0..1]", (40, 675), color=(100, 100, 100))
        self.draw_text("PRESS [R] TO RENDER 5S ANIMATION SEED (MP4)", (40, 700), color=(255, 100, 100), bold=True)

def main():
    parser = argparse.ArgumentParser(description="Audio Feature Debugger")
    parser.add_argument("--file", "-f", required=True)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--mode", type=str, choices=["vocals", "demucs", "roformer"], default="demucs")
    args = parser.parse_args()

    pygame.init(); pygame.mixer.init()
    screen = pygame.display.set_mode((1280, 720))
    pygame.display.set_caption("ARC - Feature Debugger")
    clock = pygame.time.Clock()
    file_path = Path(args.file)
    if not file_path.exists(): sys.exit(1)

    extractor = AudioFeatureExtractor(file_path, fps=args.fps, separation_mode=args.mode)
    dashboard = DebugDashboard(screen)
    pygame.mixer.music.load(str(file_path)); pygame.mixer.music.play()
    
    running, use_smoothing, use_normalization = True, True, True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_s: use_smoothing = not use_smoothing
                elif event.key == pygame.K_n: use_normalization = not use_normalization
                elif event.key == pygame.K_k: extractor.update_contrast(-0.02)
                elif event.key == pygame.K_l: extractor.update_contrast(0.02)
                elif event.key == pygame.K_u: extractor.vocal_threshold -= 0.05
                elif event.key == pygame.K_i: extractor.vocal_threshold += 0.05
                elif event.key == pygame.K_r:
                    cmd = [sys.executable, str(PROJECT_ROOT / "generators" / "animation_generator.py"), "--file", str(file_path), "--mode", extractor.separation_mode, "--out", f"render_output/{file_path.stem}_seed.mp4"]
                    os.makedirs(PROJECT_ROOT / "render_output", exist_ok=True)
                    subprocess.Popen(cmd)
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS: extractor.update_num_bands(extractor.num_bands + 1)
                elif event.key == pygame.K_MINUS: extractor.update_num_bands(extractor.num_bands - 1)
                elif event.key == pygame.K_SPACE:
                    if pygame.mixer.music.get_busy(): pygame.mixer.music.pause()
                    else: pygame.mixer.unpause()

        play_time_ms = pygame.mixer.music.get_pos()
        if play_time_ms == -1: running = False; continue
        features = extractor.get_features_at_time(play_time_ms / 1000.0, use_smoothing=use_smoothing, use_normalization=use_normalization)
        if features:
            screen.fill(DARK_BG)
            dashboard.render(features, play_time_ms / 1000.0, dt)
            s_color = (100, 255, 100) if use_smoothing else (255, 100, 100)
            n_color = (100, 255, 100) if use_normalization else (255, 100, 100)
            dashboard.draw_text(f"SMOOTHING: {'ON' if use_smoothing else 'OFF'} [S]", (20, 20), color=s_color, bold=True)
            dashboard.draw_text(f"NORMALIZED: {'ON' if use_normalization else 'OFF'} [N]", (20, 45), color=n_color, bold=True)
            dashboard.draw_text(f"CONTRAST: {extractor.contrast_level:.2f} [K/L]", (20, 70), (200, 200, 200))
            dashboard.draw_text(f"AI SEP: {extractor.separation_mode.upper()}", (20, 95), (255, 180, 50))
            pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()
