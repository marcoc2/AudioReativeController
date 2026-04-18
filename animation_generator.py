import os
import sys
import numpy as np
import pygame
from pathlib import Path
import subprocess
import shutil
from collections import deque

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from core.feature_extractor import AudioFeatureExtractor
    from core.motion import create_zigzag_preset
except ImportError:
    print("[Animation Gen] Error: Required modules not found.")
    sys.exit(1)

def draw_transparent_circle(surface, color, center, radius, alpha):
    if radius <= 0 or alpha <= 0: return
    temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(temp_surf, (*color, alpha), (radius, radius), radius)
    surface.blit(temp_surf, (center[0] - radius, center[1] - radius))

def generate_animation_mp4(audio_path, output_mp4, start_sec=0.0, duration_sec=10.0, fps=24, 
                          resolution=(1024, 1024), contrast=0.45, mode="demucs", 
                          preset="none", speed=0.25, trail_count=0, scale=1.0):
    audio_path = Path(audio_path)
    if not audio_path.exists(): return
    output_path = Path(output_mp4)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n[Animation Gen] STARTING OFFLINE RENDER\n  Input: {audio_path.name}\n{'='*60}\n")
    extractor = AudioFeatureExtractor(str(audio_path), fps=fps, separation_mode=mode)
    extractor.contrast_level = contrast
    extractor.update_num_bands(32)
    
    if start_sec + duration_sec > extractor.duration:
        duration_sec = max(0, extractor.duration - start_sec)

    temp_dir = Path("render_temp")
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    pygame.init()
    surface = pygame.Surface(resolution)
    total_frames = int(duration_sec * fps)
    center_y, left_x, right_x = resolution[1] // 2, resolution[0] // 4, (resolution[0] // 4) * 3
    max_radius = int((resolution[0] // 4 - 50) * scale)

    m_left, m_right = None, None
    if preset == "zigzag": m_left, m_right = create_zigzag_preset(resolution[1], speed_ratio=speed)

    l_history = deque(maxlen=trail_count) if trail_count > 0 else None
    r_history = deque(maxlen=trail_count) if trail_count > 0 else None

    for frame_idx in range(total_frames):
        time_sec = start_sec + (frame_idx / fps)
        cur_l_y, cur_r_y = center_y, center_y
        if m_left and m_right:
            cur_l_y, cur_r_y = m_left.update(1.0/fps), m_right.update(1.0/fps)
        
        features = extractor.get_features_at_time(time_sec, apply_gate=False)
        if not features: break
        
        surface.fill((0, 0, 0))
        l_rad, r_rad = int(min(1.0, features.get('bass', 0) * 1.2) * max_radius), int(features['stems'].get('vocals', 0) * max_radius)

        if trail_count > 0:
            for i, (py, pr) in enumerate(l_history):
                draw_transparent_circle(surface, (200, 40, 40), (left_x, py), pr, int(255 * (i+1)/(trail_count+1)))
            for i, (py, pr) in enumerate(r_history):
                draw_transparent_circle(surface, (200, 140, 40), (right_x, py), pr, int(255 * (i+1)/(trail_count+1)))
            l_history.append((cur_l_y, l_rad)); r_history.append((cur_r_y, r_rad))

        if l_rad > 1:
            pygame.draw.circle(surface, (255, 50, 50), (left_x, cur_l_y), l_rad)
            pygame.draw.circle(surface, (255, 150, 150), (left_x, cur_l_y), int(l_rad * 0.3))
        if r_rad > 1:
            pygame.draw.circle(surface, (255, 180, 50), (right_x, cur_r_y), r_rad)
            pygame.draw.circle(surface, (255, 230, 200), (right_x, cur_r_y), int(r_rad * 0.3))

        pygame.image.save(surface, str(temp_dir / f"{frame_idx:04d}.png"))
        if frame_idx % 24 == 0: sys.stdout.write(f"\r  Progress: {frame_idx}/{total_frames}"); sys.stdout.flush()

    pygame.quit()
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", str(temp_dir / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(output_mp4)], capture_output=True)
    if temp_dir.exists(): shutil.rmtree(temp_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--mode", default="demucs")
    parser.add_argument("--preset", default="none", choices=["none", "zigzag"])
    parser.add_argument("--speed", type=float, default=0.25)
    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.out is None: args.out = f"render_output/{Path(args.file).stem}_seed.mp4"
    generate_animation_mp4(args.file, args.out, start_sec=args.start, duration_sec=args.duration, mode=args.mode, preset=args.preset, speed=args.speed, trail_count=args.trail, scale=args.scale)
