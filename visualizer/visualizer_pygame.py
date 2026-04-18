import pygame
import math
import numpy as np

PINK = (236, 72, 153)
BLUE = (59, 130, 246)
DARK_BG = (10, 10, 10)
WHITE = (255, 255, 255)

class PygameVisualizer:
    """Renders audio-reactive visuals using Pygame."""
    def __init__(self, surface):
        self.surface = surface
        self.width, self.height = surface.get_size()
        self.center_x, self.center_y = self.width // 2, self.height // 2
        self.time = 0
        
    def update_surface(self, surface):
        self.surface = surface
        self.width, self.height = surface.get_size()
        self.center_x, self.center_y = self.width // 2, self.height // 2

    def draw_ncs_circle(self, features, color_primary=PINK, color_secondary=BLUE):
        pulse, spectrum = features['pulse'], features['spectrum']
        radius = 150 + (pulse - 1) * 200
        bars_count, step = 80, (math.pi * 2) / 80
        rotation = self.time * 0.15
        for i in range(bars_count):
            val = spectrum[i + 5] if i + 5 < len(spectrum) else 0
            h = 8 + (val ** 1.5) * 180
            angle = i * step + rotation
            start_dist, end_dist = radius + 10, radius + 10 + h
            start_x, start_y = int(self.center_x + math.cos(angle) * start_dist), int(self.center_y + math.sin(angle) * start_dist)
            end_x, end_y = int(self.center_x + math.cos(angle) * end_dist), int(self.center_y + math.sin(angle) * end_dist)
            pygame.draw.line(self.surface, color_primary, (start_x, start_y), (end_x, end_y), 6)
            pygame.draw.circle(self.surface, WHITE, (end_x, end_y), 2)
        pygame.draw.circle(self.surface, (40, 40, 40), (self.center_x, self.center_y), int(radius + 150), 1)

    def draw_linear_bars(self, features, color_primary=PINK, color_secondary=BLUE):
        spectrum = features['spectrum']
        bars_count = 64
        bar_w, gap = self.width // bars_count, 2
        for i in range(bars_count):
            val = spectrum[i * 2] if i * 2 < len(spectrum) else 0
            bar_h = int(10 + (val ** 1.3) * (self.height * 0.35))
            pygame.draw.rect(self.surface, color_primary, pygame.Rect(int(i * bar_w + gap), int(self.center_y - bar_h), int(bar_w - gap), bar_h), border_radius=3)
            reflect_color = (color_primary[0]//4, color_primary[1]//4, color_primary[2]//4)
            pygame.draw.rect(self.surface, reflect_color, pygame.Rect(int(i * bar_w + gap), int(self.center_y), int(bar_w - gap), int(bar_h * 0.3)), border_radius=2)
        pygame.draw.line(self.surface, (100, 100, 100), (0, self.center_y), (self.width, self.center_y), 1)

    def draw_particles(self, features, count=50):
        bass = features['bass']
        for i in range(count):
            seed = i * 127.1
            x = (math.sin(seed) * 1000 + self.time * (20 + (i % 7) * 15)) % self.width
            y = self.height - ((self.time * (10 + (i % 5)*10) + seed * 20) % self.height)
            size, alpha = 2 + (i % 3) + bass * 5, int(100 + bass * 155)
            p_color = (min(255, PINK[0] + alpha // 2), min(255, PINK[1] + alpha // 2), min(255, PINK[2] + alpha // 2))
            pygame.draw.circle(self.surface, p_color, (int(x), int(y)), int(size))

    def render(self, features, preset='NCS Circle', dt=0.016):
        self.time += dt
        if preset == 'NCS Circle': self.draw_ncs_circle(features)
        elif preset == 'Linear Bars': self.draw_linear_bars(features)
        self.draw_particles(features)
