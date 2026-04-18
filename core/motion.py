import numpy as np

class VerticalBounce:
    """
    Handles vertical oscillation (zigzag) logic within specified margins.
    Proportional to canvas resolution for consistency.
    """
    def __init__(self, resolution_y, start_y_ratio=0.5, speed_ratio=0.25, margin_ratio=0.1, initial_dir=1):
        self.res_y = resolution_y
        self.margin = int(resolution_y * margin_ratio)
        self.min_y = self.margin
        self.max_y = resolution_y - self.margin
        self.current_y = int(resolution_y * start_y_ratio)
        self.direction = initial_dir
        self.speed = int(resolution_y * speed_ratio)

    def update(self, dt):
        movement = self.direction * (self.speed * dt)
        self.current_y += movement
        if self.current_y <= self.min_y:
            self.current_y = self.min_y
            self.direction = 1
        elif self.current_y >= self.max_y:
            self.current_y = self.max_y
            self.direction = -1
        return int(self.current_y)

def create_zigzag_preset(resolution_y, speed_ratio=0.25):
    l_engine = VerticalBounce(resolution_y, start_y_ratio=0.5, initial_dir=-1, speed_ratio=speed_ratio)
    r_engine = VerticalBounce(resolution_y, start_y_ratio=0.5, initial_dir=1, speed_ratio=speed_ratio)
    return l_engine, r_engine
