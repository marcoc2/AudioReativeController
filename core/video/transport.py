"""ClipTransport — stateful playhead over a sequence of clips.

Musical events mutate the transport (reverse, next_clip, restart, ...);
``advance()`` moves the playhead by one output frame, ping-ponging at the
clip edges so playback never runs out of frames mid-bar.
"""
from __future__ import annotations


class ClipTransport:
    def __init__(self, n_clips: int, start_clip: int = 0):
        if n_clips < 1:
            raise ValueError("need at least one clip")
        self.n_clips = n_clips
        self.clip_idx = start_clip % n_clips
        self.pos = 0.0        # fractional frame position within the clip
        self.direction = 1    # +1 forward, -1 reverse
        self.speed = 1.0      # clip frames per output frame

    # ------------------------------------------------------------------
    # actions (triggerable from MIDI events)

    def reverse(self) -> None:
        self.direction *= -1

    def restart(self) -> None:
        self.pos = 0.0
        self.direction = 1

    def next_clip(self) -> None:
        self.set_clip(self.clip_idx + 1)

    def set_clip(self, idx: int) -> None:
        self.clip_idx = idx % self.n_clips
        self.pos = 0.0
        self.direction = 1

    # ------------------------------------------------------------------

    def advance(self, clip_len: int, steps: float = 1.0) -> None:
        """Move the playhead within a clip of ``clip_len`` frames.

        Bounces (ping-pong) at both edges; the loop handles overshoot
        across multiple bounces for short clips or high speeds.
        """
        if clip_len <= 1:
            self.pos = 0.0
            return
        last = float(clip_len - 1)
        self.pos += self.direction * self.speed * steps
        while self.pos < 0.0 or self.pos > last:
            if self.pos < 0.0:
                self.pos = -self.pos
                self.direction = 1
            else:
                self.pos = 2.0 * last - self.pos
                self.direction = -1

    @property
    def frame_index(self) -> int:
        return int(round(self.pos))
