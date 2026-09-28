"""Reading a whole video frame by frame through ffmpeg (uint8 RGB), for the passes that
work on a render after it is made (SAM2 masks, followed drops)."""
from __future__ import annotations

import subprocess
from typing import Iterator, Tuple

import numpy as np


def probe_video(path: str) -> Tuple[int, int, float]:
    """(width, height, fps) of the first video stream."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip().split(",")
    num, den = out[2].split("/")
    return int(out[0]), int(out[1]), float(num) / float(den)


def iter_frames(path: str) -> Iterator[np.ndarray]:
    """Every frame in order, H×W×3 uint8 RGB (read-only arrays)."""
    W, H, _ = probe_video(path)
    size = W * H * 3
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
                            "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    try:
        while True:
            buf = dec.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    finally:
        dec.stdout.close()
        dec.kill()
        dec.wait()


def open_encoder(path: str, width: int, height: int, fps: float, audio_from: str = None,
                 crf: int = 18) -> subprocess.Popen:
    """An ffmpeg that takes raw RGB frames on stdin (``.stdin.write(frame.tobytes())``);
    the audio of ``audio_from``, when given, is copied over."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-"]
    if audio_from:
        cmd += ["-i", str(audio_from), "-map", "0:v", "-map", "1:a?", "-c:a", "copy", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)
