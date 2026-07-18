"""Stream-based multi-track audio mixer via sounddevice.

Tracks are mono float32 arrays at a common sample rate.
Faders can be adjusted at any time during playback.
"""
from __future__ import annotations

import threading
from typing import Dict, Optional

import numpy as np
import sounddevice as sd


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo or multi-channel array to mono."""
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=0).astype(np.float32)


def rms_envelope(audio: np.ndarray, n_pixels: int) -> np.ndarray:
    """Downsample audio to *n_pixels* RMS values, normalised 0..1."""
    mono = to_mono(audio)
    spp  = max(1, len(mono) // n_pixels)
    n    = len(mono) // spp
    chunks = mono[: n * spp].reshape(n, spp)
    env    = np.sqrt(np.mean(chunks ** 2, axis=1))
    peak   = env.max()
    if peak > 0:
        env /= peak
    if len(env) < n_pixels:
        env = np.pad(env, (0, n_pixels - len(env)))
    return env[:n_pixels].astype(np.float32)


class AudioPlayer:
    """Real-time multi-track mixer.

    Usage::
        player = AudioPlayer(sample_rate=44100)
        player.add_track("mix",  audio_array)
        player.add_track("solo", solo_array, fader=0.8)
        player.play()
        player.set_fader("solo", 0.5)
        player.seek(32.0)
        player.pause()
    """

    def __init__(self, sample_rate: int = 44100, block_size: int = 1024) -> None:
        self.sr         = sample_rate
        self.block_size = block_size
        self._tracks:  Dict[str, np.ndarray] = {}
        self._faders:  Dict[str, float]      = {}
        self._pos:     int  = 0
        self._playing: bool = False
        self._stream:  Optional[sd.OutputStream] = None
        self._lock     = threading.Lock()

    # ------------------------------------------------------------------
    # Track management

    def add_track(self, name: str, audio: np.ndarray, fader: float = 1.0) -> None:
        with self._lock:
            self._tracks[name] = to_mono(audio)
            self._faders[name] = float(fader)

    def set_fader(self, name: str, value: float) -> None:
        self._faders[name] = float(np.clip(value, 0.0, 2.0))

    def remove_track(self, name: str) -> None:
        with self._lock:
            self._tracks.pop(name, None)
            self._faders.pop(name, None)

    @property
    def track_names(self) -> list[str]:
        return list(self._tracks.keys())

    # ------------------------------------------------------------------
    # Playback control

    def play(self, start_t: Optional[float] = None) -> None:
        if start_t is not None:
            self._pos = int(start_t * self.sr)
        self._playing = True
        if self._stream is None or not self._stream.active:
            self._stream = sd.OutputStream(
                samplerate  = self.sr,
                channels    = 2,
                dtype       = np.float32,
                blocksize   = self.block_size,
                callback    = self._callback,
            )
            self._stream.start()

    def pause(self) -> None:
        self._playing = False

    def stop(self) -> None:
        self._playing = False
        self._pos = 0
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def seek(self, t: float) -> None:
        with self._lock:
            self._pos = int(np.clip(t * self.sr, 0, self._max_len))

    @property
    def current_time(self) -> float:
        return self._pos / self.sr

    @property
    def output_latency(self) -> float:
        """Output latency of the active stream in seconds (0.0 if idle).

        ``current_time`` counts samples handed to the driver, which are
        heard roughly this much later — useful as a reference when
        calibrating UI/audio sync.
        """
        if self._stream is None:
            return 0.0
        try:
            return float(self._stream.latency)
        except Exception:
            return 0.0

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def _max_len(self) -> int:
        if not self._tracks:
            return 0
        return max(len(a) for a in self._tracks.values())

    # ------------------------------------------------------------------
    # Audio callback (runs in a separate real-time thread)

    def _callback(
        self,
        outdata: np.ndarray,
        frames:  int,
        time_info,
        status,
    ) -> None:
        with self._lock:
            if not self._playing or not self._tracks:
                outdata[:] = 0.0
                return

            mix = np.zeros(frames, dtype=np.float32)
            for name, audio in self._tracks.items():
                fader = self._faders.get(name, 1.0)
                end   = min(self._pos + frames, len(audio))
                if end <= self._pos:
                    continue
                chunk = audio[self._pos : end]
                if len(chunk) < frames:
                    chunk = np.pad(chunk, (0, frames - len(chunk)))
                mix += chunk * fader

            np.clip(mix, -1.0, 1.0, out=mix)
            outdata[:, 0] = mix
            if outdata.shape[1] > 1:
                outdata[:, 1] = mix

            self._pos += frames
            if self._pos >= self._max_len:
                self._playing = False
