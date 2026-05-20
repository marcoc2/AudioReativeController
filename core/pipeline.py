"""ARCPipeline — query interface between audio features and consumers.

FrameSlice  — one moment in time: musical context + mapped control values.
ARCPipeline — wraps extractor + grid + scene config; answers query(t).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import yaml

if TYPE_CHECKING:
    from core.rhythm.midi_automation import MidiAutomationReader


@dataclass
class FrameSlice:
    """One queried moment in time.

    bar and beat are None when the grid has no reliable markers (no MIDI,
    or beat tracking failed to find downbeats).
    controls contains values resolved from the scene YAML mappings.
    """
    frame: int
    t: float
    bar_phase: float    # 0..1 continuous position within bar (always present)
    beat_phase: float   # 0..1 continuous position within beat (always present)
    bar: Optional[int]  # absolute bar index (None if no downbeat markers)
    beat: Optional[int] # absolute beat index (None if no beat markers)
    controls: dict = field(default_factory=dict)


class ARCPipeline:
    """Combines extractor + grid + scene config into a queryable interface.

    Usage:
        pipeline = ARCPipeline.from_yaml(extractor, grid, "scenes/default.yaml")
        slice = pipeline.query(t=2.34, frame_idx=56)
    """

    def __init__(
        self,
        extractor,
        grid,
        scene: dict,
        midi_automation: "Optional[MidiAutomationReader]" = None,
    ):
        self._extractor = extractor
        self._grid = grid
        self._mappings = scene.get("controls", {})
        self._midi = midi_automation

    @classmethod
    def from_yaml(
        cls,
        extractor,
        grid,
        path: str,
        midi_automation: "Optional[MidiAutomationReader]" = None,
    ) -> "ARCPipeline":
        with open(path, "r", encoding="utf-8") as f:
            scene = yaml.safe_load(f)
        return cls(extractor, grid, scene or {}, midi_automation=midi_automation)

    @property
    def bar_duration(self) -> float:
        return self._grid.bar_duration

    def query(self, t: float, frame_idx: Optional[int] = None) -> Optional[FrameSlice]:
        """Return a FrameSlice for time t, or None if t is past the audio end."""
        features = self._extractor.get_features_at_time(t, apply_gate=False)
        if features is None:
            return None

        if frame_idx is None:
            frame_idx = int(features.get("frame_idx", 0))

        bar_phase = self._grid.phase(t)
        beat_phase = self._grid.beat_phase(t)

        # grid values available as source references in the scene YAML
        grid_vals = {"bar_phase": bar_phase, "beat_phase": beat_phase}

        controls = {}
        for name, spec in self._mappings.items():
            val = self._resolve_source(spec["source"], features, grid_vals, frame_idx)
            if val is None:
                continue
            val = float(val)
            if "scale" in spec:
                val *= float(spec["scale"])
            if "range" in spec:
                lo, hi = spec["range"]
                val = lo + val * (hi - lo)
            if "clip" in spec:
                lo, hi = spec["clip"]
                val = max(float(lo), min(float(hi), val))
            controls[name] = val

        return FrameSlice(
            frame=frame_idx,
            t=t,
            bar_phase=bar_phase,
            beat_phase=beat_phase,
            bar=self._resolve_bar(t),
            beat=self._resolve_beat(t),
            controls=controls,
        )

    # ------------------------------------------------------------------
    # internals

    def _resolve_source(
        self, source: str, features: dict, grid_vals: dict, frame_idx: int
    ):
        """Resolve a dotted source path to a scalar value.

        Supported prefixes:
            midi.<lane>   — MidiAutomationReader lane (CC or note-velocity)
            subbands.<x>  — audio sub-band energy
            stems.<x>     — stem energy (AI-extracted or prebuilt)
            <flat key>    — top-level feature (centroid, flux, …)
            bar_phase / beat_phase — from grid
        """
        if source in grid_vals:
            return grid_vals[source]

        if source.startswith("midi."):
            if self._midi is None:
                return 0.0
            return self._midi.get(source[5:], frame_idx)

        parts = source.split(".", 1)
        if len(parts) == 1:
            return features.get(source)
        top, key = parts
        sub = features.get(top)
        if isinstance(sub, dict):
            return sub.get(key)
        return None

    def _resolve_bar(self, t: float) -> Optional[int]:
        db = self._grid.downbeats
        if db is None or len(db) == 0:
            return None
        idx = int(np.searchsorted(db, t, side="right")) - 1
        return max(0, idx)

    def _resolve_beat(self, t: float) -> Optional[int]:
        beats = self._grid.beats
        if beats is None or len(beats) == 0:
            return None
        idx = int(np.searchsorted(beats, t, side="right")) - 1
        return max(0, idx)
