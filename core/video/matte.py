"""Layer mattes — any layer of a scene, held inside (or outside) the objects of a mask track.

    - source: feedback
      mask:
        track: render_output/x_base_segments   # a mask track (segment_video.py); may be left
                                               # out when the render is given --masks
        where: outside        # inside: only on the objects | outside: only on the background
        objects: [8, 9]       # which ones (ids in the track's track.json); default: all
        grow: 0               # pixels (at 720p) to grow (+) or shrink (-) the objects by
        feather: 2            # pixels (at 720p) of soft edge

The layer is worked out as usual and then kept only where the matte is: elsewhere the
picture below it shows untouched. So ``feedback`` / ``slitscan`` / a shader ``outside``
melts the world while the characters stay sharp, and ``inside`` does the opposite.
A post-op ``inside`` still sees the whole frame (a blur reaches across the outline), only
its result is clipped. The track is found by song time; before or after it there are no
objects (``inside`` shows nothing, ``outside`` everything). A track made at another size
is scaled to the render.
"""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from core.segment.mask_track import MaskTrack

WHERE = ("inside", "outside")


class LayerMatte:
    """``matte(t)`` -> float32 H×W×1 in 0..1: how much of the layer shows at each pixel."""

    def __init__(self, spec, width: int, height: int, default_track: Optional[str] = None,
                 tracks: Optional[Dict[str, MaskTrack]] = None):
        spec = dict(spec or {})
        folder = spec.get("track") or default_track
        if not folder:
            raise ValueError("mask: needs a track (mask: {track: <folder>}) or the render's --masks")
        tracks = tracks if tracks is not None else {}
        if folder not in tracks:
            tracks[folder] = MaskTrack(folder)
        self.track = tracks[folder]
        self.where = spec.get("where", "inside")
        if self.where not in WHERE:
            raise ValueError(f"mask: where {self.where!r} (use inside or outside)")
        objects = spec.get("objects")
        self.objects = None if objects is None else np.array(sorted(int(o) for o in objects))
        if self.objects is not None:
            unknown = [int(o) for o in self.objects if int(o) not in self.track.objects]
            if unknown:
                raise ValueError(f"mask: objects {unknown} are not in {folder} "
                                 f"(it has {sorted(self.track.objects)})")
        scale = height / 720.0
        self.grow = int(round(float(spec.get("grow", 0)) * scale))
        self.feather = float(spec.get("feather", 2)) * scale
        self.W, self.H = width, height
        self._last = (None, None)

    def selection(self, t: float) -> Optional[np.ndarray]:
        """The objects picked at song time ``t``: uint8 0/1 at the track's own size (None
        outside the track). The GPU compositor stretches, grows and feathers it itself."""
        i = self.track.frame_at(t)
        if i is None:
            return None
        lab = self.track.labels(i)
        return (lab > 0 if self.objects is None else np.isin(lab, self.objects)).astype(np.uint8)

    def __call__(self, t: float) -> np.ndarray:
        i = self.track.frame_at(t)
        if self._last[0] == i and i is not None:
            return self._last[1]
        if i is None:
            m = np.zeros((self.H, self.W), np.float32)
        else:
            m = self.selection(t).astype(np.float32)
            if m.shape != (self.H, self.W):
                m = cv2.resize(m, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            if self.grow:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(self.grow) + 1,) * 2)
                m = cv2.dilate(m, k) if self.grow > 0 else cv2.erode(m, k)
            if self.feather > 0.25:
                m = cv2.GaussianBlur(m, (0, 0), self.feather)
        if self.where == "outside":
            m = 1.0 - m
        m = m[..., None]
        self._last = (i, m)
        return m


def apply_matte(below: np.ndarray, above: np.ndarray, m: np.ndarray) -> np.ndarray:
    """``above`` where the matte is, ``below`` elsewhere (uint8 frames, matte H×W×1)."""
    if above is below:
        return above
    out = below.astype(np.float32) * (1.0 - m) + above.astype(np.float32) * m
    return (out + 0.5).astype(np.uint8)
