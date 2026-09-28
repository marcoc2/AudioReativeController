"""RegionTracker — follow regions of a video frame by frame with SAM2 (Meta, Apache 2.0).

Built for a render loop, not for a video on disk: frames arrive one at a time, a region
can be born on any frame (from a mask or a point) and is followed for a fixed number
of frames, many regions at once. SAM2's video predictor wants the whole video up
front and no new objects once tracking has started, so each region gets its own
inference state over its own window and all of them advance in lockstep; the image
encoder, the expensive part, runs once per frame and every live region reads that
one result.

    tracker = RegionTracker(weights_dir, model="base_plus")
    for i, frame in enumerate(frames):             # uint8 RGB, all the same size
        tracker.set_frame(i, frame)
        if something_happens:
            tracker.add("drop-7", frames=90, mask=region)   # or point=(x, y)
        masks = tracker.step()                     # {region id: bool H×W} for the live ones

Weights: a SAM 2.1 checkpoint in ``weights_dir`` — the official ``.pt`` or the fp16
``.safetensors`` repack (Kijai/sam2-safetensors, the ones ComfyUI keeps in models/sam2);
see ``find_checkpoint``.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# model -> (config inside the sam2 package, checkpoint name stems to look for)
MODELS = {
    "tiny":      ("configs/sam2.1/sam2.1_hiera_t.yaml",  ("sam2.1_hiera_tiny",)),
    "small":     ("configs/sam2.1/sam2.1_hiera_s.yaml",  ("sam2.1_hiera_small",)),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", ("sam2.1_hiera_base_plus",)),
    "large":     ("configs/sam2.1/sam2.1_hiera_l.yaml",  ("sam2.1_hiera_large",)),
}
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def find_checkpoint(weights_dir: str, model: str = "base_plus") -> Path:
    """The checkpoint file for ``model`` in ``weights_dir`` (.pt, .safetensors or -fp16.safetensors)."""
    if model not in MODELS:
        raise ValueError(f"unknown SAM2 model {model!r} (use one of {sorted(MODELS)})")
    root = Path(weights_dir)
    for stem in MODELS[model][1]:
        for name in (f"{stem}.pt", f"{stem}.safetensors", f"{stem}-fp16.safetensors"):
            if (root / name).is_file():
                return root / name
    raise FileNotFoundError(f"no SAM 2.1 {model} checkpoint in {root} "
                            f"(expected {MODELS[model][1][0]}.pt or .safetensors)")


class _Window:
    """What a region's inference state sees as its video: ``length`` frames starting at
    ``offset`` of the stream. Only the tracker's current frame can be read — in lockstep
    that is the only one SAM2 ever asks for."""

    def __init__(self, tracker: "RegionTracker", offset: int, length: int):
        self.tracker, self.offset, self.length = tracker, offset, length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.tracker._image(self.offset + int(i))


class RegionTracker:
    def __init__(self, weights_dir: str, model: str = "base_plus", device: str = "cuda"):
        import torch
        import sam2.sam2_video_predictor as vp
        from sam2.build_sam import build_sam2_video_predictor

        vp.tqdm = lambda it, **kw: it            # a progress bar per region would flood the log

        self.torch = torch
        self.device = torch.device(device)
        ckpt = find_checkpoint(weights_dir, model)
        self.predictor = build_sam2_video_predictor(MODELS[model][0], ckpt_path=None, device=device)
        if ckpt.suffix == ".safetensors":
            from safetensors.torch import load_file
            sd = load_file(str(ckpt))
        else:
            sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)["model"]
        missing, unexpected = self.predictor.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"{ckpt.name} does not fit SAM2 {model}: "
                               f"{len(missing)} missing, {len(unexpected)} unexpected tensors")
        self.predictor.eval()
        self.size = int(self.predictor.image_size)
        self._mean = torch.tensor(_MEAN, device=self.device).view(3, 1, 1)
        self._std = torch.tensor(_STD, device=self.device).view(3, 1, 1)

        self._index = -1                    # the current frame of the stream
        self._frame: Optional[np.ndarray] = None
        self._feat_key, self._feat = None, None
        self._regions: Dict[object, Tuple[dict, object, int]] = {}   # id -> (state, generator, offset)
        self._shape: Optional[Tuple[int, int]] = None

        # one image-encoder pass per frame, shared by every region's state
        original = self.predictor._get_image_feature

        def shared_feature(state, frame_idx, batch_size):
            key = state["images"].offset + frame_idx
            state["cached_features"] = {frame_idx: self._feat} if key == self._feat_key else {}
            out = original(state, frame_idx, batch_size)
            self._feat_key, self._feat = key, state["cached_features"][frame_idx]
            return out

        self.predictor._get_image_feature = shared_feature

    # ------------------------------------------------------------------ stream
    def set_frame(self, index: int, frame: np.ndarray) -> None:
        """The stream's next frame (uint8 RGB H×W×3); call before ``add``/``step`` on it."""
        if self._shape is None:
            self._shape = frame.shape[:2]
        elif frame.shape[:2] != self._shape:
            raise ValueError(f"frame {index} is {frame.shape[:2]}, the stream is {self._shape}")
        self._index, self._frame = int(index), frame

    def _image(self, index: int):
        if index != self._index:
            raise RuntimeError(f"SAM2 asked for frame {index} while the stream is on {self._index}")
        torch = self.torch
        img = torch.from_numpy(np.array(self._frame, copy=True)).to(self.device)   # decoded frames are read-only
        img = img.permute(2, 0, 1).float().div_(255.0)[None]
        img = torch.nn.functional.interpolate(img, size=(self.size, self.size), mode="bicubic",
                                              align_corners=False, antialias=True)[0]
        return (img.clamp_(0, 1) - self._mean) / self._std

    @contextmanager
    def _autocast(self):
        torch = self.torch
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            yield

    # ----------------------------------------------------------------- regions
    def add(self, region_id, frames: int, mask: Optional[np.ndarray] = None,
            point: Optional[Tuple[float, float]] = None) -> None:
        """A region born on the current frame, to follow for ``frames`` frames (this one
        included): from a mask (bool H×W, the best prompt) or a point (x, y in pixels)."""
        if (mask is None) == (point is None):
            raise ValueError("give a mask or a point")
        if region_id in self._regions:
            raise ValueError(f"region {region_id!r} is already being followed")
        import sam2.sam2_video_predictor as vp
        H, W = self._shape
        window = _Window(self, self._index, max(1, int(frames)))
        loader = vp.load_video_frames
        vp.load_video_frames = lambda **kw: (window, H, W)   # init_state reads the video from disk
        try:
            with self._autocast():
                state = self.predictor.init_state(video_path=None)
                if mask is not None:
                    self.predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=np.asarray(mask, bool))
                else:
                    self.predictor.add_new_points_or_box(
                        state, frame_idx=0, obj_id=1,
                        points=np.array([point], np.float32), labels=np.array([1], np.int32))
        finally:
            vp.load_video_frames = loader
        gen = self.predictor.propagate_in_video(state, start_frame_idx=0,
                                                max_frame_num_to_track=window.length)
        self._regions[region_id] = (state, gen, window.offset)

    def step(self) -> Dict[object, np.ndarray]:
        """Advance every live region onto the current frame: {id: bool mask H×W}. A region
        whose window is over is dropped (and missing from the result)."""
        out = {}
        with self._autocast():
            for rid, (state, gen, offset) in list(self._regions.items()):
                try:
                    idx, _, masks = next(gen)
                except StopIteration:
                    del self._regions[rid]
                    continue
                if offset + idx != self._index:
                    raise RuntimeError(f"region {rid!r} is on frame {offset + idx}, "
                                       f"the stream on {self._index}")
                out[rid] = (masks[0, 0] > 0).cpu().numpy()
                if idx >= state["num_frames"] - 1:
                    del self._regions[rid]
        return out

    def auto_masks(self, frame: np.ndarray, points_per_side: int = 32, min_iou: float = 0.6,
                   min_stability: float = 0.7) -> list:
        """Everything SAM2 finds on one picture by itself (a grid of point prompts):
        bool H×W masks, largest first. Separate from the stream; tracking is unaffected.

        SAM2's own defaults (0.8 / 0.95) keep only the masks it is sure of, and on drawn
        pictures those are the whole panel and small pieces; whole characters score lower,
        so the bars are lower here (measured on comics, 28/09/2026)."""
        key = (points_per_side, min_iou, min_stability)
        if getattr(self, "_auto_key", None) != key:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            self._auto = SAM2AutomaticMaskGenerator(self.predictor, points_per_side=points_per_side,
                                                    pred_iou_thresh=min_iou,
                                                    stability_score_thresh=min_stability)
            self._auto_key = key
        with self._autocast():
            found = self._auto.generate(np.array(frame, copy=True))
        found.sort(key=lambda m: -m["area"])
        return [m["segmentation"].astype(bool) for m in found]

    def forget(self, region_id) -> None:
        self._regions.pop(region_id, None)

    @property
    def live(self) -> int:
        return len(self._regions)
