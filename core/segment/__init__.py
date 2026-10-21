"""Segmentation — regions of the picture that effects can hold on to (optional: SAM2).

``RegionTracker`` needs the extras in ``requirements-segment.txt``; ``MaskTrack`` only
reads a mask track folder already written (numpy + OpenCV).
"""
from core.segment.mask_track import MaskTrack, default_folder, write_mask_track
from core.segment.sam2_tracker import RegionTracker, find_checkpoint

__all__ = ["MaskTrack", "RegionTracker", "default_folder", "find_checkpoint", "write_mask_track"]
