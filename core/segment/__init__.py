"""Segmentation — regions of the picture that effects can hold on to (optional: SAM2).

Needs the extras in ``requirements-segment.txt``; nothing else in ARC imports this.
"""
from core.segment.sam2_tracker import RegionTracker, find_checkpoint

__all__ = ["RegionTracker", "find_checkpoint"]
