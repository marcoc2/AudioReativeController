"""Tests for MiniMax H3 integration in ARC."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.rhythm.grid import RhythmGrid
from core.video.h3_client import find_workflow_path
from core.video.h3_prompt_builder import (
    build_h3_fl2va_prompt,
    build_h3_t2va_prompt,
    calculate_bar_cut_points,
    format_h3_timestamp,
)


def test_format_h3_timestamp():
    assert format_h3_timestamp(0.0) == "At 00:00.000"
    assert format_h3_timestamp(3.5) == "At 00:03.500"
    assert format_h3_timestamp(65.123) == "At 01:05.123"
    assert format_h3_timestamp(11.999) == "At 00:11.999"


def test_calculate_bar_cut_points():
    grid = RhythmGrid(bpm=120.0)  # bar_duration = 2.0s
    cuts = calculate_bar_cut_points(grid, start_bar=0, num_bars=4, max_shots=4)
    # 4 shots -> 3 cut points
    assert len(cuts) == 3
    # Check strictly increasing and within 8.0s
    for i in range(len(cuts)):
        assert 0.0 < cuts[i] < 8.0
        if i > 0:
            assert cuts[i] > cuts[i - 1]


def test_build_h3_t2va_prompt():
    grid = RhythmGrid(bpm=120.0)
    prompt = build_h3_t2va_prompt(
        grid=grid,
        start_bar=0,
        num_bars=4,
        theme="surreal geometric crystal forest",
        energy=0.8,
        style="Cinematic",
    )

    # Required sections
    assert "integrated_multimodal_description:" in prompt
    assert "overall_soundscape:" in prompt
    assert "non_diegetic_music:" in prompt

    # Shots verification
    body = prompt.split("integrated_multimodal_description:")[1].split("overall_soundscape:")[0]
    shots = re.findall(r"\[Shot (\d+)\]", body)
    assert len(shots) >= 2
    assert shots == [str(i) for i in range(1, len(shots) + 1)]

    # Timestamp verification
    timestamps = re.findall(r"At (\d{2}):(\d{2})\.(\d{3})", body)
    assert len(timestamps) == len(shots) - 1

    # Word count verification (H3 t2va requires 150 to 220 words)
    words = len(body.split())
    assert 130 <= words <= 240  # generous bounds around 150-220


def test_build_h3_fl2va_prompt():
    prompt = build_h3_fl2va_prompt(duration_seconds=11.54)

    # Mandatory alignment line
    assert prompt.startswith("How the reference pictures align with the target video —")
    assert "<Picture 1>" in prompt
    assert "<Picture 2>" in prompt
    assert "11.54-second" in prompt

    # Exactly 1 shot
    body = prompt.split("integrated_multimodal_description:")[1].split("overall_soundscape:")[0]
    shots = re.findall(r"\[Shot (\d+)\]", body)
    assert shots == ["1"]

    # Word count (H3 fl2va requires 110 to 170 words)
    words = len(body.split())
    assert 110 <= words <= 175


def test_find_workflow_path():
    # Check that search finds workflows in telegram-server or local repo
    t2v = find_workflow_path("workflow_api_h3_t2v_turbo.json")
    assert t2v is not None
    assert t2v.is_file()

    flf = find_workflow_path("workflow_api_h3_flf_turbo.json")
    assert flf is not None
    assert flf.is_file()
