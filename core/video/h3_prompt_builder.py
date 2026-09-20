"""H3 Prompt Builder — Rhythm-to-Prompt generator for MiniMax H3.

Translates ARC's RhythmGrid, bar divisions, and audio energy into native
MiniMax H3 video prompts with mathematically aligned cut timestamps
(`[Shot N] At MM:SS.mmm`) and energy-responsive camera choreography.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from core.rhythm.grid import RhythmGrid

# Allowed H3 camera motions and modifiers
CAM_HIGH_ENERGY = [
    "The camera rolls clockwise with large amplitude at fast speed",
    "The camera shakes strongly with small amplitude at fast speed",
    "The camera zooms in with large amplitude at fast speed",
    "The camera executes a tracking shot moving left at fast speed",
    "The camera pedestals up with large amplitude at fast speed",
]

CAM_MID_ENERGY = [
    "The camera pushes in with small amplitude at slow speed",
    "The camera executes an arc shot moving right with small amplitude",
    "The camera pans left with small amplitude at slow speed",
    "The camera tilts up with small amplitude at slow speed",
    "The camera pulls out with small amplitude at slow speed",
]

CAM_LOW_ENERGY = [
    "The camera maintains a static shot with subtle drift",
    "The camera pushes in with small amplitude at slow speed",
    "The camera pans right with small amplitude at slow speed",
    "The camera glides forward with small amplitude at slow speed",
]

CUT_PHRASES = [
    "the shot cuts to",
    "the camera cuts to",
    "the shot transitions to",
    "the shot changes to",
]


def format_h3_timestamp(seconds: float) -> str:
    """Format seconds into strict H3 timestamp format: At MM:SS.mmm"""
    total_ms = int(round(seconds * 1000))
    minutes = total_ms // 60000
    remainder_ms = total_ms % 60000
    secs = remainder_ms // 1000
    ms = remainder_ms % 1000
    return f"At {minutes:02d}:{secs:02d}.{ms:03d}"


def calculate_bar_cut_points(grid: RhythmGrid, start_bar: int, num_bars: int, max_shots: int = 4) -> List[float]:
    """Calculate shot cut timestamps (in seconds) aligned to musical bars.
    
    Returns a list of cut times strictly increasing and within the clip window.
    For N shots, returns N-1 cut points.
    """
    total_duration = num_bars * grid.bar_duration
    # Clamp number of shots between 2 and 4 (H3 t2va specification)
    num_shots = max(2, min(max_shots, num_bars))
    
    cuts = []
    # If grid has explicit downbeat timestamps
    if grid.downbeats is not None and len(grid.downbeats) > start_bar + num_bars:
        t0 = float(grid.downbeats[start_bar])
        bars_per_shot = max(1, num_bars // num_shots)
        for i in range(1, num_shots):
            bar_idx = start_bar + i * bars_per_shot
            t_cut = float(grid.downbeats[bar_idx]) - t0
            if 0.5 < t_cut < total_duration - 0.5:
                cuts.append(t_cut)
    else:
        # Mathematical fallback based on BPM bar duration
        bars_per_shot = total_duration / num_shots
        for i in range(1, num_shots):
            t_cut = i * bars_per_shot
            cuts.append(t_cut)
            
    return cuts


def build_h3_t2va_prompt(
    grid: RhythmGrid,
    start_bar: int = 0,
    num_bars: int = 4,
    theme: str = "surreal psychedelic alien ecosystem",
    energy: float = 0.5,
    style: str = "Cinematic",
    dialogue: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Build a complete MiniMax H3 text-to-video (t2va) prompt aligned to music.
    
    Args:
        grid: The project RhythmGrid containing BPM and bar timing.
        start_bar: Starting bar index in the song.
        num_bars: Number of bars the scene spans (recommended 2 to 6).
        theme: Textual subject / visual theme description.
        energy: Audio energy level [0.0 to 1.0] to modulate camera aggressiveness.
        style: Visual style preamble (Cinematic, Live-action, 2D-animated, etc.).
        dialogue: Optional list of (speaker_action, spoken_text) tuples.
        
    Returns:
        Exact 3-section formatted prompt compliant with H3 validation rules.
    """
    cut_times = calculate_bar_cut_points(grid, start_bar, num_bars, max_shots=4)
    num_shots = len(cut_times) + 1
    
    # Select camera motion bank according to musical energy
    if energy >= 0.7:
        cam_bank = CAM_HIGH_ENERGY
        pacing_desc = "dynamic high-contrast kinetic lighting and intense physical motion"
    elif energy <= 0.3:
        cam_bank = CAM_LOW_ENERGY
        pacing_desc = "contemplative atmospheric lighting and serene, fluid movement"
    else:
        cam_bank = CAM_MID_ENERGY
        pacing_desc = "vibrant balanced lighting and rhythmic, coordinated movement"

    # Descriptions for each shot around the theme
    shots_text = []
    
    # Shot 1
    shot1_cam = cam_bank[0 % len(cam_bank)]
    shot1 = (
        f"[Shot 1] {style}, in a {theme} style with {pacing_desc}, "
        f"a wide shot establishes the central environment and primary subjects. "
        f"{shot1_cam} as the atmosphere develops and elements respond in sync."
    )
    shots_text.append(shot1)
    
    # Subsequent shots
    for i, cut_t in enumerate(cut_times, start=2):
        ts = format_h3_timestamp(cut_t)
        cut_phrase = CUT_PHRASES[(i - 2) % len(CUT_PHRASES)]
        cam_move = cam_bank[(i - 1) % len(cam_bank)]
        
        if i == 2:
            shot_desc = (
                f"[{f'Shot {i}'}] {ts}, {cut_phrase} a dynamic medium perspective focusing on core focal details. "
                f"{cam_move} while the surrounding textures pulsate with shifting spectral illumination."
            )
        elif i == 3:
            shot_desc = (
                f"[{f'Shot {i}'}] {ts}, {cut_phrase} an intimate close-up capturing fine refractive textures. "
                f"{cam_move} as micro-movements ripple across the composition with fluid momentum."
            )
        else:
            shot_desc = (
                f"[{f'Shot {i}'}] {ts}, {cut_phrase} an expansive tracking view enveloping the entire scene. "
                f"{cam_move} as all spatial layers harmonize into a climactic visual crescendo."
            )
        shots_text.append(shot_desc)
        
    body = "\n".join(shots_text)
    
    # Soundscape & music sections
    if energy >= 0.7:
        soundscape = (
            "overall_soundscape:\n"
            "Heavy visceral resonant impacts, sharp kinetic swooshes, sub-bass pressure waves, "
            "and electrical hums synchronized with rapid environmental shifts."
        )
        music = (
            "non_diegetic_music:\n"
            "An aggressive, high-tempo electronic score driven by distorted modular synthesizer arpeggios, "
            "pulsating sub-bass drops, and sharp rhythmic punctuation."
        )
    else:
        soundscape = (
            "overall_soundscape:\n"
            "Subtle ambient room resonance, gentle organic rustling, deep ethereal air hum, "
            "and delicate crystalline chimes that decay softly into spatial silence."
        )
        music = (
            "non_diegetic_music:\n"
            "A lush downtempo ambient atmospheric score featuring warm analog synthesizer pads, "
            "gentle acoustic vibrations, and a slow, hypnotic harmonic progression."
        )
        
    return f"integrated_multimodal_description: {body}\n\n{soundscape}\n\n{music}"


def build_h3_fl2va_prompt(
    duration_seconds: float = 11.54,
    transition_style: str = "organic psychedelic light dispersion",
) -> str:
    """Build a generic first-and-last-frame (fl2va) morphing transition prompt.
    
    Connects <Picture 1> and <Picture 2> continuously without cuts or hallucinated objects.
    
    Args:
        duration_seconds: Effective clip duration (e.g. 11.54s for H3 turbo FLF).
        transition_style: Aesthetic nature of the morphing bridge.
        
    Returns:
        Exact FL2VA prompt ready for ComfyUI.
    """
    align_line = (
        f"How the reference pictures align with the target video — "
        f"<Picture 1> (from [Shot 1]) aligns with the 0.00-second mark of the target video; "
        f"<Picture 2> (from [Shot 1]) aligns with the {duration_seconds:.2f}-second mark of the target video."
    )
    
    body = (
        f"integrated_multimodal_description: [Shot 1] In the visual style established by <Picture 1>, "
        f"the scene opens from the composition of <Picture 1>. Subtle chromatic dispersion and liquid ripple "
        f"distortions emerge across the frame as radiant light pulses through existing contours in an {transition_style}. "
        f"The visual elements fluidly melt into an undulating flow of iridescent gradients, kaleidoscopic symmetry, "
        f"and expanding fractal geometry. Shifting optical refractions warp the depth, transforming textures into glowing "
        f"ribbons of liquid light. As the camera pushes in with small amplitude at slow speed, the kaleidoscopic ripples "
        f"gradually decelerate, and the flowing colors begin to coalesce toward the emerging silhouettes and lighting of "
        f"<Picture 2>. The optical distortions progressively smooth out, and luminous particles settle cleanly into the "
        f"exact composition, perspective, and visual arrangement established by <Picture 2> at the end of the shot."
    )
    
    soundscape = (
        "overall_soundscape:\n"
        "Sweeping phase-shifted resonant frequencies, subtle liquid bubbling tones, crystalline shimmer hum, "
        "and a deep sub-bass vibration that harmonizes into clarity as the transition resolves."
    )
    
    music = (
        "non_diegetic_music:\n"
        "An expansive ambient drone featuring rising modular synthesizer pads, shimmering space-echo textures, "
        "and a clean harmonic chord resolution matching the arrival at the final frame."
    )
    
    return f"{align_line}\n\n{body}\n\n{soundscape}\n\n{music}"
