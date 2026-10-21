# AudioReativeController (ARC)

Advanced AI-driven audio visualization and animation seed generator for video-to-video diffusion workflows.

## Features

- **AI Stem Isolation**: Integrated BS-RoFormer and Demucs models for extracting clean vocals, bass, drums, etc.
- **Rhythmic Precision**: Uses FFT-based Mel-Bass analysis for high-fidelity rhythmic pulses.
- **Animation Seed Generator**: CLI tool to render high-contrast MP4 files optimized for ControlNet/AnimateDiff.
- **Motion Engine**: Modular Pattern logic (ZigZag, Bouncing) with configurable speed, scale, and motion trails.
- **Developer Debugger**: Real-time dashboard to visualize raw audio features, spectrum, and AI stem energy.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/marcoc2/AudioReativeController.git
   cd AudioReativeController
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure `ffmpeg` is installed on your system.

## Usage

### 🚀 Animation Seed Generator (CLI)
Render a high-contrast animation seed for a specific segment of a song:

```bash
python generators/animation_generator.py --file "path/to/audio.mp3" --duration 10 --preset zigzag --trail 10 --speed 0.5 --scale 0.7
```

**Main Arguments:**
- `--preset`: `none` or `zigzag` (adds kinetic movement).
- `--trail`: Adds motion blur/ghosting (try `10`).
- `--speed`: Speed multiplier for the motion.
- `--scale`: Size of the circles relative to the canvas.
- `--mode`: `vocals`, `demucs` (6-stems), or `roformer`.

### 🎬 Clip compositor (main CLI)
Cut a folder of clips to the song's bars, with generative layers and effects driven by
the MIDI (drums, notes, chords) and the audio:

```bash
python clip_generator.py --file song.wav --midi song.mid --midi-offset 0.0     --clips path/to/clips --clip-fit cover --scene examples/esfolado_cast.yaml     --start-time 13.351 --bars 8 --resolution 854x480 --fps 30 --output render_output/out.mp4
```

Scenes are YAML (`examples/`, see `examples/README.md`). Blends and masks run on the GPU
when there is one; the numpy path is the reference and the fallback.

### 🎭 The figures of the video (SAM2 masks)
`segment_video.py` splits a render into its objects with SAM2, once (the mask track). Scenes
then use it through `--masks`:
- **`mask:`** keeps any layer inside or outside the figures;
- **`cast`** gives each instrument a figure to play: punch, shake, tint, glow, presence,
  rings, echo, carry and wake (a temporal trail; it can follow the note or the bar, be
  stamped on the beat grid, and drift).

The three-step command is in the header of each scene (`examples/esfolado_*.yaml`,
`examples/carrossel_*.yaml`); the effects catalogue is `docs/EFFECTS_BACKLOG.md`.

### 🛠️ Developer Debugger (GUI)
Monitor raw data and tune extraction parameters in real-time:

```bash
python visualizer/visualizer_debug.py --file "path/to/audio.mp3"
```

**Hotkeys:**
- `[R]`: Trigger a 5s render of the current file.
- `[S]`: Toggle Temporal Smoothing.
- `[N]`: Toggle Spectral Normalization.
- `[K/L]`: Adjust Contrast.
- `[+/-]`: Change frequency band count.

## Project Structure

- `core/`: Audio DSP and AI separation logic.
- `visualizer/`: Real-time visualization engines.
- `clip_generator.py`: main CLI (clip compositing + generative layers).
- `segment_video.py`: SAM2 mask track of a render (`core/segment/`).
- `core/video/`: compositor, layers, GPU passes (`gpu_compose.py`), cast (`cast.py`), mattes (`matte.py`).
- `examples/`: ready scenes; `docs/`: effects guidelines and backlog.
- `generators/`: legacy standalone generators (particles/geometry).
- `stems_output/`: Cache for AI-separated files.
- `render_output/`: Destination for generated MP4 seeds.
