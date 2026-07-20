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
- `generators/`: legacy standalone generators (particles/geometry).
- `stems_output/`: Cache for AI-separated files.
- `render_output/`: Destination for generated MP4 seeds.
