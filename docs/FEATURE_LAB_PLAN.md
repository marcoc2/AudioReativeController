# Feature Lab: seeing a link before choosing its numbers (plan, 2026-10-21)

## Status (2026-10-21)

Marco's answers: the features made pure (old videos may change, see below), the Lab in the
timeline's place, `clip` for every link, and a change seen on the render at once ("even
low-res"). Built:

- **Phase 0.** `get_features_at_time` smooths `bands`, `stems.*`, `vocal` and the spectrum
  along the frames (`_smoothed`, `_smoothed_spectrum`), so a value no longer depends on who
  asked before.
  - Measured on cerne: with one reader a frame it is the same as before (the spectrum within
    6e-5).
  - With 2–3 readers a frame, the old one fell up to 0.2–0.3 faster. That is what changes in
    the 11 of 104 scenes that read stems or bands, wherever more than one thing reads features
    in a frame.
  - The AI stems' files are kept (`stem_files`), and `core/feature_info.py` says what each
    feature is.
  - `core/web/lab.py`: `knob_stages`, `lab_heard`, `lab_wave`, `param_range`, `knob_draft`.
  - `clip: [lo, hi]` works for every form (`_music_knob`). `Param.range` is parsed from the
    about text (64 of 328 parameters say one).
- **Phase 1.** The Lab (the link editor's **Lab** button) has these lanes: Song, Heard (stem
  or band, ♪ "what it hears"), Feature (its range, the song's p5 / p95, the relative average
  dashed), Shaped, and Parameter.
  - **Parameter lane:** the range it expects (red past it), the layer's bars lit, and the hits
    of a trigger form.
  - **Lab bar:** Map handles (from / to → gain and bias), Fit, a Range the user can set (kept
    in `.arrange.json` `ui.ranges`), and "hold inside" (clip).
  - **Scale:** a lane stops at what is usual and marks a rare peak ("↑ up to 81.7").
  - **Lane height:** the lanes share the timeline's height.
  - **Cost:** about 0.1–0.3 s an update, for the bars in view and half a view on each side.
- **Live draft.** Every change in a link's editor (with or without the Lab) is tried on the
  Live picture before Apply.
  - The stack is built again with it: cerne's julia4d takes about 4 ms to build and 36 ms for
    the first frame.
  - Cancel takes it away. What is saved and the final render never see it.
- **Phase 2, in part.** Any link opens in the Lab. A trigger form shows the MIDI lane and its
  hits; pitch, chord and time forms show the Parameter lane only.

Marco's review of the first Lab (2026-10-21): the values were still set in the narrow area on
the right, and a stem gave only its level. Now:
- **The Lab is where the link is set.** Each lane's head (300 px wide in the Lab) holds the
  settings of the step it draws:
  - Song: what in the music (the form), with a non-feature form's own settings;
  - Heard: listen to (the song, or any stem with a file);
  - Who plays: a trigger form's who plays;
  - Feature: what of it;
  - Shaped: window / attack / release;
  - Parameter: gain, bias, hold inside, its range, Fit.

  Apply and Cancel are on the Lab's bar. The right area only shows what it writes.
- **Source, then feature.** Every source has the same features: its level, the 7 bands, the 7
  texture features, brightness and change.
  - The source is a file: `{feature: texture.noisiness, audio: input/cerne/piano.mp3}`, the
    way a trigger names whose attacks.
  - `core/feature_bank.py` analyses each file once per process (the render's knob reads it
    the same way).
  - `stems.<name>` stays, as "the older way" (the song's analysis of that stem, falling
    slowly).
  - `level` (RMS against its loudest moment) is new for the song too.

Not done:
- The Lab in a window of its own.
- A piano roll for pitch / voice and the chord names for chord / harmony.
- Comparing features.
- A lighter feature read: `get_features_at_time` computes every feature for one value.

Marco: choosing the numbers of a link (attack, release, gain, bias) against a feature's
numbers is too abstract. He wants to *see* it: the song, the audio the feature listens to
(the kick's stem, say), the feature's curve over time, and what the link's settings make of
it, so he understands how the parameter will move before he renders.

The Lab is a way of looking at the link editor that already exists (the sub area: form,
source, settings). It changes nothing in what a link is or writes: the same YAML, the same
engine. What it adds is the picture of each step, on the song's time.

## 1. What it shows

The lanes, top to bottom, all on one time axis (the timeline's own: bars, zoom, scroll,
the playhead):

| Lane | What | Why |
|---|---|---|
| **Song** | the mix's waveform | where we are in the music |
| **Heard** | the audio the feature listens to: a stem's own waveform (`stems.kick`), the mix band-passed to the band's Hz (`subbands.bass` 60-250 Hz), the mix (texture, centroid, flux) | what the number is *about*; the button ♪ plays this instead of the mix (solo), so one hears what it hears |
| **Feature** | the raw value, as the engine reads it, with its range said (0..1, -1..1, Hz) and a strip of its spread (p5, median, p95 over the song or the shown bars) | how it moves, how much, how often it sits where |
| **Shaped** | each step of the link: the glide (attack / release), the form's own step (relative: the recent average, drawn dashed, and the ratio), then gain / bias | what each setting does, seen against the step before it |
| **Parameter** | the final value inside the **parameter's range** (u_slice -0.5..0.5 as a band), what falls outside shown red | the question one actually has: does it stay where the shader wants it? |

Beside the lanes stay the settings (the sub area, as now); every change redraws the lanes
at once. The curves are drawn from the render's own code (§3), never from a re-implementation in
the page.

### Setting by hand on the picture

- **Map**: two handles on the Feature lane (the input range, from p5..p95 by default) and
  two on the Parameter lane (the output range, the parameter's range by default). Moving
  them writes gain and bias (`gain = (out_hi - out_lo) / (in_hi - in_lo)`,
  `bias = out_lo - gain * in_lo`). The YAML stays gain / bias; "Fit" puts the handles back on
  p5..p95 → the parameter's range.
- **Glide** shown as the rise and fall times on a hit of the Shaped lane (the attack's and
  the release's slopes), with the numbers; the sliders stay the way to change them.
- **Loop**: a drag on the ruler picks bars; Play loops them, with the mix or the heard
  audio (♪). The Live picture keeps running beside it (§4 phase 3 makes it show the draft).

## 2. The cases

| Case | Heard | Feature lane | Notes |
|---|---|---|---|
| `stems.<name>` (a stem's loudness) | the stem's file | RMS, `(e/max)^1.3`, 0..1 | the AI stems' files must be kept (§3) |
| `subbands.<band>` (7 bands, Hz) | the mix through a band-pass of that band (WebAudio biquads) | 0..1, linear | the band's Hz written on it |
| `texture.loudness`, `swell`, `percussive`, … | the mix | each its own range (swell -1..1, tremolo_rate in Hz) | from the metadata table (§3) |
| `centroid`, `flux` | the mix | 0..1 | "brightness", "change" said in words |
| `relative` form | as its feature | the feature, its recent average (dashed), the ratio (~1) | the ratio around 1 is why u_slice went out: the Parameter lane shows it |
| a signal (`*name`) | as the signal's own spec | as its form | read-only in the lab, "edit the signal" opens it |
| no stems (skip separation) | — | `stems.*` not offered | "Add a stem file…" (the project's `stems:`), as the dialogs do |
| features still loading | — | "the audio features are loading" | the lanes fill when they come |
| a parameter with no known range | — | — | its range from the about text when it says one ("0..11", "-0.5..0.5"); else the user sets it in the lab and it is kept (`.arrange.json`, per entry and parameter); else none drawn |

Later, the same lanes for the other forms (phase 2): **hits / adsr / before / since / count**
(the trigger's notes as a roll from the MIDI track, or the audio's onsets with the onset
strength curve, then the envelope), **pitch / voice** (a piano roll of the track and the value),
**chord / harmony** (the chord names and the value), **lfo / bar / rate / cut** (the bars and the
value; no Heard lane).

## 3. What the engine needs first (phase 0)

1. **A feature's value must not depend on who asked before.** `get_features_at_time` applies
   its gravity smoothing (instant rise, ×0.9 fall) to `bands`, `stems.*`, `vocal`,
   `bass/mid/high` **on every call**, and every knob of the scene calls it. So a stem's fall
   is faster the more knobs read features in that frame, and the Lab (which asks in another
   order) would draw something else than the render. Fix: smooth those along the frames once
   (precomputed arrays), so the call is pure. **It changes old videos a little** wherever
   `stems.*` or `bands` drive something (the fall becomes the one the code meant, once a
   frame); measured on the projects before doing it, and said.
2. **Keep the stems' files**: the AI separation's paths (stems_output/…) kept on the
   extractor, so the page can get a `media` URL for each (prebuilt ones already have paths).
3. **A table of what each feature is** (`core/feature_info.py`): name, words ("the kick
   stem's loudness", "how bright the sound is"), its range, what it listens to (a stem's
   file, a band's Hz, the mix), from what already exists (SUBBAND_RANGES, texture.py's
   docstring, the stems). The feature select groups and explains by it (by meaning, not by
   module).
4. **The steps of a link** (`knob_stages(spec, t0, t1)`): the same `_music_knob` code, run at
   the render's fps in time order, returning each step's array (raw, glided, the relative
   average and ratio, final), downsampled for the page (min/max per pixel column, so a short
   hit is never lost). One call per change; the raw lane read straight from the arrays.
5. **A parameter's range**: parsed from the about text ("a..b") into `Param.range`, and the
   user's own kept in `.arrange.json` `ui.ranges`.
6. **Waveforms**: the mix's and a stem's min/max at 200 Hz sent from Python (the page does
   not decode a 4-minute stem to draw it); decoded in the page only when ♪ plays it.

## 4. Where it opens, and the phases

**Where**: a "Lab" button in the link editor (a feature or relative link) turns the
timeline's lanes area into the Lab's lanes, the same width, axis and playhead; the
settings stay in the sub area on the right; "Back to the timeline" (or closing the link)
returns it. Nothing else moves (the toolbar, the plugin, the sub area keep their places).
The Lab can also go into a window of its own, like the panels (⧉), for a second monitor.

| Phase | What | Done when |
|---|---|---|
| 0 | §3: pure features, the stems' files kept, the feature table, `knob_stages`, ranges, waveforms | tests: a feature read in any order gives the render's value; every feature has words and a range; the old videos' change measured and reported |
| 1 | the Lab for `feature` and `relative` links: the five lanes, ♪ (stem / band / mix), the spread, Map and Fit, Loop | checked on cerne (the kick stem, u_slice's relative) in the real window |
| 2 | the other forms' lanes (hits from MIDI or onsets, pitch, chord, time) | each form drawn and checked |
| 3 | the draft on Live (the link tried in the picture before Apply, undone on Cancel); compare features (a few drawn faint over the chosen one, "which follows the kick best") | — |

## 5. Decisions for Marco

1. Phase 0.1 changes old videos a little where a stem or `bands` drives something (once
   measured). Do it, or keep the old per-call behaviour and give the Lab its own reading
   (then the Lab is not exactly the render, which [preview-must-be-the-engine] forbids)?
2. The Lab in the timeline's place (recommended: same time axis) or always in its own
   window?
3. A `clip: [lo, hi]` setting for every link (the value held inside the parameter's range,
   so a rare peak cannot throw the shape out), drawn on the Parameter lane: add it in phase 1?
