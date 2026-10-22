# Audit: the web ARC Studio against the Dear PyGui windows (2026-10-21)

Every button, menu item, shortcut, window and status message of `arc_arrange.py` and of the
parts of `arc_studio.py` the web app replaces (the dropped ones aside: see
docs/WEB_GUI_BACKLOG.md, "Dropped"), checked against `arc_studio_web.py` + `core/web/` +
`webui/`. Three read-only passes (the Arrange, the main window, the catalog demos and the
repaint) listed what was missing or different; this file keeps that list and says what was
done about each. **Fixed** means built and checked (tests, or the page in a browser);
**Kept** means the web app does it differently on purpose, and says why.

## Found missing before the audit

| What | Now |
|---|---|
| Catalog entry: demo, about, needs, file, parameters, Make demo, Add to the master (backlog 26) | Fixed: the inspector shows the picked entry (core/web/extras.py `Demos`) |
| Repaint (ComfyUI) panel (backlog 28) | Fixed: the Repaint dock tab (`Repaint`): settings in `repaint:`, Check, Repaint, Stop, progress; a finished render becomes its video |
| A layer's own YAML, edited as text | Fixed: `layer_text` / `set_layer_text`, the inspector's text with Apply |

## The Arrange (arc_arrange.py)

All fixed (2026-10-21), checked by the tests and on a copy of a project in a browser, but the
exact preview itself (it needs the GPU), which was checked through its events only.

| # | What | Now |
|---|---|---|
| 1-2 | entry panel, demos | Fixed (see above) |
| 3 | category headers | Fixed: counts, a click folds one, each tab says what it holds |
| 4 | two picks | Fixed: a clip or Generator and an FX picked at once (`pickedFx`), each for its tracks |
| 5-6 | the browser's folder | Fixed: the last one listed first; a new cubes' clips_dir is it |
| 7 | "N clips in <folder>" | Fixed |
| 8 | a Generator put from a menu | Fixed: it becomes the pick |
| 9 | the player on a click | Fixed: a clip block or a montage run plays in the player (its from, its mask) |
| 10 | the song mutes the clip | Fixed: playing the song turns the player's sound off |
| 11 | a still image | Fixed: shown as an image (no from, no sound) |
| 12 | the ghost of a dragged clip | Fixed: the bars it would take, and a word in red where it cannot go |
| 13 | rows 30 px | Fixed |
| 14 | the density row | Fixed: 10 px at the ruler's foot, dark to orange |
| 15 | block labels | Fixed: GEN, a generic name with its folder, a long name's end |
| 16 | colours by clip | Fixed: from `arr.colour` (blocks and montage cells); off greyed |
| 17 | the montage's own bars | Fixed: a strip, the bars outside dimmed |
| 18 | every 4 bars | Fixed: a stronger line (lanes and ruler) |
| 19 | the chosen lane lit | Fixed |
| 20-21 | hover texts | Fixed: kind, bars A-B of N, the file, the montage clip's path, from, reversed |
| 22 | choosing chooses bars | Fixed: a block, a cued member (first to last cue), montage bars |
| 23 | Play and Exact | Fixed: "Play" (live, looping) and "Exact" (the engine) everywhere, menus too |
| 24 | a click on an empty lane | Fixed: that bar is chosen |
| 25 | the preview's state on the picture | Fixed: progress, Stop, "stopped at frame d/t", the failure |
| 26 | a preview during a final render | Fixed: it waits for it |
| 27 | a cued member's preview | Fixed: first to last cue |
| 28 | x / y / scale linked | Fixed: the knob editor, the link said in words |
| 29 | the SAM2 line | Fixed: auto or N seeded objects, cached or at the next render |
| 30 | Add FX → sam_cutout | Fixed: from any list it opens its seeds |
| 31 | the plugin's head | Fixed: kind, category, the "~" hint; a sub-FX's on and blend |
| 32 | reactivity and triggers | Fixed: what each does, said |
| 33 | file dialogs | Fixed: they start at the value (`pick_file(..., at)`) |
| 34 | a deleted member | Fixed: its lane is chosen |
| 35 | the montage bars' runs | Fixed: their folders, a click plays one, short buttons |
| 36 | the montage's menu | Fixed: Play, Exact, Add FX to the master |
| 37 | a lane's name: its menu | Fixed: a right click on its head |
| 38 | the status line | Fixed: under the lanes, what the last edit did and the undo steps |
| 39 | "Saved!" / the error | Fixed |
| 40 | the flat-scene notice | Fixed: said once, in the log and the status line |
| 41 | the render's bars | Fixed: first and last bar fields, the size line |
| 42 | the render's progress | Fixed: "Rendering N%" on the toolbar, "starting…", "stopped at frame d/t", the % |

## The main window (arc_studio.py)

| # | What | Now |
|---|---|---|
| 1 | repaint | Fixed (Stop at any time, the video dialog, scenes with `repaint:`) |
| 2 | Save the project as it is | Fixed (one from the command line goes to projects/<song>.yaml) |
| 3-4 | stem and MIDI dialogs | Fixed |
| 5 | the default scene | Fixed: scenes/default.yaml |
| 6 | the command line | Fixed: `--stems` and `--skip-separation` also over a `--project` |
| 7 | the song line | Fixed: the timeline's corner (meter, BPM, the bar's length, the audio's) |
| 8-9 | status and log | Fixed: the status line; the log goes to the terminal too |
| 10-11 | N bars from a start, bar.beat, the playhead | Fixed: the render dialog's "from a start" |
| 12-13 | fps 24, a new fps at once | Fixed (the bars and the features again) |
| 14 | clips need their folder | Fixed: the dialog says why it cannot render |
| 15 | clips and masks dialogs | Fixed, with what needs masks |
| 16 | a seed drawn | Fixed (written with the project) |
| 17 | "whole song" from 0 | Fixed |
| 18 | gravity override, pins | Fixed: the montage and the preview share `video_cfg()` |
| 19 | the preview's bar count | Fixed: 1-8 beside Exact (remembered) |
| 20 | the knob curve | Fixed: 16 s around the playhead by default, its own min / max, a time axis |
| 21 | the score's default | Fixed: the song's sheetsage folder is written |
| 22 | a score source and signals | Fixed: "Use a signal" lists the score signals |
| 23 | the MIDI tracks before the load | Kept: they come with the bars step, under a second |
| 24-25 | a signal's name and notes | Fixed: a signal's name shown, not editable; what a signal is, said |
| 26 | signals in stack order, Refresh | Fixed |

Checked on the GPU (2026-10-21, on a copy of coracao_esfolado): the exact preview of 1, 2 and
4 bars (160 frames in about 7 s), and Make it again for chladni's demo (about 22 s). That run
found and fixed: the preview was 360x640 for a 480x768 project (now the project's own shape,
400x640), and a file made again under the same name kept its URL (now versioned).

Not checked anywhere: a repaint against ComfyUI, a real drag from the Explorer, a proxy made
from a real `.mov`.

## Marco's review (2026-10-21)

| What | Now |
|---|---|
| The instant preview did not obey the timeline | It was a WebGL sketch that left out every layer without a live version. **Live is now the engine itself** (`core/web/stream.py`): the render's own stack makes the frames a few seconds ahead of the playhead and the page shows them; when the engine is behind, the song waits for it. The WebGL one stays as **Sketch**, said to be approximate. The engine's config was checked against the old studio's for every project: the same (the render and the exact preview never differed) |
| Each request to the local server took ~250 ms | Keep-alive and no Nagle: 3 ms (the stream reaches the engine's speed, ~32 fps on coracao_esfolado) |
| The demo of an FX or Generator was across the window | Under the browser's list, where it is picked |
| BG clips showed no picture on the timeline | Their covers, bar after bar, as the montage's cells |
| Clips in the Clips lane could not be resized | Shift+drag a clip's end: all its bars pinned to it, the bars it gives up to the clip next to them |
| A link's editor (and a block's own FX) took the plugin's place: the FX being linked went out of sight | They open in their own area, under the plugin and right of the Arrange (the Arrange narrower). The plugin stays, frozen, the row they came from lit, until Apply / Cancel (a link) or Done (an FX); a link opened from that FX goes over it, Cancel goes back. Choosing another block waits for them too. A block's new FX opens there at once |
| The link's area was small, everything a little too big, some selects cut | The right column (the plugin, the link · FX area) is wider by dragging its left edge (double-click: 340 px again), kept for the next time. Either panel goes into a window of its own (⧉ in its head) and comes back (⇲, the toolbar's chip, or closing that window); the others take its place meanwhile. In pywebview the window is a plain window of the app (no about:blank address bar), whose WebView2 is handed to the page as its popup (arc_studio_web.own_popups), so the page's script drives the panel there; closing it by its X brings the panel back. Look in the toolbar: Classic (as it was) or Fine (denser: smaller type, controls and paddings, rows of 24 px), kept for the next time. A select too narrow shows its choice on hover; the knob's long forms are written under their select |
| julia4d_arc moved too abruptly | Its links in cerne were raw audio features (`subbands.bass`), which jump every frame (up to 0.27 of their range on cerne, p95), and the Julia set is chaotic near its border. The audio feature form now takes `attack` / `release` (seconds to rise, to fall; 0 as before, so old scenes keep their video), stepped on a fixed grid so a seek or a preview gets the same value as the render; a new feature link starts at 0.1 / 0.5 (6× calmer on cerne). The shader has `u_reach` (how far the music takes the shape from its rest; 1 as before) and holds `u_mode` in 0..1 and `u_slice` in -0.6..0.6 |
| Choosing a link's numbers against a feature's was too abstract; a change should be seen on the render at once | **The Feature Lab** (docs/FEATURE_LAB_PLAN.md) is opened with the link editor's Lab button and takes the timeline's place.<br>**Lanes:** the song, what the feature listens to (a stem's file, or the mix through the band; ♪ plays it), the feature, each step of the link (the render's own code), and the value inside the parameter's range.<br>**Setting it:** drag handles to map, Fit, set a range, hold inside (`clip`).<br>**Live:** every change in a link's editor goes to Live before Apply (the draft; Cancel takes it away).<br>**Features:** they are pure now (a stem's or a band's fall no longer depends on how many things read features in a frame) |
| The Lab kept the values in the narrow right area; a stem gave only its level | Each Lab lane's head (300 px) holds its step's settings: the form and what it listens to (the song or any stem), the feature, attack / release / window, gain / bias / clip / range / Fit. Apply / Cancel are on the Lab's bar. Any feature of any stem: `audio:` in the link (core/feature_bank.py, the same analysis as the song's, in the render too) |
| A new scene could not start from nothing (a Generator in bars 1-5 of cerne: "the scene uses clips") | **A scene starts empty and a bar with nothing in it is black.** The engine puts black under the stack when its first layer comes and goes by bars (or is an FX, or masked), so the first layer obeys its bars too; before, the bottom layer filled the whole song. The clip montage is an optional item at the BG's bottom (the lane "Montage", shown only when the scene has one): added from the BG's menu or inspector (the clips folder chosen then), its bars set, removed. Without a montage the cuts the transitions and cut knobs read are where the BG's blocks begin (every bar when there are none). Of the 102 scenes in scenes/ and examples/, only examples/esfolado.yaml changes: its montage's `bars: [[2, 21], [36, 51]]` were ignored, and now hold (its own comment says the video leaves in bars 22-35) |

