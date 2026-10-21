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
