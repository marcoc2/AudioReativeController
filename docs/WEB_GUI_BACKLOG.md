# Backlog: ARC Studio on pywebview (the Arrange as the main window)

The goal: one window, the Arrange, opened straight from a project, replaces both Dear PyGui
windows (`arc_studio.py` and `arc_arrange.py`). From the main window we keep only what is
still alive: the part that shows and edits the chosen object (its parameters, its knobs,
the signals it shares), the project's inputs, the render and the repaint. The rest goes
(section "Dropped").

Started from the sketch of 2026-10-20 (`arc_web.py` + `webui/`), which already opens a
project on the real `core/`, draws the lanes, edits blocks, links knobs and saves round trip.
Inventories of what the two windows do were taken the same day, and the Arrange gained more
that evening, while it was used for the first time on a real render: its own render window,
and editors in the plugin for clip reactivity, triggers, gestures, colours, number lists and
files. That version is the target. Functions are named first; line numbers drift.

Reviewed on 2026-10-20 by Gemini 4 Argon against the code; what held up is folded in below.

**The rule every item keeps: Python is the only source of truth.** The page never works out
the music. Bars, notes, the montage, every knob's value over the song, what a layer is and
what it accepts come from `core/`. Each edit is one API call, answered with the new state.
The page draws, and runs only what is drawing: the shaders it can compile, the clips as
video textures.

Order: first what makes the window usable as the only one (1–6, with 4a and 5 before any
editor), then the Arrange's full behaviour (7–16), then what comes from the main window
(17–23), then the rest.

## Already in the sketch (now `arc_studio_web.py` + `core/web/`)

- Opens a project: `read_midi` with meter and offset, `SceneDoc` + `arrangement.grouped`,
  REAPER cues, the montage (`resolve_song`) in the background.
- Lanes Master / FG / FG FX / BG / BG FX / Clips, with sub-rows (`pack`), whole-song members
  and cued members (read only).
- Moving and resizing blocks, moving a Generator between BG and FG, deleting, "again right
  after", undo/redo (kept in Python), drops from the browser, pins.
- Knobs: the number or the YAML of what moves it, its value over the song from `_knob`.
- Live preview (WebGL2): montage and BG clips as video, one-pass shaders compiled as they are.
- Engine render of the chosen bars (`clip_generator.py`), played in sync (not yet tried).
- Save into the scene itself (item 5).

---

## Foundation

## 1. [x] One app, the Arrange first

**Today.** Two windows: `arc_studio.py` opens a fixed 1110x1020 window (and with no
arguments the dev default `projects/enxame.yaml`, L3741); the Arrange is a second window
opened from a button. The sketch is `arc_web.py --browser` or a pywebview window titled
"web sketch".

**Do.** `arc_studio_web.py` (or `arc_web.py` renamed) opens one pywebview window, "ARC
Studio", sized to the screen, with the Arrange as its only view. Arguments as today:
`--project`, `--audio/--midi/--scene/--stems`, `--skip-separation`, plus `--browser` (the
same page in a browser, for tests). With no project it shows the project picker (item 2). `pywebview` goes into
`requirements.txt`. `arc_studio.py` stays until item 23 is done.

**Done when.** `python arc_studio_web.py --project projects/esfolado_elenco.yaml` opens the
Arrange of that project; closing the window ends the process and any render it started.

**Done (2026-10-20).** `arc_studio_web.py` (the sketch `arc_web.py` is gone): one maximized
window "ARC Studio", `--project` (or the path alone), `--audio/--midi/--scene/--stems`,
`--skip-separation`, `--browser` (with `--port` and, for tests, `--token`). Closing the window
stops the render and the load and ends the process (checked with the real window).
`pywebview>=6.0` is in `requirements.txt`.

## 2. [x] Opening a project, and what was open last

**Today.** "Open Project" is a file dialog (`_pick_project` L1985); nothing remembers what
was open.

**Do.** A start view: recent projects (from `render_output/.arrange.json`, which today only
holds `clips_folder`), the ones in `projects/`, "Open…" (native dialog,
`window.create_file_dialog`) and "New from an audio file…" (item 18). The toolbar keeps a
project menu to switch without restarting.

**Done when.** Reopening the app offers the last project first; switching projects
clears undo, selection, previews and the clip players.

**Done.** The start view (recent first, from `recent_projects` in `.arrange.json`; the ones in
`projects/`; Open… with the native dialog, or a typed path in a browser; New from an audio file…,
which writes `projects/<song>.yaml` and `scenes/<song>.yaml` from the studio's template, with the
song's MIDI when one of the same name is beside it). The toolbar's project button switches
projects (asking first when there are unsaved changes) and clears undo, selection, previews and
the clip players.

## 3. [x] Loading with progress

**Today.** "Load Project" (`_do_load`, a daemon thread) runs, in order: it collects the
project's own stems (the stems rows), `AudioFeatureExtractor` (librosa load, STFT, stem
separation with audio-separator when not cached under `stems_output/`, then the project's
stems added after the AI ones, then features), MIDI (`read_midi` + `shift_in_time`) or,
with no MIDI, rhythm analysis (`core.rhythm.analyzer.analyze`), `ARCPipeline`, trigger
events, the montage dry run (`MetaLibrary` + `resolve_song`, probes every clip), waveforms,
the audio player. The only feedback is one status line; separation and probing can take
minutes with no percentage. `StemService.separate` already takes a `progress_callback`,
but its handlers never call it. The Arrange waits for the load (`preview_state` "loading",
queued previews). The sketch refuses a project without MIDI.

**Do.** The window opens at once and shows the load as steps, each with its state
(waiting / running / done / failed / skipped), a bar where the step can tell how far it
is, its time, and the log under it:

1. the project file, 2. the scene (lanes appear as soon as it is read), 3. MIDI and bars
(the ruler and the MIDI lane appear), 4. the song (decoded in the page for playback and
the waveform), 5. stems (separation or cache), 6. audio features, 7. REAPER cues,
8. the clips (probe) and the montage, 9. clip covers (in the background, never blocking).

Python sends `load.step` events (`{step, state, done, of, message}`). Steps that need it
get a progress callback: the separation handlers call the `progress_callback`
`StemService.separate` already has, `AudioFeatureExtractor` takes an optional
`progress(done, of, message)` and passes it on, `MetaLibrary` reports per probed clip.
A project without MIDI gets its bars from the rhythm analysis, as the studio does, and the
project's stems (`stems:`) go in as prebuilt stems. The lanes are editable from step 3 on; what needs a later step
says so where it shows ("the montage comes when the clips are probed"). A failed step
shows why and offers "Try again"; the others go on when they can. Cancel stops the load.

**Done when.** Opening esfolado_elenco shows the lanes in under 3 s while the montage and
features keep loading, each step ticking; a project whose audio is missing fails at step 4
with the path, and the scene can still be edited.

**Done.** `core/web/load.py`: the nine steps as events, the lanes drawn after step 3 (about
0.4 s for esfolado_elenco), song + stems + features, cues and clips + covers side by side.
`AudioFeatureExtractor(progress=...)` takes `progress(stage, done, of, message)` (stage "stems"
or "features", not the plain `(done, of, message)` written above); the separation handlers tell
their steps; `MetaLibrary(progress=)` per probed clip; covers come from `arr.cover`. No MIDI: the
bars from `analyze_file`. Cancel, and "Try again" on a failed or cancelled step. Tested: a missing
song fails at step 4 and the scene still edits (`test_web_session.py`).

## 4. [x] The API as a module, with tests

**Today.** The sketch's `Api` lives in `arc_web.py`; its checks were one-off scripts.

**Do.** `core/web/` (or `arc_web/`): `server.py` (the local server: token, Range, media
allow-list, events), `session.py` (the project and the arrangement: what `Api` does now),
`load.py` (item 3), `render.py`. `tests/test_web_session.py` drives the session headless
like the sketch's checks did: open a project, every edit, undo/redo, save round trip
(unchanged = byte for byte; an edit changes only its lines), the montage after a pin.
A smoke test opens the page with `--browser` in Playwright: it boots, opens a project, a
drag edits a block.

Every workflow `tests/test_arrange_window.py` pins down today (about 24: flat scene →
groups, drag/move/resize, FX in three places, undo, montage pins, preview state machine,
Explorer drop, long clip question, SAM2 seeds, copy/paste links and FX, the plugin's
editors…) gets its session test as the item that ports it lands.

**Done when.** `pytest tests/test_web_session.py` covers every API call; the Playwright
smoke runs in CI-less local use with one command.

**Done.** `core/web/`: `server.py`, `session.py`, `load.py`, `render.py`, `api.py`.
`tests/test_web_session.py` (12 tests) drives the session headless; `tests/test_web_page.py` is
the Playwright smoke (boots, opens, a drag edits, Ctrl+S saves). Playwright is not installed in
the repo's `.venv`, so it skips there; the same flow was checked by hand in a browser. The ports
of `test_arrange_window.py`'s workflows come with the items that port them.

## 4a. [x] What the sketch got wrong

Found in the review, fixed before building on it:
- `Api._set_spans` gives each block the `file` of the old block at the same index, so
  deleting a block of a member with per-block files shifts them. Blocks carry their file
  (and `from`, speed) through the page and back.
- `edit()` in `app.js` drops an edit made while another is on its way (`if (busy) return`):
  edits go in a queue, one after another.
- The page's knob cache is keyed by the knob's text and survives opening another project,
  whose notes give other values: cleared on open.
- `_same_layer` (the save's patch) can patch the wrong one of two unnamed layers of the
  same source: matched by position among the equal ones, and tested.
- The server's `Range` handling does not refuse a start past the end (answer 416).
- The session's lock covers edits, not every read (`knob`, the montage thread): one lock
  for all of the session's state.
- `renderTimeline` rebuilds every lane on each zoom step: zoom and scroll move what is
  there (CSS transform) and rebuild once the gesture ends.

**Done.** Blocks carry their own `file`, `from` and speed; edits queue in the page; the knob
cache is cleared on open and when the features arrive; the save matches unnamed layers by what
they kept and then by place (`scene_doc.match_items`, tested); `Range` past the end answers 416;
the session has one lock and long work copies under it; zoom stretches what is drawn while the
slider moves and draws again once it stops.

## 5. [x] Save into the scene itself

**Today.** The Arrange saves into the scene (`btn_save` L432: `doc.set_layers`, `_tidy`
for flow style, then `host.reload_editor()`); the sketch writes `<scene>.web.yaml`
beside it, because `SceneDoc.apply_diff` replaces a whole list when it grows or shrinks
(adding one BG layer rewrote every knob of the group in block style), and the sketch's
`_patch` fixes that only for itself.

**Do.** Move `_patch`'s list handling into `core/scene_doc.py` (`apply_diff` keeps the
items a grown or shrunk list still has, patches an edited layer where it is), with tests
in `tests/test_scene_doc.py`, so the Dear PyGui Arrange gets it too; the sketch's own
`_patch` goes. Then Save writes the scene; the toolbar shows "unsaved changes" / "saved at
HH:MM:SS" as `_show_saved` does; undo survives a save (`_rebase`); "Save as…" writes
elsewhere; "Reload" reads the scene from disk again (asking first when there are unsaved
changes) and clears undo, as the Arrange's Reload does. Names read from the scene's
comments (`_comment_names`, `_name_from_comments`) are loaded the same way.

This item comes before every editor in 7–22: they all write through it.

**Done when.** Every scene in `examples/` loads and saves unchanged byte for byte; the
sketch's edit test (move, knob, new layer, removed layer, pin) changes only those lines.

**Done.** `apply_diff` keeps the items of a list that grew, shrank or was reordered
(`match_items`, `_patch_items`, the comment above a list's first item moving with it), and
`_tidy` moved to `scene_doc.tidy`; the Dear PyGui Arrange uses both. Save writes the scene, reads
it again and moves the undo history onto it; "unsaved changes" / "saved at"; Save as… (the window
then edits that file; the project file still names the old one until item 20 saves it); Reload
(asks first). Names from comments, also inside groups, also after a one-line layer; a comment's
name is written as `name:` only when that layer is edited, so an untouched save is the file as it
was. Tested on every scene in `examples/` and `scenes/`.

## 6. [x] Errors and the log

**Today.** Errors go to the status line and the log box (last 30 lines, L3658); the
Arrange's own status line; tracebacks only in the console.

**Do.** One dock: the log (every line, filterable by kind), with an error badge on the
tab when something failed; a toast for what the user just did. A Python exception in an
API call comes back as `{error, where}` (never a hung promise) and its traceback goes to
the log.

**Done when.** A broken knob YAML, a missing clip and a crashed render each say what and
where, without the console.

**Done.** The dock's log keeps every line (2000), filtered by errors / warnings / edits /
engine, with an error badge on its tab; a traceback folds under its error. Every API call is
guarded: an exception answers `{error, where}` (file:line in the repo) and its traceback goes to
the log. A broken knob names its layer and parameter; a clip block whose file is missing is drawn
red and logged at the load; a failed render logs the engine's last lines.

---

## The Arrange, all of it

## 7. [x] Timeline: zoom, scroll, the playhead

**Today.** Zoom −/+/all and Ctrl+wheel around the mouse's bar, "from bar" slider, Alt+wheel
along, row height 20–120 px with −/+ and Shift+wheel keeping the row under the mouse,
the ruler pinned on scroll, a MIDI density lane, rows under 30 px drop their second line.
No playhead (the sketch has one, and plays the song).

**Do.** All of it, plus: wheel scrolls rows, the ruler shows meter changes (the sketch does)
and REAPER markers when the project has an `.rpp`, the playhead follows while playing.
Zoom and scroll stay smooth with a scene of 50+ blocks (item 4a).

**Done (2026-10-21).** The lanes scroll in one area with the ruler and the heads pinned;
zoom −/+/All and Ctrl+wheel around the mouse's bar (stretched while it moves, drawn again when it
stops), Alt+wheel along, rows 20–120 px with −/+ and Shift+wheel keeping the row under the mouse,
heads drop their second line under 30 px. The ruler shows meter changes, how busy the MIDI is
(a strip per bar, from Python) and the REAPER project's markers and regions
(`reaper_project.markers`, from the project's `reaper:` or the scene's cues). The playhead runs on
the ruler and the lanes and the view follows it.

## 8. [x] Placing: pick in the browser, drag on a lane

**Today.** A Generator or clip picked in the browser is placed by dragging across a lane
(`_new_member` L619; `_can_put` decides place or choose bars); a lane has an extra empty
row ("drag here: new"); FX picked go onto FX tracks; `_insert_member` puts a new member on
a sub-row below what it overlaps. The sketch only takes drops (4 bars) and "+".

**Do.** Both ways: drag from the browser, or pick and drag across a lane to choose its
bars. The empty row at the bottom of FG and BG. Placement rules from Python
(`arrangement`): FG clip → `new_foreground` (mask track, backdrop), BG clip → background,
cover, loop.

**Done.** A click in the browser picks (shown under the list); dragging across a lane puts
it on that sub-row (`edits.new_member`: Generators with their clips folder, a BG clip as
background, an FG clip through `new_foreground`, an FX only on an FX track); FG and BG have the
empty row "drag here: new"; drops from the browser still put four bars.

## 9. [x] Moving between rows, and what that means

**Today.** `_move_to_row`: moving a block of a multi-block member splits it into a new
member; crossing BG↔FG recasts a clip (`_recast` L1082: contain and placement, or cover);
FX tracks swap; one undo step. Resizing a clip's start moves its `from` so the picture
keeps its place in time (L1511). Shift+drag on an end resizes (EDGE 8 px). The sketch
moves whole members and forbids clips crossing.

**Do.** The same, in Python (`session.move(ref, k, lane, row, first, last)`), with the
end's resize with Shift held (the cursor and the lit end say it).

**Done.** (Resizing needs Shift, as Marco asked of the Arrange window on 2026-10-20: the end
lights and the cursor turns only with Shift held.) `edits.move`: a block dragged within its row takes its bars from the others
(`arrangement.place`); onto another sub-row or the other lane of its kind it becomes a member of
its own when it had others, a clip is recast for its lane (`arrangement.recast`, moved to core
with the Arrange window's other rules), one undo step. Resizing a start moves its `from`. The ends
resize with Shift held, lit.

## 10. [x] Hidden bars and cued members

**Today.** `mute:` spans drawn dark with "hidden"; menu "Hide in bar N / in the chosen
bars / this cut", "Show again"; inspector "hidden in bars"; cued members show where their
cuts come from (`_draw_cues_info` L2429).

**Do.** All of it. Cued members stay read only, shown in grey with their source.

**Done.** `mute:` spans drawn dark red with "hidden"; the menu's Hide in bar N / in the chosen
bars / this cut and Show again (`edits.mute`); the inspector's "hidden in bars" field; cued
members grey, read only, with where their cuts come from, and a note while they are read.

## 11. [x] The montage row

**Today.** Runs in each clip's colour with names, "none" dark, pins with a corner mark,
the clips layer's `bars` as a strip; a click picks the run and chooses its clip in the
browser; choices "Put <clip> here" (a pin, or a BG block when `from` is not 0), "Nothing
here", "Make <clip> a BG block", "Back to the montage's choice", "Preview"; up to 12 runs
in the inspector.

**Do.** All of it, the runs from `resolve_song` with each run's place in its clip (add
`src_t0` to `Segment`, so the live preview starts the clip where the render will).

Choosing the CLIPS lane opens the montage's own settings, the scene's `video:` section,
with widgets instead of YAML: `clip_per_bar`, `bars_per_clip`, `clip_order`, `clip_end`,
`seed`, gravity, and its `triggers` drawn with the same trigger rows as a layer's clip
reactivity (item 17). They drive the montage, the `{cut: …}` knobs and the FX that work
at the cuts (`transition`, `datamosh`), so the montage row resolves again after each change.

**Done.** Runs from `resolve_song` with each cut's place in its clip (`Segment.src_t0`, and
its direction): the live preview starts the clip where the render does. A click picks the run and
its clip; Put / Nothing here / Make it a BG block / Back to the montage / Preview from
`edits.montage_menu`; up to 12 runs in the inspector. The Clips head opens the montage's settings
(`clip_per_bar`, `bars_per_clip`, `clip_order`, `clip_end`, `seed`), undoable and saved through
`apply_video`. Gravity is a trigger's, so it lives in the triggers; they are YAML text here until
item 17 brings their rows.

## 12. [x] Menus and shortcuts

**Today.** The right-click menu (inventory section 3: Add FX to it ▸ category ▸ entry, Open
plugin, Turn on/off, copy/paste music links, copy/paste FX, hide/show bars, earlier/later
in the chain, Preview its bars, delete this block / the member, Put a Generator here, Add
FX on the lane here / all the song, Add FX to the master). Keys: Ctrl+Z, Ctrl+Y /
Ctrl+Shift+Z, Delete, Space (preview); double-click opens the plugin.

**Do.** Every item, with the catalog submenus by category and their flags (`*` takes input,
`(masks)`, `(cuts)`). Space plays the song (as in the sketch); Shift+Space previews the
chosen bars.

**Done.** The right click asks Python what can be done there (`edits.menu`, `montage_menu`)
and shows the catalog by category with its flags; double-click opens the plugin's knobs; Space
plays, Shift+Space renders the chosen bars, Delete deletes the chosen block (its last: the
member) or empties the montage's chosen bars, Ctrl+Z / Ctrl+Y. "Preview its bars" plays them
live in a loop (the engine's exact preview is item 22).

## 13. [x] Copy and paste: music links and FX

**Today.** `links_of` / `_takes` / `paste_fx` (L1199–1343): copy a member's knobs (not
constants) or FX, paste onto another member that can take them; SAM seeds reset to auto
on another clip.

**Do.** The same calls in Python; the menu says what will be pasted and where it cannot go.

**Done.** `copy_links`, `copy_fx`, `paste_links`, `paste_fx` on the session, with
`links_of` / `takes` / `pasted_fx` in core; the menu says what will be pasted and only offers it
where it fits.

## 14. [x] FX in three places

**Today.** FX on a block (its own `fx:`), on a lane (FX track), on the master (inserted
under trailing `meters`); each FX list has on, bars ("9-12, 30-33", empty = all), plugin or
seeds, earlier/later, remove; the FG's "spill". The sketch shows only "+N fx".

**Do.** The FX of a block listed in its inspector and drawn as a strip on the block; lane
and master FX as today; reorder by drag in the list; "spill" for a block and for the FG.

**Done.** A block's FX listed in its inspector and drawn as a strip on it; the lane's (its head)
and the master's lists; each row: on, bars, earlier/later, remove; drag to reorder a block's or a
lane's list; spill for a block with FX and for the FG.

## 15. [x] Names

**Today.** Names come from `name:`, else from the YAML comment at the end of the line or
the line above (`_comment_names` L261), also inside groups; the inspector renames.

**Do.** The same names in the lanes and the inspector; renaming writes `name:`.

**Done.** The names from `name:` or the comments in the lanes and the inspector; renaming writes
`name:` (empty: back to its file's or its kind's).

## 16. [x] The final video's size

**Today.** A resolution combo in both windows, kept in sync (`set_resolution` L3054,
`RES_CHOICES` L3042); the preview's aspect follows it.

**Do.** One combo in the toolbar, with the choices and a typed `WxH` (even numbers, as the
studio's `res_input` checks), saved in the project's `render.resolution`; the live preview
and the engine preview follow it (the sketch reads it at open only).

**Done.** One chooser in the toolbar with the studio's sizes by shape (`RESOLUTIONS`, now in
`core/web/util.py`, which the studio imports) and "Type a size…" (even numbers); written into the
project's `render.resolution`, the rest of the file as it was; the live preview and the engine's
preview follow it.

---

## From the main window: the object, the project, the render

## 17. [x] The plugin: what the chosen object is and does

**Today.** Two editors of the same thing. The Arrange's plugin window (`_draw_plugin`
L2735: on, blend, mix or opacity first, each catalog parameter with `~` for knobs, a number
or the knob's summary, "Who plays" for trigger parameters, Link/Edit, back to default,
extra keys) and the main window's layer rows (uniforms tree with "about", "params (YAML)"
text from `doc.layer_text` that keeps comments and `*aliases`). The sketch has a slider
per knob and its YAML.

**Do.** One inspector, beside the lanes, for whatever is chosen (block, member, lane, the
master, a montage run, a span):
- head: name, kind, category, file, live or at the render, cued, the whole song, its own FX;
- where and when: lane, sub-row, bars (each block, first/last; clips also `from` and speed),
  on/off, blend, opacity or mix;
- every parameter from the catalog with its about, default, value or knob, back to default;
- the layer's other keys as YAML text, round trip with its comments and aliases (the
  fallback for anything without a widget);
- for an FG clip: placement x / y / scale (each can follow the music), fit, flip, loop;
  cutout (by a mask track, whole frame, all but the backdrop);
- for a clip block: "use the chosen clip and its 'from'" (BG) / "use the browser's 'from'"
  (FG), which take them from the browser's clip and player.

And every editor the Arrange's plugin grew (it decides by the parameter, `_draw_plugin`),
each a widget instead of YAML:
- **clip reactivity** (`_draw_reactivity`, the `clip_reactivity` section): when a clip
  changes, in what order, and its triggers;
- **triggers** (`_draw_trigger`, `_add_trigger`, `_rename_trigger`, `_drop_trigger`): name,
  who plays it (`_who`, through the knob editor's trigger mode), what it does, softer hits
  left out, until another comes in, not near another's hits, gravity;
- **gestures** (`_draw_gestures`, `_gestures_of`: the layer class's `GESTURES`, for eyes,
  mouths, chladni, flame…): each `hits:` row, who plays it, which gesture, how long;
- **feature parameters** (`_is_from`, `_link_from`: a `…_from` names what in the music it
  reads), **"Chords from"** (where `harmony` reads its chords);
- **colours** (`_is_colour`), **flags** (`_is_flag`), **number lists** (`_is_numbers`),
  **files and folders** (`_picks`, `_pick_param`: the native dialog), any other value as
  one line of text (`_text_cell`, empty = back to the default);
- **`mask:`** on any layer (keep it in or out of the figures of a mask track) and the
  **cast**'s roles and acts (who plays which figure, punch, shake, tint… wake), which only
  YAML reaches today (docs/GUI_BACKLOG.md item 10).

**Done (2026-10-21).** `core/web/plugin.py` decides each parameter's editor as `_draw_plugin`
does (number, flag, colour, number list, knob with Link / Edit / Who plays / Chords from, a text
line with its file or folder dialog, back to default) and writes it (`set_value`, `set_text`,
one undo step each; editing inside a signal gives that place its own copy). Its sections: clip
reactivity with its triggers (who plays, actions, softer hits left out, until, not near, gravity;
renaming and dropping a trigger fix the others' `until` / `exclude`), gestures (`hits:` rows; the
Chladni and the Flame now carry their `GESTURES` as class attributes, so both windows show them as
rows), the cast's roles (who, act, envelope, amount, colour, plays, the other keys as YAML), and
`mask:` on any layer. An FX of a block opens its own plugin (⚙ in its list). Integers are written
as integers. Not done: editing the layer's whole YAML text in place (it is shown, read only).

## 18. [x] The knob editor

**Today.** Only in the main window (`_open_knob_editor` L1195 and `core/knob_forms.py`):
the form by what in the music (19 forms grouped by meaning), its source by kind (MIDI
track and notes, audio stem, transcription, features), the form's own settings, shaping
(bias, gain), `rate` with a nested `from`; tie it to a signal or name it as a new one;
"Draw curve" over the song with bars and cuts; Apply (the Arrange gets it through
`PluginTarget.apply`).

**Do.** The same editor in the inspector, opened from a knob's Link/Edit, in three steps:

- **18a. The forms.** Widgets from `knob_forms.FORMS` (sent by Python, not copied into JS)
  for its three modes, knob, trigger ("who plays") and score; sources from the project
  (track names, stems from `stems:` and the separation, `_feature_names`); shaping; the
  YAML it will write. Apply is one undoable edit.
- **18b. Nested sources.** `rate` with its `from` (a knob inside a knob), edited in place.
- **18c. The curve.** `knob_forms.sample` over the visible bars, with the bars, the cuts and
  the playhead, redrawn as the form changes.

**Done.** `core/web/knobs.py`: the page draws `knob_forms.FORMS` as Python sends it, for the
three modes (knob, who plays, chords from); sources from the project (MIDI tracks, the features
with the stems the load found, the transcription); shaping; `rate` with its nested `from` drawn
in place (18b); the YAML it writes and its curve over the bars in view, 8 bars from the playhead
or the song, with the bars and the cuts (18c, `knob_forms.sample`); Apply is one undo step. Who
plays keeps what a trigger does (its actions, envelope, gesture).

## 19. [x] Signals (the scene's anchors)

**Today.** The main window's signals panel (`_draw_signals` L1588): each `&name`, its kind,
value, where it is defined and who uses it, "edit"; the Arrange keeps aliases when a
plugin edits (`set_param` with links).

**Do.** A Signals tab beside the browser's: the list, and choosing one lights every block
that uses it. Editing a signal changes every layer that follows it (as `_replace_ids`
does), and the knob editor offers "use a signal" / "name this one".

**Done.** A Signals tab beside the browser's: each `&name`, its kind, value, where it is defined
and who follows it (in the scene's stack order, as the file writes them); choosing one lights
every block that uses it; Edit opens the knob editor where it is defined, and the places that
follow it follow the new value (undo keeps the old one). The knob editor offers "use a signal"
and "name this one"; a trigger's or a score's source can follow a signal.

## 20. [x] The project's inputs

**Today.** Main window only: audio, MIDI, MIDI offset, meter ("22:6/4,27:5/4"), scene
(and "New…" from a template, `btn_new_scene` L2998), stems rows (name, file) and "skip AI
separation"; Save Project always writes `projects/<audio stem>.yaml` (`_project_dict`
L1827).

**Do.** A "Project" panel with the same fields (native file dialogs), saved into the
project file that was opened, every key it had kept (`sync_offset_ms`, `clips.cache_size`,
`overrides`, `repaint:`…, even the ones no widget shows); changing audio, MIDI, offset,
meter or stems reloads only the steps that depend on it (item 3). "New scene…" from the
template (`btn_new_scene`); "New from an audio file…" makes a project with a new scene.
`sync_offset_ms` (the studio's −500..+500 ms between picture and sound) shifts the live
preview against the song.

**Done.** A Project tab in the dock (audio, MIDI, its offset, meter changes, scene and New
scene…, the project's stems, skip the separation, the picture against the sound), written into
the project file round trip (`core/web/project.py`); what changed says what loads again (the
scene, the bars, the song, the features). `sync_offset_ms` shifts the live preview.

## 21. [x] Render

**Today.** Two ways. The Arrange's "Render video..." window (`open_render`, `btn_render`,
`_render_tick`): the whole song or some bars, where it goes, Render / Stop, the frame it is
on and about how long is left (`_FRAME_LINE` in the studio parses the engine's `fi/n`),
"Open the video" and "Show it in its folder"; it saves the scene first, since the render
reads the file, and runs the studio's renderer (`render_state`, `stop_render`). The main
window's settings: mode, codec x264/nvenc, bars or full song, start (s or bar.beat, or from
the scrubber), FPS, clips folder, order, cache, gravity override, fit, turn wide, stems,
profile, masks; output `render_output/<song>_<scene>[_from<N>s].mp4`; the codec is not
saved in the project.

**Do.** The Arrange's render window as it is (saved first, progress with time left, Stop,
open the video, show its folder), moved to `core/web/render.py` from the studio's renderer
rather than written again, plus the settings above saved in the project's `render:` (codec
too). Before any GPU work it shows the GPU's use and free memory (`nvidia-smi`), so a
render is never started blind on a busy card.

**Done.** "Render video…": the whole song or the chosen bars, where it goes, the GPU's use and
memory, the settings (codec, fps, fit, order, seed, cache, start, clips and masks folders, turn
wide, stems, profile, gravity override) saved in the project's `render:` and `clips:` (the codec
too); it saves the scene first when it has edits; progress with the time left, Stop, open the
video or its folder. One command builder (`render.build_cmd`, the studio's flags) for this and
the subprocess preview.

## 22. [x] Preview through the engine, and the live one

**Today.** The Arrange's preview (`preview_bars` L2011) asks the studio for an in-process
render of at most 8 bars (`_clip_preview_frames`: ClipComposer fast-forwarded,
`build_compositor`, masks), frames kept in RAM, audio-locked, with the state machine
unloaded / loading / busy / ready and queueing. The sketch has a live WebGL preview and an
engine preview through `clip_generator.py`.

**Do.** Keep both, named by what they are: **Live** (instant, what the page can draw, the
rest listed as "at the render") and **Exact** (the engine, the chosen bars, played in sync).
Exact runs in-process when the session already has the library and features (as
`_do_preview` does; frames sent as a short video or JPEG stream to `/media`), else as a
subprocess; it queues while the project loads and is cancelled by the next one, as today.
Live gains: each clip's place from item 11, block FX that are one-pass shaders, FG clips
without a cutout.

Two things Live must not do:
- **fight Python for the GPU**: while the engine renders, separates stems, runs SAM2 or a
  preview, the page stops its WebGL loop and says so (the 4090 is often busy with other
  jobs too);
- **fail silently on a clip it cannot decode**: WebView2 plays H.264/VP9 MP4 and WebM, not
  every `.mov` (ProRes), `.mkv` or `.avi` that `arrangement.VIDEOS` accepts. Such a clip
  gets a small H.264 proxy made by ffmpeg once (in `render_output/.web/proxies/`, by the
  file's hash), and until it exists the montage row and the preview say "making a proxy".

**Done, the exact one not yet run.** Live and Exact. Exact (`core/web/preview.py`) runs in
this process with the session's library and features (the montage fast-forwarded by a dry run,
`build_compositor`, frames piped to ffmpeg), queues while the project loads, is cancelled by the
next one, falls back to the engine's subprocess; at most 8 bars. Live stops drawing while Python
works on the GPU (renders, the exact preview, stem separation) and says so. Clips WebView2 cannot
play (by container and codec, ffprobe) get an H.264 proxy made once in
`render_output/.web/proxies/`; until then the preview says "making a proxy". Not verified: the
exact preview itself (the GPU was busy with 21 of 24 GB in use) and a proxy made for a real
`.mov`.

The characters in Live (2026-10-21): each FG clip on its cues (REAPER's, a cue file's: song s,
clip s, length, rate, hold) or its blocks, placed as `Footage` places it (x, y, scale, which may
follow the music, fit, flip), cut out by a VP9 WebM with alpha that `core/web/cutouts.py` makes
once from the engine's own matte (the mask track, or the SAM2 cache of its `sam_cutout`); the
FG's FX work on them keeping their figures. A SAM2 cutout not made yet shows the whole frame and
says so. The session's knob values now take gain and bias (`_music_knob`), so the meters, the
places and the curves read as the render does.

## 23. [ ] Retire the Dear PyGui windows

**Do.** When 1–22 are done and two things are proven: every workflow of
`tests/test_arrange_window.py` has its session test (item 4), and files dropped from
Explorer arrive with their full paths in WebView2 (item 25). Then `arc_studio.py` and
`arc_arrange.py` (and `arc_window.py`: the Win32 drop hook, its resize cursor and its
native folder dialog, replaced by pywebview's `create_file_dialog`) move to `archive/` per
WORKSPACE_PLAN section 6, their tests with them; README points at the new app.

---

## The rest of the Arrange

## 24. [x] The browser: clips

**Today.** Folder field with "…" and the remembered folder, filter ("final.mp4"), covers
made by ffmpeg in the background (`arr.cover`, up to 400, a generation counter cancels a
stale list), name and length, the clip player (192 px, from `from` to the end, OpenCV
thread), "sound: on/off" (its own audio player, 8 clips cached), the "from (s)" slider,
"Cut it out (SAM2)". Dragging a clip puts its length in bars (images 4 bars); one over
30 s asks how many bars ("Put it" / "The whole video" / Cancel); several clips go one after
another.

**Do.** All of it. Covers from `arr.cover` (Python, cached), not from video elements in the
page. The player is a `<video>` with its sound; the length question is a small dialog.

**Done (2026-10-21).** Any folder (typed, "…" with the native dialog, or dropped from the
Explorer; the last one remembered), the filter, covers and lengths made by Python
(`core/web/clips.py`, a generation counter drops a stale list), the clip player (a `<video>`, its
sound on or off, from "from" to the end, draggable onto a lane), "Cut it out (SAM2)". A clip goes
on a lane as long as it plays (an image 4 bars); one over 30 s asks how many bars (Put it / The
whole video / Cancel); several go one after another (`place_clips`).

## 25. [x] Files from Explorer

**Today.** `arc_window.py` hooks the Win32 window (`WM_DROPFILES`) to get real paths;
dropped on the lanes they become clips from the bar under the mouse; on the browser a
folder opens it, a file is chosen.

**Do.** pywebview's drop events give the full path (`pywebviewFullPath`); the same
behaviour without Win32 code. Check that WebView2 delivers paths for folders too.

**Done.** A drop handler registered from Python (pywebview's DOM events) gets the full paths
(`pywebviewFullPath`, folders too); on a lane the clips go there from the bar and the sub-row
under the mouse, on the browser a folder is listed and a clip is chosen. In a plain browser
(`--browser`) the page says it needs the app window. Checked: the handler registers in the real
window, and a drop was simulated with the paths it gives (a real drag from the Explorer not yet).

## 26. [x] Catalog and demos

**Today.** Generators and FX by category, search, flags, about tooltips; picking one shows
its demo (ready / stale / rendering / missing / failed, "Make demo" / "Make it again",
plays looping, click for full size), its about, needs, file and parameters; FX get "Add to
the master".

**Do.** All of it; demos play as `<video>` from `demos.video_path`, "Make demo" runs
`python -m core.demos render --only KEY --force` with its progress in the log.

**Done (2026-10-21).** A click on an entry in the browser shows it in the inspector
(`core/web/extras.py` `Demos`): its demo looping (click: bigger), its state (ready, stale,
rendering, missing, failed), Make demo / Make it again (the command above, its lines in the log),
its about, what it needs, its file, its parameters (~ the music can move it); FX get "Add to the
master". Categories show their counts and fold. Not checked: making a demo (the GPU was busy).

## 27. [x] SAM2 seeds

**Today.** The seeds window (`open_sam`, `_sam_spec`, `sam_press`, `_sam_draw`): auto
(objects 1–16, whole or parts, reseed every N s) or my seeds (boxes and +/− points per
object, each its colour, right click removes), the frame slider, keep figures / every
object / all but object 1, soft edge, the cache state; OK writes the `sam_cutout` FX,
Cancel removes one just added.

**Do.** In two steps, the spec always written by Python (`_sam_spec`):
- **27a. Auto.** The `sam_cutout` FX's options as an inspector section: objects, whole or
  parts, reseed, keep, soft edge, the cache state.
- **27b. My seeds.** A panel over the preview: the frame comes from Python (OpenCV seek,
  JPEG), the frame slider, objects, boxes and +/− points drawn in the page.

**Done, 27a and 27b in one panel** (as the Arrange window's): auto (objects, whole or parts,
new objects every N s) or my seeds (objects, box / + point / − point on a frame Python reads with
OpenCV, the frame slider, a right click takes a seed away), keep (figures / every object / all
but object 1), soft edge, the cache state; OK writes the `sam_cutout` FX (`sam_apply`), Cancel
takes away one just added. Opened from "Cut out by SAM2…", the FX's ✂ in a list, Seeds… on a
character, the browser's "Cut it out (SAM2)", and Add FX → sam_cutout.

## 28. [x] Repaint (ComfyUI)

**Today.** Main window only: video, start, workflow (`reference/*.json`), scene with
`repaint:`, prompt, FPS, seconds; "Check ComfyUI/GPU" (queue and `nvidia-smi`), Repaint
(`repaint.py`, its lines in the log), Stop (kills it and cancels the jobs).

**Do.** A Repaint panel with the same, its settings saved in the project's `repaint:`;
after a render, its output is offered as the video to repaint.

**Done (2026-10-21).** The Repaint dock tab (`Repaint` in `core/web/extras.py`): the same
fields, saved in `repaint:`, scenes with a top-level `repaint:`, Check (ComfyUI's queue and the
GPU), Repaint with its progress, Stop at any time (kills it and cancels ComfyUI's jobs); a
finished render becomes its video and start. Not run against ComfyUI.

---

## Dropped (not migrated)

Say if any of these should stay.

- The particle modes (`particles`, `particles_v2`, `particles_v3`, `geometry`) and their
  scripts in `generators/`, the pygame preview (`_surface_to_dpg`), the Mode combo.
- The AUDIO CONTROLS and MIDI AUTOMATION tables, and `ARCPipeline` in the GUI (only those
  and the particle preview use it).
- The CLIP DECK and its pin strip (the montage row pins), the DAW timeline with the scrubber
  and the "SAIDA" output lane (the Arrange's montage row and playhead).
- The SCENE — LAYERS stack editor (the lanes and the inspector replace it; the YAML text of
  a layer stays, item 17).
- The SCENE — CLIP TRIGGERS rows editor and the trigger events lane, as they are: the
  scene's `video:` settings and its triggers move to the montage's inspector (item 11),
  with the Arrange's trigger rows (item 17).
- The global gravity and clip-order overrides as GUI widgets (they stay in the project's
  `clips:` and in the Render panel's options).
