"""A REAPER project read back: items, stretch markers, loops, glues, and the clip cues they give."""
import numpy as np
import pytest
import soundfile as sf

from core.reaper_project import Item, clip_cues, exact_pieces, item_spans, parse, read_project

RPP = """<REAPER_PROJECT 0.1 "7.0" 1
  TEMPO 90 4 4
  <TRACK {A}
    NAME voice
    MUTESOLO 0 0 0
    <ITEM
      POSITION 10
      LENGTH 1
      LOOP 1
      SOFFS 2
      PLAYRATE 1 1 0 -1 0 0.0025
      <SOURCE WAVE
        FILE "Media\\vocals-imported.wav"
      >
    >
    <ITEM
      POSITION 12
      LENGTH 2
      LOOP 1
      SOFFS 0.5
      PLAYRATE 1 1 0 -1 0 0.0025
      SM 0 0.5 + 1 1.0 + 2 2.5
      <SOURCE WAVE
        FILE "Media\\vocals-imported.wav"
      >
    >
    <ITEM
      POSITION 20
      LENGTH 0.5
      MUTE 1 0
      <SOURCE WAVE
        FILE "Media\\vocals-imported.wav"
      >
    >
    <ITEM
      POSITION 30
      LENGTH 1.5
      LOOP 1
      SOFFS 0
      <SOURCE WAVE
        FILE "Media\\glued.wav"
      >
    >
  >
  <TRACK {B}
    NAME other
    <ITEM
      POSITION 0
      LENGTH 4
      <SOURCE MIDI
        HASDATA 1 960 QN
        E 0 90 3c 60
      >
    >
  >
>
"""


def test_parse_reads_tracks_items_and_markers(tmp_path):
    (tmp_path / "x.rpp").write_text(RPP, encoding="utf-8")
    proj = read_project(str(tmp_path / "x.rpp"))
    assert proj.tempo == 90 and [t.name for t in proj.tracks] == ["voice", "other"]
    first, stretched, muted, _ = proj.tracks[0].items
    assert (first.position, first.length, first.offset, first.loop) == (10, 1, 2, True)
    assert stretched.markers == [(0, 0.5), (1, 1.0), (2, 2.5)]
    assert muted.muted and proj.tracks[1].items[0].kind == "MIDI"
    assert parse(RPP).child("TRACK").get("NAME") == ["voice"]


def test_stretch_markers_make_straight_pieces():
    it = Item(position=12, length=2, offset=0.5, rate=1, pitch=0, file="f", kind="WAVE",
              markers=[(0, 0.5), (1, 1.0), (2, 2.5)])
    spans = item_spans(it)
    assert [(s.t0, s.t1, s.s0, s.rate) for s in spans] == [(12, 13, 0.5, 0.5), (13, 14, 1.0, 1.5)]


def test_a_rate_and_a_loop():
    it = Item(position=0, length=3, offset=1.5, rate=2, pitch=0, file="f", kind="WAVE")
    spans = item_spans(it, source_length=4.0)                 # 1.5 -> 4 in 1.25 s, then again from 0
    assert [(round(s.t0, 3), round(s.t1, 3), round(s.s0, 3)) for s in spans] == \
        [(0, 1.25, 1.5), (1.25, 3, 0.0)]
    assert all(s.rate == 2 for s in spans)


def test_a_glue_is_found_sample_for_sample():
    rng = np.random.default_rng(1)
    sr = 8000
    src = rng.uniform(-0.5, 0.5, 4 * sr).astype(np.float32)
    glued = np.concatenate([src[sr:2 * sr], np.zeros(sr // 4, np.float32), src[3 * sr:int(3.5 * sr)]])
    got = exact_pieces(glued, {"v": src}, sr, chunk=512)
    assert [(round(a, 3), round(b, 3), round(at, 3)) for a, b, _, at in got] == \
        [(0.0, 1.25, 1.0), (1.25, 1.75, 3.0)]                 # the quiet joins the piece before


def test_clip_cues_follow_the_edit(tmp_path):
    sr = 8000
    rng = np.random.default_rng(2)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not read")                             # only its name and folder matter here
    (tmp_path / "clip").mkdir()
    vocals = rng.uniform(-0.5, 0.5, 6 * sr).astype(np.float32)
    sf.write(tmp_path / "clip" / "vocals.wav", vocals, sr)
    (tmp_path / "Media").mkdir()
    sf.write(tmp_path / "Media" / "vocals-imported.wav", vocals, sr)          # REAPER's copy
    sf.write(tmp_path / "Media" / "glued.wav", np.concatenate([vocals[4 * sr:5 * sr],
                                                             vocals[sr:2 * sr]]), sr)
    (tmp_path / "x.rpp").write_text(RPP, encoding="utf-8")
    cues = clip_cues(str(tmp_path / "x.rpp"), str(clip))
    assert cues == [[10.0, 2.0, 1.0],                         # straight
                    [12.0, 0.5, 1.0, 0.5], [13.0, 1.0, 1.0, 1.5],   # stretched
                    [30.0, 4.0, 1.0], [31.0, 1.0, 0.5]]       # the glue, back to its two pieces
    assert clip_cues(str(tmp_path / "x.rpp"), str(clip), tracks=["other"]) == []


MIDI_RPP = """<REAPER_PROJECT 0.1 "7.0" 1
  TEMPO 120 4 4
  <TRACK {P}
    NAME piano
    <ITEM
      POSITION 10
      LENGTH 3
      LOOP 1
      SOFFS 0.5
      <SOURCE MIDI
        HASDATA 1 960 QN
        E 0 90 3c 64
        E 480 80 3c 00
        E 480 90 40 50
        Em 0 90 43 50
        E 480 80 40 00
        E 480 b0 7b 00
      >
    >
  >
>
"""


def test_midi_items_give_their_notes_and_a_midi_file(tmp_path):
    (tmp_path / "m.rpp").write_text(MIDI_RPP, encoding="utf-8")
    proj = read_project(str(tmp_path / "m.rpp"))
    from core.reaper_project import item_notes, write_midi
    notes = item_notes(proj.tracks[0].items[0], proj.tempo)
    # a 1 s source (4 x 480 ticks at 120 BPM), from 0.5 s in, looping, for 3 s; the muted note left out
    assert [(round(t, 3), p) for t, p, *_ in notes] == [(10.0, 64), (10.5, 60), (11.0, 64), (11.5, 60),
                                                      (12.0, 64), (12.5, 60)]
    out = tmp_path / "piano.mid"
    assert write_midi(str(tmp_path / "m.rpp"), ["piano"], str(out)) == {"piano": 6}
    from core.rhythm.midi_reader import read_midi
    read = read_midi(str(out))
    got = read[1] if isinstance(read, tuple) else read
    assert any(getattr(n, "track", "") == "piano" for n in got)
