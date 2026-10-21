"""The web inspector's editors on the session: the plugin's rows and what each writes, the
knob editor (and signals), the project's inputs, the render's settings and command line."""
import pytest
import yaml

from tests.test_web_session import _member, _session, project  # noqa: F401  (the fixture)


def _row(p, name):
    return next(r for r in p["rows"] if r["name"] == name)


def _new(s, lane, key, first=1, last=4):
    st = s.new_member(lane, key, first, last)
    return st["select"]["ref"]


def test_a_shaders_knobs_are_numbers_that_can_follow_the_music(project):
    s = _session(project)
    bub = _member(s.state(), "the bubbles")["ref"]
    p = s.plugin(bub)
    assert p["kind"] == "Generator" and p["rows"][0]["name"] == "opacity"
    u = _row(p, "u_s")
    assert u["widget"] == "number" and u["keypath"] == ["uniforms", "u_s"] and u["button"]["label"] == "Link"
    s.set_value(bub, ["uniforms", "u_s"], 0.75)
    assert s.get(bub)["uniforms"]["u_s"] == 0.75
    s.set_value(bub, ["uniforms", "u_s"], None)                 # back to its default: uniforms goes
    assert "uniforms" not in s.get(bub)


def test_text_cells_flags_and_mask(project):
    s = _session(project)
    eyes = _new(s, "bg", "eyes")
    p = s.plugin(eyes)
    hits = _row(p, "hits")
    assert hits["widget"] == "gestures" and "blink" in hits["gestures"]
    s.add_gesture(eyes, ["hits"], "blink")
    assert s.plugin(eyes)["rows"][[r["name"] for r in s.plugin(eyes)["rows"]].index("hits")]["items"][0]["gesture"] == "blink"
    s.drop_item(eyes, ["hits"], 0)
    assert "hits" not in s.get(eyes)
    s.set_mask(eyes, {"on": True, "track": "render_output/footage/x_segments", "objects": "1, 3", "where": "outside"})
    assert s.get(eyes)["mask"] == {"track": "render_output/footage/x_segments", "objects": [1, 3], "where": "outside"}
    assert s.plugin(eyes)["mask"]["on"]
    s.set_mask(eyes, {"on": False})
    assert "mask" not in s.get(eyes)
    assert "error" in s.set_text(eyes, ["roles"], "{unclosed: [")


def test_the_chladni_hits_are_gesture_rows_too(project):
    s = _session(project)
    ch = _new(s, "bg", "chladni")
    assert _row(s.plugin(ch), "hits")["widget"] == "gestures"


def test_clip_reactivity_and_its_triggers(project):
    s = _session(project)
    cubes = _new(s, "bg", "cubes")
    r = _row(s.plugin(cubes), "clip_reactivity")
    assert r["widget"] == "reactivity" and r["cfg"]["clip_order"] == "sequential"
    kp = ["clip_reactivity", "triggers"]
    assert s.add_trigger(cubes, kp)["added"] == "kick"
    assert s.add_trigger(cubes, kp)["added"] == "snare"
    s.set_trigger(cubes, kp, "snare", "until", "kick")
    s.set_trigger(cubes, kp, "snare", "exclude", "kick")
    s.set_trigger(cubes, kp, "kick", "gravity", True)
    s.set_trigger(cubes, kp, "kick", "gravity.peak", 4)
    s.set_trigger(cubes, kp, "kick", "actions", ["reverse", "next_clip"])
    s.rename_trigger(cubes, kp, "kick", "bumbo")
    trs = s.get(cubes)["clip_reactivity"]["triggers"]
    assert list(trs) == ["bumbo", "snare"]
    assert trs["snare"]["until"] == "bumbo" and trs["snare"]["exclude"]["trigger"] == "bumbo"
    assert trs["bumbo"]["gravity"]["peak"] == 4.0 and trs["bumbo"]["actions"] == ["next_clip", "reverse"]
    s.drop_trigger(cubes, kp, "bumbo")
    assert "until" not in s.get(cubes)["clip_reactivity"]["triggers"]["snare"]
    s.set_value(cubes, ["clip_reactivity", "clip_order"], "shuffle")
    assert s.get(cubes)["clip_reactivity"]["clip_order"] == "shuffle"


def test_the_casts_roles(project):
    s = _session(project)
    cast = _new(s, "fgfx", "cast")
    r = _row(s.plugin(cast), "roles")
    assert r["widget"] == "roles" and "wake" in r["acts"]
    s.add_role(cast, ["roles"])
    s.set_role(cast, ["roles"], 0, "who", "1, 2")
    s.set_role(cast, ["roles"], 0, "act", "tint")
    s.set_role(cast, ["roles"], 0, "color", "chord")
    s.set_role(cast, ["roles"], 0, "extra", "{shadow: 0.5}")
    role = s.get(cast)["roles"][0]
    assert role["who"] == [1, 2] and role["act"] == "tint" and role["color"] == "chord" and role["shadow"] == 0.5
    assert "error" in s.set_role(cast, ["roles"], 0, "extra", "- a list")


def test_the_knob_editor_writes_a_form_and_its_curve(project):
    s = _session(project)
    bub = _member(s.state(), "the bubbles")["ref"]
    got = s.knob_open(bub, ["uniforms", "u_s"], "knob", 0.4)
    assert got["state"]["form"] == "constant" and got["state"]["main"] == 0.4 and "kick" in got["sources"]["tracks"]
    assert "hits" in got["schema"]["forms"]
    st = s.knob_new("hits", got["state"])
    st["source"]["kind"], st["source"]["track"] = "track", "kick"
    chk = s.knob_check(st, "knob", 0, 4)
    assert chk["yaml"].startswith("{hits: {track: kick") and max(chk["values"]) > 0.9 and len(chk["bars"]) >= 2
    s.knob_apply(bub, ["uniforms", "u_s"], st, "knob", "kicks")
    assert s.get(bub)["uniforms"]["u_s"]["hits"]["track"] == "kick"
    sig = next(x for x in s.signals() if x["name"] == "kicks")
    assert sig["kind"] == "knob" and sig["def"]["ref"] == bub and sig["uses"] == []
    fg = next(m for m in s.state()["lanes"] if m["lane"] == "fg")["ref"]
    s.knob_apply(fg, ["uniforms", "u_s"], st, "knob", "", "kicks")          # follows the signal
    assert s.get(fg)["uniforms"]["u_s"] is s.get(bub)["uniforms"]["u_s"]
    st2 = s.knob_new("hits", st)
    st2["params"]["envelope"] = 0.6
    s.knob_apply(bub, ["uniforms", "u_s"], st2, "knob")                       # the signal, changed where it is defined
    assert s.get(fg)["uniforms"]["u_s"]["envelope"] == 0.6
    s.save()
    text = s.scene.read_text(encoding="utf-8")
    assert "&kicks" in text and "*kicks" in text
    s.step(True)                                                              # undo: before the save's edits
    assert "error" in s.knob_apply(bub, ["uniforms", "u_s"], st, "knob", "1bad")


def test_who_plays_keeps_what_a_trigger_does(project):
    s = _session(project)
    cubes = _new(s, "bg", "cubes")
    kp = ["clip_reactivity", "triggers"]
    s.add_trigger(cubes, kp)
    got = s.knob_open(cubes, kp + ["kick"], "trigger")
    st = got["state"]
    st["source"]["kind"], st["source"]["track"], st["source"]["notes"] = "track", "kick", "36, 38"
    s.knob_apply(cubes, kp + ["kick"], st, "trigger")
    t = s.get(cubes)["clip_reactivity"]["triggers"]["kick"]
    assert t["track"] == "kick" and t["notes"] == [36, 38] and t["actions"] == ["next_clip"]


def test_the_projects_inputs_are_written_and_say_what_to_load_again(project, tmp_path):
    s = _session(project)
    f = s.project_fields()
    assert f["scene"].endswith("scene.yaml")
    got = s.save_project({"midi_offset": 0.25, "meter": "3:6/4,5:4/4", "sync_offset_ms": 900})
    assert got["reload"] == ["bars"] and got["fields"]["sync_offset_ms"] == 500
    data = yaml.safe_load(project.read_text(encoding="utf-8"))
    assert data["midi_offset"] == 0.25 and data["meter"] == "3:6/4,5:4/4" and data["render"]["fps"] == 30
    assert "error" in s.save_project({"meter": "x"})
    assert "error" in s.save_project({"audio": str(tmp_path / "nope.wav")})
    stem = tmp_path / "voice.wav"
    stem.write_bytes(b"x")
    assert s.save_project({"stems": [{"name": "voice", "path": str(stem)}]})["reload"] == ["features"]


def test_the_render_command_and_where_it_goes(project, tmp_path):
    from core.web.project import checked_render
    from core.web.render import build_cmd, output_path
    s = _session(project)
    s.proj["render"].update(codec="nvenc", clip_fit="cover")
    s.proj["clips"] = {"dir": str(tmp_path), "order": "shuffle", "seed": 7, "gravity": {"enabled": True, "peak": 2}}
    out = output_path(s.audio, s.scene, 3, 6)
    cmd = build_cmd(s.proj, s.audio, s.scene, 4.0, 4, "1080x1920", out, final=True)
    joined = " ".join(cmd)
    for part in ("--clips", "--clip-order shuffle", "--seed 7", "--codec nvenc", "--start-time 4.000", "--clip-fit cover",
                 "--gravity-peak 2.0", "--resolution 1080x1920", "--bars 4", "--midi"):
        assert part in joined, part
    assert out.name == "missing_scene_bars3-6.mp4"
    assert checked_render({"codec": "x264", "fps": "25"}) == {"render.codec": "x264", "render.fps": 25}
    with pytest.raises(ValueError):
        checked_render({"codec": "prores"})


def test_a_knobs_values_take_its_gain_and_bias(project):
    s = _session(project)
    got = s.knob("{hits: {track: kick}, envelope: 0.25, gain: 0.06, bias: 0.72}")["values"]
    assert min(got) >= 0.72 - 1e-6 and max(got) <= 0.78 + 1e-6


def test_a_layers_own_yaml_edited_as_text(project):
    s = _session(project)
    river = _member(s.state(), "the river")["ref"]
    text = s.layer_text(river)["text"]
    assert "role: background" in text and "blocks" not in text
    st = s.set_layer_text(river, text.replace("role: background", "role: background\nhold: 0.5"))
    spec = s.get(river)
    assert spec["hold"] == 0.5 and len(spec["blocks"]) == 2
    assert "error" in s.set_layer_text(river, "role: [unclosed")
    assert "error" in s.set_layer_text(river, "hold: 1")


def test_the_text_of_an_fx_added_since_the_scene_was_read(project):
    s = _session(project)
    river = _member(s.state(), "the river")["ref"]
    n = len(s.get(river).get("fx") or [])
    s.add_fx(river, "rgb_split")
    s.add_fx(river, "rgb_split")
    text = s.layer_text([*river, "fx", n + 1])["text"]
    assert "rgb_split" in text
