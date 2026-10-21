"""A new project from a song's folder: the mix found (the file the others' names start
with), its MIDI, the other audio files as stems named by the person, no AI separation."""
import wave

import yaml

import core.web.api as api_mod


def _wav(path, seconds=1.0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\0\0" * int(8000 * seconds))
    return path


def test_a_song_folder_becomes_a_project(tmp_path, monkeypatch):
    monkeypatch.setattr(api_mod, "ROOT", tmp_path)
    song = tmp_path / "input" / "cerne"
    song.mkdir(parents=True)
    for name in ("cerne_21_10.wav", "cerne_21_10-001.wav", "cerne_21_10-002.wav"):
        _wav(song / name)
    (song / "cerne.mid").write_bytes(b"MThd")
    api = api_mod.Api()
    got = api.scan_song_folder(str(song))
    assert got["mix"].endswith("cerne_21_10.wav") and got["midi"].endswith("cerne.mid")
    assert got["name"] == "cerne" and not got["exists"]
    stems = [a for a in got["audios"] if a["path"] != got["mix"]]
    assert [a["name"] for a in stems] == ["", ""]                 # numbered by REAPER: named here
    spec = {"name": "cerne", "audio": got["mix"], "midi": got["midi"],
            "stems": [{"name": "guitar", "path": stems[0]["path"]}, {"name": "", "path": stems[1]["path"]}]}
    assert "name" in api.new_song_project(spec)["error"]
    spec["stems"][1]["name"] = "piano"
    assert "path" in api.new_song_project(spec)
    proj = yaml.safe_load((tmp_path / "projects" / "cerne.yaml").read_text(encoding="utf-8"))
    assert set(proj["stems"]) == {"guitar", "piano"} and proj["skip_separation"] is True
    assert (tmp_path / "scenes" / "cerne.yaml").is_file()
    assert "exists" in api.new_song_project(spec)["error"]
