"""Repaint: the per-frame plan the music gives, and finding the img2img chain in a workflow."""
import json
from pathlib import Path

import pytest

from core.rhythm.midi_reader import MidiNote
from core.video.repaint import Workflow, plan_frames

REF = Path(__file__).resolve().parent.parent / "reference" / "workflow_api_rundiff.json"


def _graph():
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x.safetensors"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "neg", "clip": ["4", 1]}},
        "54": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
        "55": {"class_type": "VAEEncode", "inputs": {"pixels": ["54", 0], "vae": ["4", 2]}},
        "53": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "cfg": 7, "denoise": 0.5,
                                                    "model": ["4", 0], "positive": ["6", 0],
                                                    "negative": ["7", 0], "latent_image": ["55", 0]}},
        "17": {"class_type": "VAEDecode", "inputs": {"samples": ["53", 0], "vae": ["4", 2]}},
        "19": {"class_type": "SaveImage", "inputs": {"filename_prefix": "p", "images": ["17", 0]}},
    }


def test_workflow_finds_the_chain_by_links():
    wf = Workflow(_graph())
    assert (wf.sampler, wf.positive, wf.negative, wf.load, wf.save) == ("53", "6", "7", "54", "19")
    job = wf.job("f.png", "acid", None, 0.66, 42, "out/f", steps=12)
    assert job["54"]["inputs"]["image"] == "f.png"
    assert job["6"]["inputs"]["text"] == "acid"
    assert job["7"]["inputs"]["text"] == "neg"            # None keeps the file's negative
    assert job["53"]["inputs"] | {} == {**job["53"]["inputs"], "denoise": 0.66, "seed": 42, "steps": 12}
    assert wf.graph["53"]["inputs"]["denoise"] == 0.5    # the template is untouched


@pytest.mark.skipif(not REF.is_file(), reason="reference workflow not present")
def test_reference_workflow_loads():
    Workflow.load_file(REF)


def test_workflow_without_loadimage_is_refused():
    g = _graph()
    g["55"]["inputs"]["pixels"] = ["4", 0]
    with pytest.raises(ValueError):
        Workflow(g)


def _kick(t):
    return MidiNote(time=t, pitch=36, velocity=100, channel=9, duration=0.1, track="kick")


def test_plan_kick_raises_denoise_and_decays():
    cfg = {"denoise": 0.3, "kick": {"track": "kick", "denoise": 0.8, "envelope": 0.2},
           "seed": {"base": 10, "per": "bar"}, "prompts": ["a", "b"]}
    plan = plan_frames(2.0, 10, 0.0, lambda t: int(t), cfg, [_kick(0.5)])
    assert len(plan) == 20
    assert plan[4].denoise == 0.3                       # before the hit
    assert plan[5].denoise == pytest.approx(0.8)        # on it
    assert plan[5].denoise > plan[6].denoise > plan[9].denoise > 0.3
    assert [p.seed for p in plan[:10]] == [10] * 10 and plan[10].seed == 11
    assert plan[0].prompt == "a" and plan[10].prompt == "b"


def test_plan_prompts_every_n_bars_and_start_offset():
    cfg = {"denoise": 0.4, "prompts": {"every": 2, "list": ["x", "y"]}}
    plan = plan_frames(1.0, 4, 3.0, lambda t: int(t), cfg)
    assert plan[0].t_song == 3.0 and plan[0].bar == 3
    assert [p.prompt for p in plan] == ["y"] * 4        # bar 3 → (3 // 2) % 2 = 1
    assert all(p.denoise == 0.4 for p in plan)


EDIT = REF.parent / "workflow_api_edit.json"


@pytest.mark.skipif(not EDIT.is_file(), reason="edit workflow not present")
def test_edit_workflow_prompt_and_seed_without_denoise():
    wf = Workflow.load_file(EDIT)
    assert wf.sampler is None and not wf.denoise
    assert not any(v["class_type"] == "Image Comparer (rgthree)" for v in wf.graph.values())
    job = wf.job("f.png", "make it claymation", "ignored", 0.5, 99, "out/f")
    assert job[wf.load]["inputs"]["image"] == "f.png"
    assert job[wf.positive]["inputs"]["text"] == "make it claymation"
    assert job[wf.seed[0]]["inputs"][wf.seed[1]] == 99
