"""Repaint — every frame of an ARC render through a ComfyUI img2img (denoise) workflow.

The render stays the drawing underneath; the diffusion model paints over it. The music
drives the brush: the kick pushes the denoise up (the picture melts on the hit and comes
back), the prompt changes every few bars, and the seed is held for a bar so the painting
does not boil more than the music asks. At 12 fps (``fps: 12``) the repainted frames look
like hand-made animation, each drawing held for two or three video frames.

The workflow is a ComfyUI API graph, either img2img (LoadImage → VAEEncode → KSampler with
denoise < 1 → VAEDecode → SaveImage) or an edit model that takes the frame as a reference and
a prompt saying what to change (Flux 2: "make it claymation"); see :class:`Workflow`. The nodes
are found by their types and links, so the ids in the file do not matter.
"""
from __future__ import annotations

import copy
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from core.rhythm.midi_reader import MidiNote, select_notes
from core.video.h3_client import ComfyH3Client as ComfyClient


@dataclass
class FramePlan:
    index: int        # output frame number
    t_video: float    # seconds into the source video
    t_song: float     # seconds into the song
    bar: int          # the song's bar (0-based)
    denoise: float
    seed: int
    prompt: str


# ── the workflow ─────────────────────────────────────────────────────────────

class Workflow:
    """A ComfyUI API graph with the frame's way through it located.

    Two shapes are known: the img2img one (a KSampler with denoise < 1 over VAEEncode(LoadImage),
    where the denoise is the brush), and the edit one (a Flux-2-style SamplerCustomAdvanced
    guided by a text prompt with the frame as a ReferenceLatent: no denoise, the prompt says
    what to change). Preview-only nodes (image comparers, previews) are dropped: they only
    matter in the UI.
    """

    UI_ONLY = ("PreviewImage", "Image Comparer (rgthree)")

    def __init__(self, graph: Dict[str, dict]):
        self.graph = {k: v for k, v in graph.items() if v.get("class_type") not in self.UI_ONLY}
        graph = self.graph
        loads = [k for k, v in graph.items() if v.get("class_type") == "LoadImage"]
        if len(loads) != 1:
            raise ValueError(f"the workflow needs exactly one LoadImage, found {len(loads)}")
        self.load = loads[0]
        saves = [k for k, v in graph.items() if v.get("class_type") == "SaveImage"]
        if not saves:
            raise ValueError("the workflow needs a SaveImage node")
        self.save = saves[0]

        ks = [k for k, v in graph.items() if v.get("class_type") in ("KSampler", "KSamplerAdvanced")]
        if len(ks) == 1:                                   # img2img
            self.sampler = ks[0]
            inputs = graph[self.sampler]["inputs"]
            self.positive = self._text_node(inputs["positive"][0])
            self.negative = self._text_node(inputs["negative"][0])
            encode = inputs["latent_image"][0]
            if graph[encode]["inputs"].get("pixels", [None])[0] != self.load:
                raise ValueError("the KSampler's latent must come from VAEEncode(LoadImage)")
            self.seed = (self.sampler, "seed")
            self.denoise = True
            return
        # edit: the prompt is the one text encoder that is not zeroed out into the negative
        texts = [k for k, v in graph.items() if v.get("class_type") == "CLIPTextEncode"]
        if len(texts) != 1:
            raise ValueError(f"no KSampler and {len(texts)} text encoders: cannot tell the prompt")
        self.sampler, self.positive, self.negative, self.denoise = None, texts[0], None, False
        noise = [k for k, v in graph.items() if v.get("class_type") == "RandomNoise"]
        self.seed = None
        if noise:
            s = graph[noise[0]]["inputs"]["noise_seed"]
            self.seed = (s[0], "seed") if isinstance(s, list) else (noise[0], "noise_seed")

    def _text_node(self, node: str) -> str:
        if "text" not in self.graph[node].get("inputs", {}):
            raise ValueError(f"node {node} feeding the KSampler has no 'text' input")
        return node

    @classmethod
    def load_file(cls, path: str | Path) -> "Workflow":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def job(self, image: str, prompt: str, negative: Optional[str], denoise: float, seed: int,
            prefix: str, steps: Optional[int] = None, cfg: Optional[float] = None) -> Dict[str, dict]:
        """The graph for one frame."""
        g = copy.deepcopy(self.graph)
        g[self.load]["inputs"]["image"] = image
        g[self.positive]["inputs"]["text"] = prompt
        if negative is not None and self.negative:
            g[self.negative]["inputs"]["text"] = negative
        if self.seed:
            g[self.seed[0]]["inputs"][self.seed[1]] = int(seed)
        if self.sampler:
            s = g[self.sampler]["inputs"]
            s["denoise"] = float(denoise)
            if steps is not None:
                s["steps"] = int(steps)
            if cfg is not None:
                s["cfg"] = float(cfg)
        g[self.save]["inputs"]["filename_prefix"] = prefix
        return g


# ── the plan: what each frame gets ───────────────────────────────────────────

def plan_frames(duration: float, fps: float, start: float, bar_of: Callable[[float], int],
                cfg: dict, notes: Sequence[MidiNote] = ()) -> List[FramePlan]:
    """One entry per output frame: the time it shows, and the denoise, seed and prompt the
    music gives it.

    ``cfg`` is the scene's ``repaint:`` block::

        denoise: 0.35                                        # between hits
        kick: {track: [kick, kick-2], denoise: 0.7, envelope: 0.3}
        seed: {base: 7, per: bar}                            # or per: frame / a number
        prompts: {every: 2, list: ["...", "..."]}            # or a plain list, one per bar
    """
    base = float(cfg.get("denoise", 0.35))
    kick = cfg.get("kick")
    hits = np.array([], dtype=float)
    if kick:
        spec = {k: v for k, v in kick.items() if k in ("track", "notes")}
        hits = np.array(sorted({round(n.time, 4) for n in select_notes(notes, spec)}), dtype=float)
    peak = float(kick.get("denoise", 0.7)) if kick else base
    env_s = float(kick.get("envelope", 0.3)) if kick else 1.0

    seed_cfg = cfg.get("seed", {"base": 0, "per": "bar"})
    if not isinstance(seed_cfg, dict):
        seed_cfg = {"base": int(seed_cfg), "per": "song"}
    seed0, seed_per = int(seed_cfg.get("base", 0)), seed_cfg.get("per", "bar")

    prompts = cfg.get("prompts") or [""]
    if isinstance(prompts, dict):
        every, plist = int(prompts.get("every", 1)), list(prompts["list"])
    else:
        every, plist = 1, list(prompts)

    out = []
    for k in range(int(math.floor(duration * fps + 1e-6))):
        tv = k / fps
        ts = start + tv
        bar = int(bar_of(ts))
        d = base
        i = int(np.searchsorted(hits, ts + 1e-4, side="right")) - 1
        if i >= 0:
            d = base + (peak - base) * math.exp(-(ts - hits[i]) / env_s)
        seed = seed0 + (bar if seed_per == "bar" else k if seed_per == "frame" else 0)
        out.append(FramePlan(k, round(tv, 4), round(ts, 4), bar, round(float(np.clip(d, 0.0, 1.0)), 4),
                             seed, plist[(bar // every) % len(plist)]))
    return out


# ── running it ───────────────────────────────────────────────────────────────

JOB_MARK = "arc_repaint/"     # every job we queue saves under this prefix


def queue_status(client: ComfyClient) -> dict:
    """Who is in ComfyUI's queue: {running, pending, ours} (ours = jobs a repaint queued)."""
    q = client._get("/queue")
    items = list(q.get("queue_running", [])) + list(q.get("queue_pending", []))
    ours = [it for it in items if JOB_MARK in json.dumps(it)]
    return {"running": len(q.get("queue_running", [])), "pending": len(q.get("queue_pending", [])),
            "ours": len(ours)}


def cancel_jobs(client: ComfyClient) -> int:
    """Take our repaint jobs out of ComfyUI's queue (and stop ours if it is the one running);
    other people's jobs are left alone. Returns how many were ours."""
    q = client._get("/queue")
    pending = [it[1] for it in q.get("queue_pending", []) if JOB_MARK in json.dumps(it)]
    if pending:
        client._post("/queue", {"delete": pending})
    running = [it for it in q.get("queue_running", []) if JOB_MARK in json.dumps(it)]
    if running:
        client._post("/interrupt", {})
    return len(pending) + len(running)


def _read_frames(video: Path, wanted: Sequence[float]):
    """The source frames nearest to each wanted time (seconds), keyed by source frame index;
    also the source's fps and frame count."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = sorted({min(int(round(t * fps)), count - 1) for t in wanted})
    out, i, want = {}, 0, set(idx)
    while want:
        ok, img = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = img
            want.discard(i)
        i += 1
    cap.release()
    return out, fps, count


def repaint(source: str | Path, output: str | Path, workflow: Workflow, plan: List[FramePlan],
            cfg: dict, client: Optional[ComfyClient] = None, work: Optional[Path] = None,
            in_flight: int = 6, log=print) -> Path:
    """Send every planned frame through ComfyUI and put the painted frames back together with
    the source's audio. Frames already painted with the same plan entry are reused, so a run
    that stops can pick up where it was."""
    source, output = Path(source), Path(output)
    client = client or ComfyClient(cfg.get("server", "http://127.0.0.1:8188"))
    if not client.is_available():
        raise RuntimeError(f"ComfyUI is not answering at {client.server_url}")
    work = work or output.with_suffix("")
    frames_dir = work / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    plan_file = work / "plan.json"
    before = {}
    if plan_file.is_file():
        before = {p["index"]: p for p in json.loads(plan_file.read_text(encoding="utf-8"))}
    plan_file.write_text(json.dumps([asdict(p) for p in plan], indent=1), encoding="utf-8")

    todo = [p for p in plan
            if not ((frames_dir / f"f{p.index:05d}.png").is_file() and before.get(p.index) == asdict(p))]
    log(f"repaint: {len(plan)} frames, {len(plan) - len(todo)} already painted, {len(todo)} to go")
    frames, src_fps, count = _read_frames(source, [p.t_video for p in todo])
    run = f"arc_{int(time.time())}"
    negative, steps, cfg_scale = cfg.get("negative"), cfg.get("steps"), cfg.get("cfg")

    pending: List[tuple] = []
    t0, done = time.time(), 0

    def collect(entry):
        nonlocal done
        pid, p = entry
        files = client.wait_for_completion(pid, timeout_s=600, poll_interval=0.2)
        client.retrieve_output(files[0], frames_dir / f"f{p.index:05d}.png")
        done += 1
        if done % 10 == 0 or done == len(todo):
            rate = (time.time() - t0) / done
            log(f"  {done}/{len(todo)}  {rate:.2f} s/frame  ~{rate * (len(todo) - done):.0f}s left")

    tmp = work / "upload"
    tmp.mkdir(exist_ok=True)
    for p in todo:
        img = frames[min(int(round(p.t_video * src_fps)), count - 1)]
        f = tmp / f"{run}_{p.index:05d}.png"
        cv2.imwrite(str(f), img)
        name = client.upload_image(f)
        f.unlink()
        job = workflow.job(name, p.prompt, negative, p.denoise, p.seed,
                           prefix=f"{JOB_MARK}{run}/f{p.index:05d}", steps=steps, cfg=cfg_scale)
        pending.append((client.queue_prompt(job), p))
        if len(pending) >= in_flight:
            collect(pending.pop(0))
    while pending:
        collect(pending.pop(0))

    fps = float(cfg.get("fps", 12))
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-framerate", str(fps), "-i", str(frames_dir / "f%05d.png"),
        "-i", str(source), "-map", "0:v", "-map", "1:a?",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",          # an edit model may hand back odd sizes
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(output),
    ], check=True)
    log(f"repaint: wrote {output}")
    return output
