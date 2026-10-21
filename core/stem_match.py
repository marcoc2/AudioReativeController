"""Where each piece of a stem comes from in a clip: the cue list that lip-syncs the clip.

A voice made by chopping a clip's own sound (a sampler: pieces of the singing, moved
around and repeated on the bar grid, maybe transposed) is matched back to the clip, so
a ``footage`` layer (core/video/footage.py) can show, at every moment the stem sounds,
the frames that made that sound:

    cues = find_cues("input/x/voice.mp3", "input/x/clip.mp4")
    cues["cues"]    # [[song s, clip s, seconds], ...] (a 4th value: the rate, when not 1;
                    #  0 holds the frame: a sustained note)
    cues["semitones"]   # how far the stem was transposed from the clip

How: the clip's sound is first transposed like the stem (the shift is found from the two
melodies: the median gap between the notes where their shapes agree). Then every second
of the stem (a window every ``step``) is slid over the clip's log-mel spectrogram, and the
best place kept. Where consecutive windows agree on ``song - clip`` the piece runs at
normal speed: those windows make a cue. Where two cues touch, the cut is put frame by
frame where the stem stops looking like the first and starts looking like the second.
What the stem sounds and no cue covers is reported (``uncovered``): a stretched note, a
reverse, or sound that is not from the clip at all.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

SR = 16000
HOP = 160                      # 10 ms frames
DT = HOP / SR


def load(path: str) -> np.ndarray:
    """Mono audio at ``SR`` (a video's sound too)."""
    import librosa
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return librosa.load(str(path), sr=SR, mono=True)[0]


def mel(y: np.ndarray) -> np.ndarray:
    """Log-mel frames (64 bands × frames), each band's mean removed."""
    import librosa
    m = librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP, n_mels=64, fmin=100, fmax=7000)
    m = np.log1p(m * 1000.0)
    return (m - m.mean(1, keepdims=True)).astype(np.float32)


def slide(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Cosine likeness of ``Q`` (bands × n) to every n-frame window of ``R``."""
    n = Q.shape[1]
    if R.shape[1] < n:
        return np.zeros(0, np.float32)
    win = np.lib.stride_tricks.sliding_window_view(R, n, axis=1)       # bands × K × n
    dots = np.einsum("bn,bkn->k", Q, win)
    energy = np.concatenate([[0.0], np.cumsum((R.astype(np.float64) ** 2).sum(0))])
    norms = np.sqrt(np.maximum(energy[n:] - energy[:-n], 1e-12))
    return (dots / (np.linalg.norm(Q) * norms + 1e-9)).astype(np.float32)


def active_spans(y: np.ndarray, below_peak_db: float = 40.0, gap: float = 0.4,
                 shortest: float = 0.15) -> List[Tuple[float, float]]:
    """Where ``y`` sounds: (start, end) in seconds; quiets shorter than ``gap`` are bridged."""
    import librosa
    rms = librosa.feature.rms(y=y, frame_length=2 * HOP, hop_length=HOP)[0]   # ±10 ms edges
    db = 20 * np.log10(rms + 1e-9)
    on = db > db.max() - below_peak_db
    spans: List[Tuple[float, float]] = []
    start = None
    for i, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = i * DT
        elif not v and start is not None:
            if spans and start - spans[-1][1] < gap:
                spans[-1] = (spans[-1][0], i * DT)
            else:
                spans.append((start, i * DT))
            start = None
    return [(a, b) for a, b in spans if b - a >= shortest]


def _contour(y: np.ndarray) -> np.ndarray:
    """The melody, in MIDI numbers every 20 ms (NaN where unvoiced)."""
    import librosa
    f, voiced, _ = librosa.pyin(y, fmin=70, fmax=900, sr=SR, frame_length=1024, hop_length=320)
    return 12 * np.log2(np.where(voiced, f, np.nan) / 440.0) + 69


def estimate_semitones(stem: np.ndarray, clip: np.ndarray,
                       spans: Sequence[Tuple[float, float]], budget: float = 40.0) -> float:
    """How far the stem's melody sits from the clip's (to the nearest quarter tone), from at
    most ``budget`` seconds of the stem: for each 2-s stretch, the clip stretch whose melody
    has the same shape, and the gap between them; the median of the good ones."""
    parts, used = [], 0.0
    for a, b in spans:
        take = min(b - a, budget - used)
        if take <= 0:
            break
        parts.append(stem[int(a * SR):int((a + take) * SR)])
        used += take
    if not parts:
        return 0.0
    song = _contour(np.concatenate(parts))
    ref = _contour(clip)
    W, step = 100, 50                                    # 2 s windows every 1 s (20 ms frames)
    shifts = []
    for i in range(0, len(song) - W + 1, step):
        q = song[i:i + W]
        if np.mean(~np.isnan(q)) < 0.4:
            continue
        best = (0.0, 0.0)
        for k in range(0, len(ref) - W + 1, 2):
            r = ref[k:k + W]
            both = ~np.isnan(q) & ~np.isnan(r)
            if both.sum() < W * 0.3:
                continue
            d = q[both] - r[both]
            shift = float(np.median(d))
            score = np.mean(np.abs(d - shift) < 0.7) * both.sum() / W
            if score > best[0]:
                best = (score, shift)
        if best[0] >= 0.4:
            shifts.append(best[1])
    if not shifts:
        return 0.0
    return round(float(np.median(shifts)) * 4) / 4


def _frame_likeness(S: np.ndarray, R: np.ndarray, t0: float, t1: float, d: float,
                    half: int = 15) -> np.ndarray:
    """Per stem frame in [t0, t1): cosine of the 0.3 s of stem around it to the clip ``d`` s
    earlier (a single 10 ms frame is too noisy to say which piece it belongs to)."""
    i0, i1 = int(round(t0 / DT)), int(round(t1 / DT))
    shift = int(round(d / DT))
    out = np.full(max(0, i1 - i0), -1.0, np.float32)
    for k, i in enumerate(range(i0, i1)):
        a0, a1 = max(0, i - half), min(S.shape[1], i + half)
        b0, b1 = a0 - shift, a1 - shift
        if b0 < 0 or b1 > R.shape[1] or a1 <= a0:
            continue
        a, b = S[:, a0:a1], R[:, b0:b1]
        a = a - a.mean(1, keepdims=True)
        out[k] = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return out


def _cut_between(S, R, t0: float, t1: float, d_a: float, d_b: float) -> float:
    """Where in [t0, t1] the stem stops being the clip at ``d_a`` and starts being it at ``d_b``."""
    if t1 - t0 < 2 * DT:
        return (t0 + t1) / 2
    a = _frame_likeness(S, R, t0, t1, d_a)
    b = _frame_likeness(S, R, t0, t1, d_b)
    gain = np.concatenate([[0.0], np.cumsum(a - b)])          # a before the cut, b after it
    return t0 + int(np.argmax(gain)) * DT


def find_cues(stem_path: str, clip_path: str, semitones: Optional[float] = None,
              win: float = 1.0, step: float = 0.25, min_score: float = 0.45,
              tolerance: float = 0.06, shortest: float = 0.5, hold_score: float = 0.3) -> dict:
    """The cue list (see the module docstring). ``semitones`` None: found from the melodies.

    A stretch the stem holds (a sustained note: its windows keep landing on about the same
    spot of the clip, within half a second) becomes a still cue, rate 0, on that spot."""
    import librosa
    stem, clip = load(stem_path), load(clip_path)
    spans = active_spans(stem)
    if semitones is None:
        semitones = estimate_semitones(stem, clip, spans)
    if semitones:
        clip = librosa.effects.pitch_shift(clip, sr=SR, n_steps=float(semitones))
    S, R = mel(stem), mel(clip)
    clip_len = len(clip) / SR
    n = int(round(win / DT))
    hits = []                                              # (song t, song - clip, score)
    for a, b in spans:
        t = a
        while t + win <= b + step / 2:
            i = int(round(t / DT))
            Q = S[:, i:i + n]
            if Q.shape[1] == n:
                Q = Q - Q.mean(1, keepdims=True)
                sc = slide(Q, R)
                if len(sc):
                    k = int(np.argmax(sc))
                    hits.append((t, t - k * DT, float(sc[k])))
            t += step
    runs: List[List[Tuple[float, float, float]]] = []
    for h in hits:
        if h[2] < min_score:
            runs.append([])
            continue
        if runs and runs[-1] and abs(h[1] - np.median([x[1] for x in runs[-1]])) <= tolerance \
                and h[0] - runs[-1][-1][0] <= step * 1.5:
            runs[-1].append(h)
        else:
            runs.append([h])
    runs = [r for r in runs if len(r) >= 2]
    cues = []                                              # [start, end, song - clip]
    sure = []                                              # what the windows vouch for: the middle
    homes = []                                             # of the first to that of the last
    for r in sorted(runs):
        sure.append((r[0][0] + win / 2, r[-1][0] + win / 2))
        homes.append(next(((a, b) for a, b in spans if a - step <= r[0][0] <= b), None))
        cues.append([sure[-1][0] - win / 2, sure[-1][1] + win / 2, float(np.median([x[1] for x in r]))])
    for k, c in enumerate(cues):
        home = homes[k]
        if k and sure[k][0] - sure[k - 1][1] <= win:           # a close neighbour: cut where it changes
            lo, hi = sorted((sure[k - 1][1] - win / 4, sure[k][0] + win / 4))
            cut = _cut_between(S, R, lo, hi, cues[k - 1][2], c[2])
            cues[k - 1][1] = c[0] = cut
        elif home is not None and sure[k][0] - home[0] <= win:  # the first: from where the sound starts
            c[0] = home[0]
        if home is not None and home[1] - sure[k][1] <= win and \
                (k + 1 == len(cues) or sure[k + 1][0] > home[1]):
            c[1] = home[1]                                        # the last: to where it stops
    out = []
    for start, end, d in cues:
        start = max(start, d)                              # the clip starts at 0 ...
        end = min(end, d + clip_len)                       # ... and ends at its length
        if end - start >= shortest:
            out.append([round(start, 3), round(start - d, 3), round(end - start, 3)])
    for a, b in _uncovered(spans, out):                   # a held note: the windows stay on one spot
        near = [x for x in hits if a - step <= x[0] and x[0] + win <= b + win / 2 and x[2] >= hold_score]
        where = [x[0] - x[1] for x in near]
        if len(where) >= 2 and np.subtract(*np.percentile(where, [75, 25])) <= 0.5:
            out.append([round(a, 3), round(float(np.median(where)), 3), round(b - a, 3), 0])
    out.sort()
    return {"stem": str(stem_path), "clip": str(clip_path), "semitones": float(semitones),
            "cues": out, "uncovered": _uncovered(spans, out)}


def _uncovered(spans, cues, shortest: float = 0.3) -> List[List[float]]:
    """The parts of ``spans`` no cue covers (longer than ``shortest``)."""
    covered = sorted((c[0], c[0] + c[2]) for c in cues)
    out = []
    for a, b in spans:
        t = a
        for c0, c1 in covered:
            if c1 <= t or c0 >= b:
                continue
            if c0 - t > shortest:
                out.append([round(t, 3), round(c0, 3)])
            t = max(t, c1)
        if b - t > shortest:
            out.append([round(t, 3), round(b, 3)])
    return out
