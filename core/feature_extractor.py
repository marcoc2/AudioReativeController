import numpy as np
import librosa
import sys
import os
import hashlib
import shutil
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from core.stem_service import StemService
except ImportError:
    try:
        from stem_service import StemService
    except ImportError:
        print("[Feature Extractor] Error: StemService not found.")
        StemService = None

from core.texture import TEXTURE_NAMES, texture_features

def _bm_points(num_bands: int):
    """Return (b_point, m_point) that trisect num_bands with no empty slice."""
    b = max(1, num_bands // 4)
    m = max(b + 1, num_bands // 2)
    return b, m

def _safe_mean(arr) -> float:
    return float(np.mean(arr)) if len(arr) > 0 else 0.0


class AudioFeatureExtractor:
    """
    Advanced feature extractor with dynamic AI separation modes.
    """
    def __init__(self, file_path, fps=60, temporal_smoothing=0.7, frequency_smoothing=1.5, separation_mode="demucs",
                 prebuilt_stems=None, skip_separation=False, max_seconds=None, progress=None):
        self.file_path = Path(file_path)             # Absolute path to the audio file
        self.fps = fps                               # Frames per second for the output animation
        self.sample_rate = 0                         # Audio sample rate (e.g., 44100Hz)
        self.duration = 0                            # Total duration of the audio in seconds
        self.y = None                                # Raw audio waveform data
        self.separation_mode = separation_mode       # AI model for stem separation
        self.spectrogram = None                      # Normalized frequency matrix (0 to 1)
        self.times = None                            # Timestamps for each spectrogram slice
        self.bin_maxima = None                       # Peak energy per frequency bin across the track
        self.stems_energy = {}                       # Isolated energy for each instrument (vocals, drums, etc)
        self.stem_types = ["vocals", "drums", "bass", "guitar", "piano", "other"] # Target stems for AI
        self.contrast_level = 0.20                   # Margin used to define noise floor and peak ceiling
        self.bin_floors = None                       # Calculated noise floor per frequency
        self.bin_peaks = None                        # Calculated peak ceiling per frequency
        self.num_bands = 3                           # Number of frequency groups (Bass, Mid, High)
        self.band_bins = []                          # Indices grouping raw frequencies into bands
        self.band_maxima = []                        # Peak energy per custom band
        self.band_floors = []                        # Noise floor per custom band
        self.band_peaks = []                         # Peak ceiling per custom band
        self.noise_threshold = 0.05                  # Minimum energy below which is considered silence
        self.temporal_smoothing = temporal_smoothing # Alpha factor for time-based smoothing
        self.frequency_smoothing = frequency_smoothing # Sigma factor for frequency-wise blurring
        self.gravity_factor = 0.9                    # Decay speed for values (simulates inertia)
        self.vocal_threshold = 0.1                   # Noise gate specifically for the vocal stem
        self.prev_features = None                    # The last call's features (no longer read: see _smoothed)
        self._smooth = {}                            # apply_gate -> (bands, stems) smoothed along the frames
        self.stem_files = {}                         # stem name -> its audio file (the AI's cache, the project's)
        self.n_fft = 2048                            # FFT window size (frequency resolution)
        self.hop_length = 512                        # Samples between analysis windows (time resolution)
        self.stem_service = StemService() if (StemService and not skip_separation) else None
        self.prebuilt_stems = prebuilt_stems or {}  # {name: file_path} — loaded additively after AI
        self.max_seconds = max_seconds               # Analyse only the first N seconds (None = whole file)
        self.progress = progress                     # progress(stage, done, of, message): stage "features" or "stems"

        self.load_audio()
        self.precompute_features()
        self.update_num_bands(3) 

    def _tell(self, stage, done, of, message=""):
        if self.progress is not None:
            self.progress(stage, done, of, message)

    def load_audio(self):
        self._tell("features", 0, 4, "decoding the song")
        self.y, self.sample_rate = librosa.load(self.file_path, sr=None, duration=self.max_seconds)
        self.duration = librosa.get_duration(y=self.y, sr=self.sample_rate)
        self.hop_length = int(self.sample_rate / self.fps)

    def precompute_features(self):
        self._tell("features", 1, 4, "the spectrum")
        self.stft_mag = np.abs(librosa.stft(self.y, n_fft=self.n_fft, hop_length=self.hop_length))
        self.spectrogram = librosa.amplitude_to_db(self.stft_mag, ref=np.max)
        spec_min, spec_max = self.spectrogram.min(), self.spectrogram.max()
        if spec_max > spec_min:
            self.spectrogram = (self.spectrogram - spec_min) / (spec_max - spec_min + 1e-6)
        self.spectrogram = np.nan_to_num(self.spectrogram)
        if self.frequency_smoothing > 0:
            self.spectrogram = gaussian_filter1d(self.spectrogram, sigma=self.frequency_smoothing, axis=0)
        self._tell("features", 2, 4, "waiting for the stems")
        self._separate_stems()
        if self.prebuilt_stems:
            self._load_prebuilt_stems()
        heard = sum(1 for e in self.stems_energy.values() if e is not None and len(e) and float(np.max(e)) > 0)
        self._tell("stems", 1, 1, f"{heard} with sound")
        self._tell("features", 3, 4, "onsets, sub-bands, texture")
        self._calculate_percentiles()
        self._precompute_audio_features()
        self.times = librosa.frames_to_time(np.arange(self.spectrogram.shape[1]), sr=self.sample_rate, hop_length=self.hop_length)
        self._tell("features", 4, 4, "done")

    SUBBAND_RANGES = {
        "sub_bass":   (20,    60),
        "bass":       (60,   250),
        "low_mid":    (250,  500),
        "mid":        (500, 2000),
        "high_mid":   (2000, 4000),
        "presence":   (4000, 6000),
        "brilliance": (6000, 20000),
    }

    def _precompute_audio_features(self):
        n_frames = self.stft_mag.shape[1]
        sr = self.sample_rate

        # 3.1 spectral centroid (Hz), normalized by Nyquist -> 0..1
        centroid_hz = librosa.feature.spectral_centroid(
            S=self.stft_mag, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        nyquist = sr / 2.0
        self.centroid = np.clip(centroid_hz / (nyquist + 1e-6), 0.0, 1.0)

        # 3.2 chroma (12, T) + dominant_pitch (T,)
        self.chroma = librosa.feature.chroma_stft(
            S=self.stft_mag ** 2, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        self.dominant_pitch = np.argmax(self.chroma, axis=0).astype(np.int32)

        # 3.3 spectral flux: sum of positive magnitude diffs per frame, normalized
        diff = np.diff(self.stft_mag, axis=1, prepend=self.stft_mag[:, :1])
        flux = np.sum(np.maximum(diff, 0.0), axis=0)
        f_max = float(flux.max())
        self.flux = flux / (f_max + 1e-6) if f_max > 0 else flux

        # 3.4 onset detection -> boolean mask per frame index
        onset_frames = librosa.onset.onset_detect(
            y=self.y, sr=sr, hop_length=self.hop_length, units="frames"
        )
        self.onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
        self.onset_mask = np.zeros(n_frames, dtype=bool)
        valid = (onset_frames >= 0) & (onset_frames < n_frames)
        self.onset_mask[onset_frames[valid]] = True

        # 3.5 sub-band energies (named Hz ranges), each normalized 0..1
        bin_freqs = np.arange(self.stft_mag.shape[0]) * (sr / self.n_fft)
        self.subband_names = list(self.SUBBAND_RANGES.keys())
        self.subbands = {}
        for name, (lo, hi) in self.SUBBAND_RANGES.items():
            mask = (bin_freqs >= lo) & (bin_freqs < hi)
            if mask.any():
                energy = self.stft_mag[mask].mean(axis=0)
            else:
                energy = np.zeros(n_frames)
            e_max = float(energy.max())
            self.subbands[name] = energy / (e_max + 1e-6) if e_max > 0 else energy

        # 3.5b the level: its RMS against its loudest moment (^1.3), as a stem's energy is read
        rms = librosa.feature.rms(y=self.y, hop_length=self.hop_length)[0][:n_frames]
        rms = np.pad(rms, (0, n_frames - len(rms)))
        r_max = float(rms.max())
        self.level = (rms / r_max) ** 1.3 if r_max > 0 else rms

        # 3.6 texture: loudness, swell, harmonic change, ... for material without attacks
        self.texture_names = list(TEXTURE_NAMES)
        self.texture = texture_features(self.stft_mag, sr, self.n_fft, self.hop_length)

    def _load_prebuilt_stems(self):
        """Load extra audio files into stems_energy. Called after normal separation."""
        n_frames = self.spectrogram.shape[1]
        for k, (name, path) in enumerate(self.prebuilt_stems.items()):
            self._tell("stems", k, len(self.prebuilt_stems), f"the project's {name}")
            try:
                sy, _ = librosa.load(path, sr=self.sample_rate, duration=self.max_seconds)
                energy = librosa.feature.rms(y=sy, hop_length=self.hop_length)[0]
                energy = energy[:n_frames] if len(energy) >= n_frames else np.pad(energy, (0, n_frames - len(energy)))
                e_max = float(energy.max())
                self.stems_energy[name] = (energy / e_max) ** 1.3 if e_max > 0 else energy
                self.stem_files[name] = str(path)
                print(f"[Stems] loaded {name} <- {Path(path).name}")
            except Exception as e:
                print(f"[Stems] failed to load {name} ({path}): {e}")
                self.stems_energy[name] = np.zeros(n_frames)

    def _separate_stems(self):
        if not self.stem_service:
            self._tell("stems", 0, 1, "AI separation skipped")
            for s in self.stem_types: self.stems_energy[s] = np.zeros(self.spectrogram.shape[1])
            return

        try:
            file_stats = str(self.file_path.stat().st_size) + str(self.file_path.stat().st_mtime)
            file_hash = hashlib.md5((str(self.file_path) + file_stats).encode()).hexdigest()
            stems_root = PROJECT_ROOT / "stems_output"
            cache_folder = stems_root / file_hash / self.separation_mode
            expected_stems = ["vocals"] if self.separation_mode == "vocals" else self.stem_types
            found_stems = {}
            if cache_folder.exists():
                for s_type in expected_stems:
                    files = list(cache_folder.glob(f"*{s_type}*.flac"))
                    if files: found_stems[s_type] = str(files[0])
            if len(found_stems) < len(expected_stems):
                self._tell("stems", 0, 1, f"separating with {self.separation_mode} (minutes, on the GPU)")
                stems = self.stem_service.separate(str(self.file_path), mode=self.separation_mode,
                                                   progress_callback=lambda msg, frac: self._tell("stems", frac, 1, msg))
                if stems:
                    old_path = Path(stems[0].file_path).parent
                    if cache_folder.exists(): shutil.rmtree(cache_folder)
                    cache_folder.mkdir(parents=True, exist_ok=True)
                    for stem in stems:
                        shutil.move(stem.file_path, str(cache_folder / Path(stem.file_path).name))
                    try: shutil.rmtree(old_path)
                    except: pass
                    for s_type in expected_stems:
                        files = list(cache_folder.glob(f"*{s_type}*.flac"))
                        if files: found_stems[s_type] = str(files[0])

            else:
                self._tell("stems", 0, 1, "from the cache")
            for k, (s_type, s_path) in enumerate(found_stems.items()):
                self._tell("stems", k, max(1, len(found_stems)), f"reading {s_type}")
                sy, _ = librosa.load(s_path, sr=self.sample_rate)
                energy = librosa.feature.rms(y=sy, hop_length=self.hop_length)[0]
                e_max = energy.max()
                self.stems_energy[s_type] = (energy / e_max) ** 1.3 if e_max > 0 else energy
                self.stem_files[s_type] = str(s_path)
            for s_type in self.stem_types:
                if s_type not in self.stems_energy:
                    self.stems_energy[s_type] = np.zeros(self.spectrogram.shape[1])
        except Exception as e:
            import traceback
            print(f"[Stems] Separation failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            for s in self.stem_types: self.stems_energy[s] = np.zeros(self.spectrogram.shape[1])

    def _calculate_percentiles(self):
        low_p, high_p = self.contrast_level * 100, (1.0 - self.contrast_level) * 100
        self.bin_floors = np.percentile(self.spectrogram, low_p, axis=1)
        self.bin_peaks = np.percentile(self.spectrogram, high_p, axis=1)
        self.bin_maxima = np.max(self.spectrogram, axis=1)

    def update_contrast(self, level_delta):
        self.contrast_level = max(0.0, min(0.45, self.contrast_level + level_delta))
        self._calculate_percentiles()
        self.update_num_bands(self.num_bands)

    def update_num_bands(self, n):
        self.num_bands = max(1, min(n, 64))
        freqs = librosa.mel_frequencies(n_mels=self.num_bands + 1, fmin=20, fmax=self.sample_rate / 2)
        bin_indices = [int(f * (self.n_fft / self.sample_rate)) for f in freqs]
        self.band_bins, self.band_maxima, self.band_floors, self.band_peaks = [], [], [], []
        for i in range(self.num_bands):
            start = bin_indices[i]
            end = max(start + 1, bin_indices[i+1])
            self.band_bins.append((start, end))
            if self.bin_peaks is not None:
                self.band_peaks.append(np.max(self.bin_peaks[start:end]))
                self.band_floors.append(np.mean(self.bin_floors[start:end]))
            else: self.band_peaks.append(1.0); self.band_floors.append(0.0)
        self.prev_features = None
        self._smooth = {}

    def _band_energies(self, frames) -> np.ndarray:
        """The bands' energies (num_bands x len(frames)), normalised as ``get_features_at_time``'s."""
        out = np.zeros((len(self.band_bins), len(frames)))
        for i, (start, end) in enumerate(self.band_bins):
            e = self.spectrogram[start:end, frames].mean(axis=0) if end > start else np.zeros(len(frames))
            p_floor, p_peak = self.band_floors[i], self.band_peaks[i]
            out[i] = (np.clip((e - p_floor) / (p_peak - p_floor + 1e-6), 0.0, 1.0) if p_peak > self.noise_threshold
                      else e * 0.2)
        return out

    def _smoothed(self, apply_gate: bool):
        """The bands and the stems with the gravity (an instant rise, a fall of ``gravity_factor``
        a frame) run along the frames once: (bands num_bands x T, {stem: T}). Before, it ran on
        each call, so a value depended on how many readers had asked before it (each layer's
        knobs ask every frame) and on the order of the calls (a preview, a seek)."""
        got = self._smooth.get(apply_gate)
        if got is not None:
            return got
        n = self.spectrogram.shape[1]
        names = list(self.stems_energy)
        rows = [self._band_energies(np.arange(n))]
        for s_type in names:
            e = np.asarray(self.stems_energy[s_type], dtype=float)[:n]
            e = np.pad(e, (0, n - len(e)))
            if apply_gate and s_type in self.stem_types:
                e = np.maximum(0, (e - self.vocal_threshold) / (1.0 - self.vocal_threshold + 1e-6))
            rows.append(e[None, :])
        raw = np.vstack(rows)
        out = raw.copy()
        g = self.gravity_factor
        for i in range(1, n):
            prev, cur = out[:, i - 1], raw[:, i]
            out[:, i] = np.where(cur > prev, cur, prev * g + cur * (1 - g))
        nb = len(self.band_bins)
        got = (out[:nb], {s_type: out[nb + k] for k, s_type in enumerate(names)})
        self._smooth[apply_gate] = got
        return got

    def _smoothed_spectrum(self, idx: int, depth: int = 24) -> np.ndarray:
        """The spectrum blended with the frames before it (``temporal_smoothing`` a frame), the
        blend summed over the last ``depth`` frames (0.7^24 < 2e-4) instead of carried from call
        to call."""
        f = self.temporal_smoothing
        k = min(idx, depth)
        cols = self.spectrogram[:, idx - k:idx + 1][:, ::-1]           # now, one before, ...
        w = (1 - f) * f ** np.arange(k + 1)
        w[-1] = f ** k                                                  # the oldest carries the rest
        return cols @ w

    def apply_gravity(self, current_val, prev_val):
        if prev_val is None or current_val > prev_val: return current_val
        return prev_val * self.gravity_factor + current_val * (1 - self.gravity_factor)

    def get_features_at_time(self, time_sec, use_smoothing=True, use_normalization=True, apply_gate=True):
        if time_sec > self.duration: return None
        idx = np.searchsorted(self.times, time_sec) - 1
        idx = max(0, min(idx, self.spectrogram.shape[1] - 1))
        stems_out = {}
        for s_type in self.stems_energy:
            energy = self.stems_energy[s_type][idx] if idx < len(self.stems_energy[s_type]) else 0
            if apply_gate and s_type in self.stem_types:
                energy = max(0, (energy - self.vocal_threshold) / (1.0 - self.vocal_threshold + 1e-6))
            stems_out[s_type] = energy
        band_energies, raw_spectrum = [], self.spectrogram[:, idx]
        for i, (start, end) in enumerate(self.band_bins):
            energy = np.mean(raw_spectrum[start:end]) if (end > start) else 0
            if use_normalization:
                p_floor, p_peak = self.band_floors[i], self.band_peaks[i]
                energy = max(0.0, min(1.0, (energy - p_floor) / (p_peak - p_floor + 1e-6))) if p_peak > self.noise_threshold else energy * 0.2
            band_energies.append(energy)
        current_features = {"spectrum": raw_spectrum, "bands": np.array(band_energies), "stems": stems_out, "vocal": stems_out["vocals"], "frame_idx": idx}
        current_features["centroid"] = float(self.centroid[idx])
        current_features["chroma"] = self.chroma[:, idx]
        current_features["dominant_pitch"] = int(self.dominant_pitch[idx])
        current_features["flux"] = float(self.flux[idx])
        current_features["onset"] = bool(self.onset_mask[idx])
        current_features["subbands"] = {n: float(self.subbands[n][idx]) for n in self.subband_names}
        current_features["texture"] = {n: float(self.texture[n][idx]) for n in self.texture_names}
        current_features["level"] = float(self.level[idx]) if idx < len(self.level) else 0.0
        current_features["nyquist"] = self.sample_rate / 2.0               # the spectrum spans 0 .. this (Hz)
        i0 = int(max(0.0, time_sec) * self.sample_rate)
        current_features["wave"] = self.y[i0:i0 + 512]                      # the waveform from here on (-1 .. 1)
        if use_smoothing:
            smoothed = {}
            bands_s, stems_s = self._smoothed(bool(apply_gate))
            smoothed["bands"] = bands_s[:, idx].copy()
            smoothed_stems = {s_type: float(stems_s[s_type][idx]) if s_type in stems_s else stems_out[s_type]
                              for s_type in stems_out}
            smoothed["stems"], smoothed["vocal"], smoothed["spectrum"] = smoothed_stems, smoothed_stems.get("vocals", 0.0), self._smoothed_spectrum(idx)
            b_point, m_point = _bm_points(self.num_bands)
            smoothed["bass"]  = _safe_mean(smoothed["bands"][:b_point])
            smoothed["mid"]   = _safe_mean(smoothed["bands"][b_point:m_point])
            smoothed["high"]  = _safe_mean(smoothed["bands"][m_point:])
            smoothed["pulse"], smoothed["frame_idx"] = 1.0 + (smoothed["bass"] ** 2) * 0.15, idx
            for k in ("centroid", "chroma", "dominant_pitch", "flux", "onset", "subbands", "texture", "nyquist", "wave", "level"):
                smoothed[k] = current_features[k]
            self.prev_features = smoothed
            return smoothed
        else:
            b_point, m_point = _bm_points(self.num_bands)
            current_features["bass"]  = _safe_mean(current_features["bands"][:b_point])
            current_features["mid"]   = _safe_mean(current_features["bands"][b_point:m_point])
            current_features["high"]  = _safe_mean(current_features["bands"][m_point:])
            current_features["pulse"] = 1.0 + (current_features["bass"] ** 2) * 0.15
            self.prev_features = current_features
            return current_features
