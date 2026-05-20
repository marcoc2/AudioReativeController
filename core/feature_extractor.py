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
                 prebuilt_stems=None, skip_separation=False):
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
        self.prev_features = None                    # Cache of previous frame for smoothing logic
        self.n_fft = 2048                            # FFT window size (frequency resolution)
        self.hop_length = 512                        # Samples between analysis windows (time resolution)
        self.stem_service = StemService() if (StemService and not skip_separation) else None
        self.prebuilt_stems = prebuilt_stems or {}  # {name: file_path} — loaded additively after AI

        self.load_audio()
        self.precompute_features()
        self.update_num_bands(3) 

    def load_audio(self):
        self.y, self.sample_rate = librosa.load(self.file_path, sr=None)
        self.duration = librosa.get_duration(y=self.y, sr=self.sample_rate)
        self.hop_length = int(self.sample_rate / self.fps)

    def precompute_features(self):
        self.stft_mag = np.abs(librosa.stft(self.y, n_fft=self.n_fft, hop_length=self.hop_length))
        self.spectrogram = librosa.amplitude_to_db(self.stft_mag, ref=np.max)
        spec_min, spec_max = self.spectrogram.min(), self.spectrogram.max()
        if spec_max > spec_min:
            self.spectrogram = (self.spectrogram - spec_min) / (spec_max - spec_min + 1e-6)
        self.spectrogram = np.nan_to_num(self.spectrogram)
        if self.frequency_smoothing > 0:
            self.spectrogram = gaussian_filter1d(self.spectrogram, sigma=self.frequency_smoothing, axis=0)
        self._separate_stems()
        if self.prebuilt_stems:
            self._load_prebuilt_stems()
        self._calculate_percentiles()
        self._precompute_audio_features()
        self.times = librosa.frames_to_time(np.arange(self.spectrogram.shape[1]), sr=self.sample_rate, hop_length=self.hop_length)

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

    def _load_prebuilt_stems(self):
        """Load extra audio files into stems_energy. Called after normal separation."""
        n_frames = self.spectrogram.shape[1]
        for name, path in self.prebuilt_stems.items():
            try:
                sy, _ = librosa.load(path, sr=self.sample_rate)
                energy = librosa.feature.rms(y=sy, hop_length=self.hop_length)[0]
                energy = energy[:n_frames] if len(energy) >= n_frames else np.pad(energy, (0, n_frames - len(energy)))
                e_max = float(energy.max())
                self.stems_energy[name] = (energy / e_max) ** 1.3 if e_max > 0 else energy
                print(f"[Stems] loaded {name} <- {Path(path).name}")
            except Exception as e:
                print(f"[Stems] failed to load {name} ({path}): {e}")
                self.stems_energy[name] = np.zeros(n_frames)

    def _separate_stems(self):
        if not self.stem_service:
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
                stems = self.stem_service.separate(str(self.file_path), mode=self.separation_mode)
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

            for s_type, s_path in found_stems.items():
                sy, _ = librosa.load(s_path, sr=self.sample_rate)
                energy = librosa.feature.rms(y=sy, hop_length=self.hop_length)[0]
                e_max = energy.max()
                self.stems_energy[s_type] = (energy / e_max) ** 1.3 if e_max > 0 else energy
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
        if use_smoothing and self.prev_features:
            smoothed, f = {}, self.temporal_smoothing
            prev_bands = self.prev_features["bands"]
            smoothed["bands"] = np.array([self.apply_gravity(band_energies[i], prev_bands[i] if i < len(prev_bands) else 0) for i in range(len(band_energies))])
            smoothed_stems, prev_stems = {}, self.prev_features["stems"]
            for s_type in stems_out: smoothed_stems[s_type] = self.apply_gravity(stems_out[s_type], prev_stems.get(s_type, 0))
            smoothed["stems"], smoothed["vocal"], smoothed["spectrum"] = smoothed_stems, smoothed_stems.get("vocals", 0.0), (1 - f) * raw_spectrum + f * self.prev_features["spectrum"]
            b_point, m_point = _bm_points(self.num_bands)
            smoothed["bass"]  = _safe_mean(smoothed["bands"][:b_point])
            smoothed["mid"]   = _safe_mean(smoothed["bands"][b_point:m_point])
            smoothed["high"]  = _safe_mean(smoothed["bands"][m_point:])
            smoothed["pulse"], smoothed["frame_idx"] = 1.0 + (smoothed["bass"] ** 2) * 0.15, idx
            for k in ("centroid", "chroma", "dominant_pitch", "flux", "onset", "subbands"):
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
