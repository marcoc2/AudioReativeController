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

class AudioFeatureExtractor:
    """
    Advanced feature extractor with dynamic AI separation modes.
    """
    def __init__(self, file_path, fps=60, temporal_smoothing=0.7, frequency_smoothing=1.5, separation_mode="demucs"):
        self.file_path = Path(file_path)
        self.fps = fps
        self.sample_rate = 0
        self.duration = 0
        self.y = None
        self.separation_mode = separation_mode
        self.spectrogram = None
        self.times = None
        self.bin_maxima = None
        self.stems_energy = {}
        self.stem_types = ["vocals", "drums", "bass", "guitar", "piano", "other"]
        self.contrast_level = 0.20 
        self.bin_floors = None
        self.bin_peaks = None
        self.num_bands = 3
        self.band_bins = []
        self.band_maxima = []
        self.band_floors = []
        self.band_peaks = []
        self.noise_threshold = 0.05 
        self.temporal_smoothing = temporal_smoothing 
        self.frequency_smoothing = frequency_smoothing 
        self.gravity_factor = 0.9 
        self.vocal_threshold = 0.1 
        self.prev_features = None
        self.n_fft = 2048
        self.hop_length = 512
        self.stem_service = StemService() if StemService else None

        self.load_audio()
        self.precompute_features()
        self.update_num_bands(3) 

    def load_audio(self):
        self.y, self.sample_rate = librosa.load(self.file_path, sr=None)
        self.duration = librosa.get_duration(y=self.y, sr=self.sample_rate)
        self.hop_length = int(self.sample_rate / self.fps)

    def precompute_features(self):
        stft = np.abs(librosa.stft(self.y, n_fft=self.n_fft, hop_length=self.hop_length))
        self.spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
        spec_min, spec_max = self.spectrogram.min(), self.spectrogram.max()
        if spec_max > spec_min:
            self.spectrogram = (self.spectrogram - spec_min) / (spec_max - spec_min + 1e-6)
        self.spectrogram = np.nan_to_num(self.spectrogram)
        if self.frequency_smoothing > 0:
            self.spectrogram = gaussian_filter1d(self.spectrogram, sigma=self.frequency_smoothing, axis=0)
        self._separate_stems()
        self._calculate_percentiles()
        self.times = librosa.frames_to_time(np.arange(self.spectrogram.shape[1]), sr=self.sample_rate, hop_length=self.hop_length)

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
            print(f"[Feature Extractor] Error: {e}")
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
        for s_type in self.stem_types:
            energy = self.stems_energy[s_type][idx] if idx < len(self.stems_energy[s_type]) else 0
            if apply_gate:
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
        if use_smoothing and self.prev_features:
            smoothed, f = {}, self.temporal_smoothing
            prev_bands = self.prev_features["bands"]
            smoothed["bands"] = np.array([self.apply_gravity(band_energies[i], prev_bands[i] if i < len(prev_bands) else 0) for i in range(len(band_energies))])
            smoothed_stems, prev_stems = {}, self.prev_features["stems"]
            for s_type in self.stem_types: smoothed_stems[s_type] = self.apply_gravity(stems_out[s_type], prev_stems.get(s_type, 0))
            smoothed["stems"], smoothed["vocal"], smoothed["spectrum"] = smoothed_stems, smoothed_stems["vocals"], (1 - f) * raw_spectrum + f * self.prev_features["spectrum"]
            b_point, m_point = max(1, self.num_bands // 4), max(1, self.num_bands // 2)
            smoothed["bass"], smoothed["mid"], smoothed["high"] = np.mean(smoothed["bands"][:b_point]), np.mean(smoothed["bands"][b_point:m_point]), np.mean(smoothed["bands"][m_point:])
            smoothed["pulse"], smoothed["frame_idx"] = 1.0 + (smoothed["bass"] ** 2) * 0.15, idx
            self.prev_features = smoothed
            return smoothed
        else:
            b_point, m_point = max(1, self.num_bands // 4), max(1, self.num_bands // 2)
            current_features["bass"], current_features["mid"], current_features["high"] = np.mean(current_features["bands"][:b_point]), np.mean(current_features["bands"][b_point:m_point]), np.mean(current_features["bands"][m_point:])
            current_features["pulse"] = 1.0 + (current_features["bass"] ** 2) * 0.15
            self.prev_features = current_features
            return current_features
