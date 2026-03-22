import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import linregress


class OTDRFeatureExtractor:
    """Extracts physical features from Optical Time-Domain Reflectometer waveforms."""

    def extract(self, waveform: np.ndarray, distance_km: float) -> dict:
        """Extract features from a single OTDR waveform.

        Args:
            waveform: 1-D array of backscatter power values (dB).
            distance_km: total fibre length represented by the waveform.

        Returns:
            Dictionary containing extracted features.
        """
        waveform = np.asarray(waveform, dtype=float)
        n = len(waveform)
        x_km = np.linspace(0.0, distance_km, n)

        # Attenuation slope via linear regression (dB/km)
        slope, intercept, r_value, p_value, std_err = linregress(x_km, waveform)
        attenuation_slope = float(slope)

        # Noise floor: mean of last 5 % of waveform samples
        tail_start = max(0, int(0.95 * n))
        noise_floor = float(np.mean(waveform[tail_start:]))

        # Reflections: peaks that exceed mean + 3·std
        waveform_mean = np.mean(waveform)
        waveform_std = np.std(waveform)
        threshold = waveform_mean + 3.0 * waveform_std
        peak_indices, _ = find_peaks(waveform, height=threshold)

        n_reflections = int(len(peak_indices))
        reflection_positions = x_km[peak_indices] if n_reflections > 0 else np.array([])
        max_reflection_amplitude = float(np.max(waveform[peak_indices])) if n_reflections > 0 else float(noise_floor)

        # Backscatter coefficient estimated from the attenuation slope
        # Convention: backscatter_coefficient ≈ -slope / 2  (one-way loss → two-way)
        backscatter_coefficient = float(-attenuation_slope / 2.0)

        return {
            "attenuation_slope": attenuation_slope,
            "noise_floor": noise_floor,
            "n_reflections": n_reflections,
            "reflection_positions": reflection_positions,
            "max_reflection_amplitude": max_reflection_amplitude,
            "backscatter_coefficient": backscatter_coefficient,
        }

    def extract_batch(self, waveforms: list, distances: list) -> pd.DataFrame:
        """Extract features from multiple waveforms and return a DataFrame.

        Args:
            waveforms: list of 1-D numpy arrays.
            distances: list of total fibre lengths in km (one per waveform).

        Returns:
            DataFrame with one row per waveform.  The *reflection_positions*
            column stores numpy arrays.
        """
        records = []
        for waveform, distance in zip(waveforms, distances):
            features = self.extract(waveform, distance)
            records.append(features)
        return pd.DataFrame(records)
