import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ITU-T G.652D standard single-mode fibre parameters
# ---------------------------------------------------------------------------
_ALPHA_DB_KM = 0.2          # attenuation coefficient (dB/km)
_BACKSCATTER_COEFF_DBM = -78.0  # Rayleigh backscatter coefficient (dBm/m)
_PULSE_WIDTH_NS = 100       # pulse width (ns)
_REFRACTIVE_INDEX = 1.4677  # group refractive index
_CONNECTOR_LOSS_DB = 0.5    # per connector
_SPLICE_LOSS_DB = 0.1       # per splice

# Derived constants
_ALPHA_LINEAR = _ALPHA_DB_KM / (10.0 * math.log10(math.e))  # km^-1  ≈ 0.04607 km^-1
_SPEED_OF_LIGHT_KM_S = 2.998e5  # km/s
_PULSE_WIDTH_KM = (
    _SPEED_OF_LIGHT_KM_S / _REFRACTIVE_INDEX * (_PULSE_WIDTH_NS * 1e-9) / 2.0
)  # spatial extent of pulse in km (~0.010 km)

# Noise floor (dB) – corresponds to detector shot noise at ~-90 dBm
_NOISE_FLOOR_DB = -90.0
_NOISE_STD_DB = 0.05  # std-dev of Gaussian noise added to dB-domain trace values


class OTDRPhysicsSimulator:
    """Generates synthetic OTDR waveforms based on real ITU-T G.652D fibre physics."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_nominal_waveform(
        self,
        length_km: float,
        n_splices: int = 20,
        n_connectors: int = 2,
        n_points: int = 2000,
    ) -> np.ndarray:
        """Generate a nominal OTDR backscatter trace in dB vs. distance.

        P(z) [dB] = P0_dB - 2*alpha_dB*z  + backscatter_correction
                  + Fresnel peaks at connectors/splices
                  + noise floor

        Parameters
        ----------
        length_km     : total fibre length to simulate.
        n_splices     : number of fusion splices distributed uniformly.
        n_connectors  : number of connectors (typically at fibre ends).
        n_points      : number of distance samples.

        Returns
        -------
        waveform : 1-D array of dB values; index 0 = distance 0, last = length_km.
        """
        z = np.linspace(0.0, length_km, n_points)
        dz = z[1] - z[0]  # km per sample

        # ------------------------------------------------------------------
        # 1. Two-way Rayleigh backscatter power (continuous component)
        #    P(z) [dBm] = P0 [dBm] + BS_coeff + 10*log10(dz) - 2*alpha*z
        # ------------------------------------------------------------------
        p0_dbm = 0.0  # normalised launch power
        # Convert dBm/m backscatter coefficient to dBm/km
        bs_per_km = _BACKSCATTER_COEFF_DBM + 10.0 * math.log10(dz * 1e3)
        waveform = p0_dbm + bs_per_km - 2.0 * _ALPHA_DB_KM * z

        # ------------------------------------------------------------------
        # 2. Fresnel reflections – connectors (large) and splices (small)
        # ------------------------------------------------------------------
        # Connector positions: first and last (and uniformly if more than 2)
        connector_positions = np.linspace(0.0, length_km, n_connectors) if n_connectors > 0 else []
        for cp in connector_positions:
            waveform = self._add_fresnel_peak(
                waveform, z, cp, reflection_db=_CONNECTOR_LOSS_DB * 3.0, loss_db=_CONNECTOR_LOSS_DB
            )

        # Splice positions: uniformly distributed (skip positions 0 and length_km)
        if n_splices > 0:
            splice_pos = np.linspace(0.0, length_km, n_splices + 2)[1:-1]
            for sp in splice_pos:
                waveform = self._add_fresnel_peak(
                    waveform, z, sp, reflection_db=0.0, loss_db=_SPLICE_LOSS_DB
                )

        # ------------------------------------------------------------------
        # 3. Noise floor – deterministic seed based on length and splice count
        # ------------------------------------------------------------------
        rng = np.random.default_rng(seed=int(length_km * 100) + n_splices)
        noise = rng.normal(0.0, _NOISE_STD_DB, size=len(z))
        waveform += noise

        # Clamp to noise floor
        waveform = np.maximum(waveform, _NOISE_FLOOR_DB)
        return waveform

    def generate_seismic_fault(
        self,
        waveform: np.ndarray,
        event_position_km: float,
        magnitude_pga: float,
    ) -> np.ndarray:
        """Introduce a gradual additional loss beyond *event_position_km*.

        ΔdB(z) = 0.01 * magnitude_pga * (z - event_position_km)  for z > event_pos.

        Parameters
        ----------
        waveform          : nominal OTDR waveform (from ``generate_nominal_waveform``).
        event_position_km : distance at which the seismic damage begins.
        magnitude_pga     : PGA-based severity (proxy for shaking intensity).

        Returns
        -------
        Modified waveform (new array, original is not mutated).
        """
        n = len(waveform)
        z = np.linspace(0.0, self._estimate_length(waveform), n)
        modified = waveform.copy()
        for i, zi in enumerate(z):
            if zi > event_position_km:
                delta = 0.01 * magnitude_pga * (zi - event_position_km)
                modified[i] -= delta
        return np.maximum(modified, _NOISE_FLOOR_DB)

    def generate_anchor_drag(
        self,
        waveform: np.ndarray,
        position_km: float,
    ) -> np.ndarray:
        """Introduce a sharp Fresnel reflection at *position_km* (anchor cable drag).

        The trace continues beyond *position_km* but with an extra 3 dB loss.

        Returns a new waveform array.
        """
        n = len(waveform)
        z = np.linspace(0.0, self._estimate_length(waveform), n)
        modified = waveform.copy()

        # Add Fresnel reflection peak at drag location
        modified = self._add_fresnel_peak(
            modified, z, position_km, reflection_db=3.0, loss_db=3.0
        )
        return np.maximum(modified, _NOISE_FLOOR_DB)

    def generate_aging(
        self,
        waveform: np.ndarray,
        age_years: float,
        segment_id: int,
    ) -> np.ndarray:
        """Raise the noise floor proportional to cable age.

        Gaussian noise with σ = 0.002 * age_years is added using *segment_id*
        as a deterministic RNG seed (physics-based, reproducible).
        """
        rng = np.random.default_rng(seed=segment_id)
        noise_std = 0.002 * float(age_years)
        noise = rng.normal(0.0, noise_std, size=len(waveform))
        return np.maximum(waveform + noise, _NOISE_FLOOR_DB)

    def generate_sabotage(
        self,
        waveform: np.ndarray,
        cut_position_km: float,
        cable_length_km: float,
    ) -> np.ndarray:
        """Simulate a complete physical cut at *cut_position_km*.

        All samples beyond the cut are set to the total-reflection value and
        then to zero (detector noise floor) once the Fresnel echo decays.

        Returns a new waveform array.
        """
        n = len(waveform)
        z = np.linspace(0.0, cable_length_km, n)
        modified = waveform.copy()

        cut_idx = int(np.searchsorted(z, cut_position_km))
        if cut_idx >= n:
            return modified  # cut beyond simulated range

        # Fresnel reflection at cut face: Δ ≈ +15 dB (end-face back-reflection)
        peak_half_width = max(1, int(n * 0.003))  # ~0.3 % of trace width
        for k in range(cut_idx, min(cut_idx + peak_half_width, n)):
            modified[k] = modified[max(cut_idx - 1, 0)] + 15.0

        # Signal termination: set everything beyond cut to noise floor
        modified[cut_idx + peak_half_width :] = _NOISE_FLOOR_DB
        return modified

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_fresnel_peak(
        waveform: np.ndarray,
        z: np.ndarray,
        position_km: float,
        reflection_db: float,
        loss_db: float,
    ) -> np.ndarray:
        """Add a localised Fresnel reflection at *position_km* and a step loss.

        The peak is modelled as a Gaussian with σ = _PULSE_WIDTH_KM.
        Beyond *position_km* an additional one-way loss of *loss_db* is applied.
        """
        modified = waveform.copy()
        sigma = max(_PULSE_WIDTH_KM, (z[-1] - z[0]) / len(z) * 2)

        if reflection_db > 0:
            peak = reflection_db * np.exp(-0.5 * ((z - position_km) / sigma) ** 2)
            modified += peak

        # Step loss applied to all samples at and beyond the event position
        beyond = z >= position_km
        modified[beyond] -= loss_db

        return modified

    @staticmethod
    def _estimate_length(waveform: np.ndarray) -> float:
        """Estimate the cable length encoded in a waveform using its length.

        Uses the heuristic that waveforms are generated with 2000 points/cable.
        Falls back to len(waveform) * 0.05 km per sample.
        """
        return len(waveform) * 0.05
