import cmath
import math

import numpy as np


class PDEFeatureEncoder:
    """Encodes submarine cable physical parameters as PDE-derived features."""

    # ------------------------------------------------------------------
    # Telegrapher's equations
    # ------------------------------------------------------------------

    def encode_telegrapher_params(
        self,
        R: float,
        L: float,
        C: float,
        G: float,
        length_km: float,
    ) -> np.ndarray:
        """Encode transmission-line parameters via the Telegrapher equations.

        Args:
            R: resistance per unit length (Ω/m).
            L: inductance per unit length (H/m).
            C: capacitance per unit length (F/m).
            G: conductance per unit length (S/m).
            length_km: cable length in kilometres.

        Returns:
            1-D array [wave_speed, |char_impedance|, |propagation_const.real|,
                       length_km] of shape (4,).
        """
        # Phase velocity (m/s)
        wave_speed = 1.0 / math.sqrt(L * C)

        # Angular frequency at 1 kHz
        omega = 2.0 * math.pi * 1e3
        j = complex(0, 1)

        Z = R + j * omega * L   # series impedance per unit length
        Y = G + j * omega * C   # shunt admittance per unit length

        char_impedance = cmath.sqrt(Z / Y)
        propagation_const = cmath.sqrt(Z * Y)

        return np.array([
            wave_speed,
            abs(char_impedance),
            abs(propagation_const.real),
            length_km,
        ], dtype=float)

    # ------------------------------------------------------------------
    # Euler-Bernoulli beam equation
    # ------------------------------------------------------------------

    def encode_euler_bernoulli(
        self,
        EI: float,
        rho_A: float,
        length_m: float,
    ) -> np.ndarray:
        """Encode structural parameters via the Euler-Bernoulli beam equation.

        The fundamental natural frequency of a simply-supported beam is:
            f_1 = (π/L)² · √(EI / ρA)  [rad/s]

        Args:
            EI: flexural rigidity (N·m²).
            rho_A: mass per unit length (kg/m).
            length_m: cable segment length in metres.

        Returns:
            1-D array [natural_freq, EI/1e8, rho_A, length_m] of shape (4,).
        """
        length_safe = max(length_m, 1e-6)
        natural_freq = (math.pi / length_safe) ** 2 * math.sqrt(EI / max(rho_A, 1e-12))

        return np.array([
            natural_freq,
            EI / 1e8,
            rho_A,
            length_m,
        ], dtype=float)

    # ------------------------------------------------------------------
    # Combined encoder
    # ------------------------------------------------------------------

    def encode_cable(self, cable_dict: dict, config_dict: dict) -> np.ndarray:
        """Encode a cable using both PDE models, returning an 8-dimensional vector.

        Keys read from *cable_dict*:
            length_km (float)
            length_m  (float, optional – derived from length_km if absent)

        Keys read from *config_dict*:
            R, L, C, G  – Telegrapher parameters
            EI          – flexural rigidity (N·m²)
            rho_A       – mass per unit length (kg/m)

        Returns:
            Concatenation of ``encode_telegrapher_params`` (4-dim) and
            ``encode_euler_bernoulli`` (4-dim) → shape (8,).
        """
        length_km = float(cable_dict.get("length_km", 1.0))
        length_m = float(cable_dict.get("length_m", length_km * 1000.0))

        R = float(config_dict.get("R", 0.034))
        L = float(config_dict.get("L", 0.0002))
        C = float(config_dict.get("C", 0.00015))
        G = float(config_dict.get("G", 1e-6))
        EI = float(config_dict.get("EI", 2.5e8))
        rho_A = float(config_dict.get("rho_A", 8900.0 * math.pi * 0.02 ** 2))

        tele_feats = self.encode_telegrapher_params(R, L, C, G, length_km)
        eb_feats = self.encode_euler_bernoulli(EI, rho_A, length_m)

        return np.concatenate([tele_feats, eb_feats])
