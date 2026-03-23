from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .mc_dropout import MCDropoutUncertainty


class UncertaintyMapper:
    """Maps per-node uncertainty estimates onto spatial (lat/lon) coordinates."""

    def __init__(self, mc_dropout: MCDropoutUncertainty) -> None:
        self.mc_dropout = mc_dropout
        self._last_epistemic: np.ndarray | None = None

    def compute_spatial_uncertainty(
        self,
        data,
        landing_points: list,
    ) -> pd.DataFrame:
        """Compute spatial uncertainty for each graph node.

        Args:
            data: torch_geometric Data object fed to the MC-Dropout predictor.
            landing_points: List of dicts (or objects) with keys/attributes
                            ``lat``, ``lon``, and optionally ``cable_id``.
                            Must have the same length as the number of nodes
                            in *data*.

        Returns:
            DataFrame with columns: lat, lon, epistemic, aleatoric, cable_id.
        """
        uncertainty = self.mc_dropout.predict_with_uncertainty(data)
        epistemic = uncertainty["epistemic"].ravel()
        aleatoric = uncertainty["aleatoric"].ravel()

        self._last_epistemic = epistemic

        n_nodes = len(epistemic)
        rows = []
        for i in range(n_nodes):
            if i < len(landing_points):
                pt = landing_points[i]
                if isinstance(pt, dict):
                    lat = pt.get("lat", float("nan"))
                    lon = pt.get("lon", float("nan"))
                    cable_id = pt.get("cable_id", "")
                else:
                    lat = getattr(pt, "lat", float("nan"))
                    lon = getattr(pt, "lon", float("nan"))
                    cable_id = getattr(pt, "cable_id", "")
            else:
                lat, lon, cable_id = float("nan"), float("nan"), ""

            rows.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "epistemic": float(epistemic[i]),
                    "aleatoric": float(aleatoric[i]),
                    "cable_id": cable_id,
                }
            )

        return pd.DataFrame(rows, columns=["lat", "lon", "epistemic", "aleatoric", "cable_id"])

    def get_high_uncertainty_nodes(self, threshold: float = 0.1) -> List[int]:
        """Return node indices whose epistemic uncertainty exceeds *threshold*.

        :meth:`compute_spatial_uncertainty` must be called first.

        Args:
            threshold: Minimum epistemic uncertainty value.

        Returns:
            List of integer node indices sorted by descending epistemic
            uncertainty.

        Raises:
            RuntimeError: If spatial uncertainty has not been computed yet.
        """
        if self._last_epistemic is None:
            raise RuntimeError(
                "Call compute_spatial_uncertainty() first."
            )
        indices = np.where(self._last_epistemic > threshold)[0]
        sorted_indices = indices[np.argsort(self._last_epistemic[indices])[::-1]]
        return sorted_indices.tolist()
