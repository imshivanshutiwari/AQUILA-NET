import logging
import os
from typing import Tuple

import networkx as nx
import numpy as np

try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    logging.getLogger(__name__).warning("PyYAML not installed; config will use defaults.")

from data.ais_vessel_loader import AISVesselLoader
from data.gebco_bathymetry import GEBCOBathymetry
from data.icpc_fault_parser import ICPCFaultParser
from data.iris_seismic_fetcher import IRISSeismicFetcher
from data.otdr_physics_simulator import OTDRPhysicsSimulator
from data.telegeography_fetcher import TeleGeographyFetcher

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = {
    "attenuation_coeff_dB_per_km": 0.2,
    "backscatter_coeff_dBm": -78.0,
    "refractive_index": 1.4677,
    "fault_classes": ["seismic", "anchor_drag", "aging", "sabotage"],
    "n_cls_nodes_real": 20,
    "seismic_min_magnitude": 4.0,
    "ais_anchor_speed_threshold_knots": 0.5,
}

# Representative Atlantic cable route waypoints (used for seismic bounding-box)
_ATLANTIC_ROUTE = [
    (40.63, -73.93),   # New York area landing
    (40.50, -60.00),   # Mid-Atlantic
    (45.00, -40.00),
    (50.00, -20.00),
    (53.33, -6.25),    # Dublin area landing
]


class DatasetBuilder:
    """Orchestrates all data fetchers to build an AQUILA-NET training dataset."""

    def __init__(self, config_path: str = "configs/cable_config.yaml"):
        self._config = _DEFAULT_CONFIG.copy()
        if _YAML_AVAILABLE and os.path.exists(config_path):
            with open(config_path, "r") as fh:
                loaded = yaml.safe_load(fh) or {}
            self._config.update(loaded)
            logger.info("Loaded config from %s.", config_path)
        else:
            logger.info("Using default config (config_path=%s not found or yaml unavailable).", config_path)

        self._tg_fetcher = TeleGeographyFetcher()
        self._iris_fetcher = IRISSeismicFetcher()
        self._gebco = GEBCOBathymetry()
        self._ais_loader = AISVesselLoader()
        self._icpc_parser = ICPCFaultParser()
        self._otdr = OTDRPhysicsSimulator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all_real_data(self) -> dict:
        """Run all fetchers and return a unified data dictionary.

        Keys: cables, landing_points, seismic_events, bathymetry, vessels, fault_events
        """
        logger.info("Fetching TeleGeography cable data …")
        cables = self._tg_fetcher.fetch_cables()
        landing_points = self._tg_fetcher.fetch_landing_points()

        logger.info("Fetching IRIS seismic events …")
        seismic_events = self._iris_fetcher.fetch_events_near_cables(
            cable_coords=_ATLANTIC_ROUTE,
            radius_km=500,
            starttime="2020-01-01",
            endtime="2024-12-31",
            minmagnitude=self._config.get("seismic_min_magnitude", 4.0),
        )

        logger.info("Loading GEBCO bathymetry …")
        self._gebco.load()
        bathymetry_df = self._gebco.extract_cable_route_bathymetry(_ATLANTIC_ROUTE)

        logger.info("Downloading AIS vessel data …")
        vessels = self._ais_loader.download_ais_data()

        logger.info("Loading ICPC fault statistics …")
        fault_events = self._icpc_parser.get_fault_events()

        payload = {
            "cables": cables,
            "landing_points": landing_points,
            "seismic_events": seismic_events,
            "bathymetry": bathymetry_df,
            "vessels": vessels,
            "fault_events": fault_events,
        }
        logger.info(
            "fetch_all_real_data complete: %d cables, %d landing points, "
            "%d fault events.",
            len(cables),
            len(landing_points),
            len(fault_events),
        )
        return payload

    def build_training_dataset(self) -> Tuple[np.ndarray, np.ndarray, nx.Graph]:
        """Build feature matrix X, label vector y, and cable graph for training.

        Waveforms are generated with OTDRPhysicsSimulator for each cable
        segment.  Labels correspond to the fault classes defined in config.

        Returns
        -------
        X     : 2-D float32 array, shape (n_samples, n_features).
        y     : 1-D int32 array of class indices, shape (n_samples,).
        graph : nx.Graph of landing-point nodes and cable edges.
        """
        graph = self._tg_fetcher.build_graph_from_real_data()

        fault_classes = self._config.get(
            "fault_classes", ["seismic", "anchor_drag", "aging", "sabotage"]
        )
        class_to_idx = {c: i for i, c in enumerate(fault_classes)}
        n_nodes = max(self._config.get("n_cls_nodes_real", 20), 20)

        # Retrieve cables; limit to first n_nodes for tractability
        cables = self._tg_fetcher._cables or self._tg_fetcher.fetch_cables()
        cables = cables[:n_nodes] if len(cables) > n_nodes else cables
        if not cables:
            # Fallback: synthetic 20 cable stubs
            cables = [{"cable_id": f"c{i}", "length_km": 1000 + i * 50} for i in range(n_nodes)]

        seismic_events_list = self._iris_fetcher.fetch_events_near_cables(
            cable_coords=_ATLANTIC_ROUTE,
            radius_km=500,
            starttime="2020-01-01",
            endtime="2024-12-31",
            minmagnitude=4.0,
        )
        n_seismic = len(seismic_events_list) if seismic_events_list else 0

        X_rows = []
        y_rows = []

        for seg_idx, cable in enumerate(cables):
            length_km = float(cable.get("length_km") or 1000.0)
            n_splices = max(5, int(length_km / 50))
            n_connectors = 2

            nominal = self._otdr.generate_nominal_waveform(
                length_km=length_km,
                n_splices=n_splices,
                n_connectors=n_connectors,
            )

            # -- Nominal (no fault) -- label = -1 (or add as "nominal" class if defined)
            # We generate one sample per fault class per cable segment
            for fault_class in fault_classes:
                fault_idx = class_to_idx[fault_class]

                if fault_class == "seismic":
                    # Magnitude cycles 5.0 → 5.5 → 6.0 per segment, approximating
                    # the Gutenberg-Richter b≈1 distribution in discrete steps.
                    magnitude = 5.0 + (seg_idx % 3) * 0.5
                    waveform = self._otdr.generate_seismic_fault(
                        nominal,
                        event_position_km=length_km * 0.4,
                        magnitude_pga=magnitude,
                    )
                elif fault_class == "anchor_drag":
                    waveform = self._otdr.generate_anchor_drag(
                        nominal, position_km=length_km * 0.3
                    )
                elif fault_class == "aging":
                    waveform = self._otdr.generate_aging(
                        nominal, age_years=15.0, segment_id=seg_idx
                    )
                elif fault_class == "sabotage":
                    waveform = self._otdr.generate_sabotage(
                        nominal,
                        cut_position_km=length_km * 0.6,
                        cable_length_km=length_km,
                    )
                else:
                    waveform = nominal.copy()

                # Compute simple statistical features from the waveform
                features = self._extract_waveform_features(waveform, length_km, seg_idx)
                X_rows.append(features)
                y_rows.append(fault_idx)

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.int32)
        logger.info(
            "build_training_dataset: X=%s, y=%s, graph nodes=%d edges=%d",
            X.shape,
            y.shape,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return X, y, graph

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_waveform_features(waveform: np.ndarray, length_km: float, seg_idx: int) -> np.ndarray:
        """Compute a fixed-length feature vector from an OTDR waveform."""
        # Statistical moments
        mean_val = float(np.mean(waveform))
        std_val = float(np.std(waveform))
        min_val = float(np.min(waveform))
        max_val = float(np.max(waveform))
        median_val = float(np.median(waveform))

        # Slope (linear regression coefficient)
        n = len(waveform)
        x_idx = np.arange(n, dtype=np.float64)
        slope = float(np.polyfit(x_idx, waveform.astype(np.float64), 1)[0])

        # Energy / integrated power
        energy = float(np.sum(waveform ** 2)) / n

        # Peak-to-peak
        ptp = max_val - min_val

        # Count "anomaly" samples below a threshold
        threshold = mean_val - 2.0 * std_val
        anomaly_fraction = float(np.mean(waveform < threshold))

        # Gradient statistics
        grad = np.diff(waveform)
        grad_mean = float(np.mean(np.abs(grad)))
        grad_max = float(np.max(np.abs(grad)))

        return np.array(
            [
                length_km,
                float(seg_idx),
                mean_val,
                std_val,
                min_val,
                max_val,
                median_val,
                slope,
                energy,
                ptp,
                anomaly_fraction,
                grad_mean,
                grad_max,
            ],
            dtype=np.float32,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    builder = DatasetBuilder()
    builder.fetch_all_real_data()
