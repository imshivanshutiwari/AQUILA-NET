import math
from typing import List

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data


class GraphConstructor:
    """Constructs PyTorch Geometric graphs from TeleGeography-style cable data."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_from_telegeography(self, cables: list, landing_points: list) -> Data:
        """Build a graph from TeleGeography cable and landing-point records.

        Args:
            cables: list of dicts with keys: 'landing_points' (list of landing
                    point IDs), 'length_km', 'rfs' (year), 'type'.
            landing_points: list of dicts with keys: 'id', 'lat', 'lon',
                            'country'.

        Returns:
            torch_geometric.data.Data with x, edge_index, edge_attr.
        """
        # Index landing points by ID
        id_to_idx = {lp["id"]: i for i, lp in enumerate(landing_points)}

        # Encode countries
        countries = [lp.get("country", "Unknown") for lp in landing_points]
        le = LabelEncoder()
        country_encoded = le.fit_transform(countries).astype(float)

        # Node features: [lat, lon, country_encoded]
        node_features = []
        for i, lp in enumerate(landing_points):
            node_features.append([
                float(lp.get("lat", 0.0)),
                float(lp.get("lon", 0.0)),
                float(country_encoded[i]),
            ])
        x = torch.tensor(node_features, dtype=torch.float)

        # Build edges from cables
        edge_src, edge_dst = [], []
        edge_attrs = []

        cable_type_encoder = _CableTypeEncoder()
        current_year = 2024

        for cable in cables:
            lp_ids = cable.get("landing_points", [])
            length_km = float(cable.get("length_km", 0.0))
            rfs = cable.get("rfs", current_year)
            try:
                age_years = float(current_year - int(rfs))
            except (ValueError, TypeError):
                age_years = 10.0
            ctype_enc = cable_type_encoder.encode(cable.get("type", "fiber"))

            # Connect every pair of landing points on the cable
            valid_ids = [lid for lid in lp_ids if lid in id_to_idx]
            for i in range(len(valid_ids)):
                for j in range(i + 1, len(valid_ids)):
                    src = id_to_idx[valid_ids[i]]
                    dst = id_to_idx[valid_ids[j]]

                    attr = [length_km, age_years, ctype_enc]

                    # Add both directions for undirected graph
                    edge_src.extend([src, dst])
                    edge_dst.extend([dst, src])
                    edge_attrs.extend([attr, attr])

        if edge_src:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 3), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def add_seismic_features(self, data: Data, seismic_events: list) -> Data:
        """Augment each node with an inverse-distance-weighted seismic risk score.

        Args:
            data: existing Data object with node features containing lat/lon
                  as the first two columns.
            seismic_events: list of dicts with keys 'lat', 'lon', 'magnitude'.

        Returns:
            Updated Data object with an extra seismic_risk column appended to x.
        """
        n_nodes = data.x.shape[0]
        seismic_risk = torch.zeros(n_nodes, 1, dtype=torch.float)

        for i in range(n_nodes):
            lat_node = float(data.x[i, 0])
            lon_node = float(data.x[i, 1])
            risk = 0.0
            for event in seismic_events:
                dist_km = _haversine(lat_node, lon_node,
                                     float(event["lat"]), float(event["lon"]))
                dist_km = max(dist_km, 1.0)  # avoid division by zero
                risk += float(event.get("magnitude", 1.0)) / (dist_km ** 2)
            seismic_risk[i, 0] = risk

        data.x = torch.cat([data.x, seismic_risk], dim=1)
        return data

    def normalize_features(self, data: Data) -> Data:
        """Z-score normalise node feature matrix in-place.

        Args:
            data: Data object whose x tensor will be normalised column-wise.

        Returns:
            The same Data object with normalised x.
        """
        mean = data.x.mean(dim=0, keepdim=True)
        std = data.x.std(dim=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        data.x = (data.x - mean) / std
        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _CableTypeEncoder:
    _mapping = {"fiber": 0.0, "coaxial": 0.25, "hybrid": 0.5, "power": 0.75}

    def encode(self, cable_type: str) -> float:
        return self._mapping.get(str(cable_type).lower(), 1.0)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))
