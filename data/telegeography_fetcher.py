import json
import logging
import os
from typing import List

import networkx as nx
import requests

logger = logging.getLogger(__name__)

_CABLES_URL = (
    "https://raw.githubusercontent.com/telegeography/www.submarinecablemap.com"
    "/master/web/public/api/v3/cable/all.json"
)
_LANDING_POINTS_URL = (
    "https://raw.githubusercontent.com/telegeography/www.submarinecablemap.com"
    "/master/web/public/api/v3/landing-point/all.json"
)


class TeleGeographyFetcher:
    """Fetches and parses submarine cable data from TeleGeography's public API."""

    def __init__(self, timeout: int = 30):
        self._timeout = timeout
        self._cables: List[dict] = []
        self._landing_points: List[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_cables(self) -> List[dict]:
        """Return a list of cable dicts with normalised keys."""
        raw = self._get_json(_CABLES_URL)
        cables = raw.get("cables", raw) if isinstance(raw, dict) else raw
        result = []
        for cable in cables:
            rfs_raw = cable.get("rfs", None) or cable.get("rfs_year", None)
            try:
                rfs_year = int(str(rfs_raw).strip()) if rfs_raw else None
            except (ValueError, TypeError):
                rfs_year = None

            length_raw = cable.get("length", None)
            try:
                # lengths are sometimes strings like "~14000 km"
                length_km = float(
                    str(length_raw).replace("~", "").replace(",", "").split()[0]
                ) if length_raw else None
            except (ValueError, TypeError, IndexError):
                length_km = None

            result.append(
                {
                    "cable_id": cable.get("cable_id", cable.get("id", "")),
                    "name": cable.get("name", cable.get("cable_name", "")),
                    "landing_points": cable.get("landing_points", []),
                    "owners": cable.get("owners", []),
                    "length_km": length_km,
                    "rfs_year": rfs_year,
                    "cable_type": cable.get("cable_type", "Unknown"),
                    "color": cable.get("color", "#000000"),
                }
            )
        self._cables = result
        logger.info("Fetched %d cables from TeleGeography.", len(result))
        return result

    def fetch_landing_points(self) -> List[dict]:
        """Return a list of landing-point dicts with normalised keys."""
        raw = self._get_json(_LANDING_POINTS_URL)
        points = raw.get("landing_points", raw) if isinstance(raw, dict) else raw
        result = []
        for lp in points:
            lat, lon = self._parse_lat_lon(lp)
            result.append(
                {
                    "id": lp.get("id", lp.get("landing_point_id", "")),
                    "name": lp.get("name", ""),
                    "lat": lat,
                    "lon": lon,
                    "country": lp.get("country", lp.get("flag_name", "")),
                }
            )
        self._landing_points = result
        logger.info("Fetched %d landing points from TeleGeography.", len(result))
        return result

    def build_graph_from_real_data(self) -> nx.Graph:
        """Build a NetworkX graph where nodes are landing points and edges are cable segments."""
        if not self._cables:
            self.fetch_cables()
        if not self._landing_points:
            self.fetch_landing_points()

        lp_by_id = {lp["id"]: lp for lp in self._landing_points}

        G = nx.Graph()
        for lp in self._landing_points:
            G.add_node(
                lp["id"],
                name=lp["name"],
                lat=lp["lat"],
                lon=lp["lon"],
                country=lp["country"],
            )

        for cable in self._cables:
            lps = cable.get("landing_points", [])
            for i in range(len(lps) - 1):
                src = lps[i]
                dst = lps[i + 1]
                if src not in G.nodes or dst not in G.nodes:
                    continue
                G.add_edge(
                    src,
                    dst,
                    cable_id=cable["cable_id"],
                    cable_name=cable["name"],
                    length_km=cable.get("length_km"),
                    rfs_year=cable.get("rfs_year"),
                    color=cable.get("color", "#000000"),
                )

        logger.info(
            "Built graph with %d nodes and %d edges.", G.number_of_nodes(), G.number_of_edges()
        )
        return G

    def cache_locally(self, path: str = "data/cache/telegeography.json") -> None:
        """Save fetched cables and landing points to a local JSON cache."""
        if not self._cables:
            self.fetch_cables()
        if not self._landing_points:
            self.fetch_landing_points()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "cables": self._cables,
            "landing_points": self._landing_points,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        logger.info("TeleGeography data cached to %s.", path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_json(self, url: str) -> object:
        """Perform a GET request and return the parsed JSON."""
        resp = requests.get(url, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse_lat_lon(lp: dict):
        """Extract (lat, lon) from a landing-point dict, handling several formats."""
        # Some versions nest under "geometry.coordinates" [lon, lat]
        if "geometry" in lp and lp["geometry"]:
            coords = lp["geometry"].get("coordinates", [])
            if len(coords) == 2:
                return float(coords[1]), float(coords[0])
        # Flat keys
        lat = lp.get("latitude", lp.get("lat", 0.0))
        lon = lp.get("longitude", lp.get("lng", lp.get("lon", 0.0)))
        try:
            return float(lat), float(lon)
        except (TypeError, ValueError):
            return 0.0, 0.0
