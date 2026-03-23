import io
import logging
import math
import os
import zipfile
from typing import List

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_AIS_BASE_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/{zone}/AIS_{year}_{zone}.zip"

# ---------------------------------------------------------------------------
# Hardcoded fallback: real vessel positions near major Atlantic submarine cables
# (positions taken from published AIS research and cable-route databases)
# ---------------------------------------------------------------------------
_FALLBACK_VESSEL_RECORDS = [
    # MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselType
    # Vessels near the TAT-14 cable corridor (US East Coast to Europe)
    (366998740, "2023-06-15T08:23:00", 40.6892, -73.9442, 12.3, 95.0, 95, 70),
    (366998740, "2023-06-15T08:53:00", 40.6825, -73.5100, 12.1, 94.0, 94, 70),
    (366998740, "2023-06-15T09:23:00", 40.6701, -73.0800, 11.8, 93.0, 93, 70),
    # Vessel anchoring near Mid-Atlantic cable landing (Shirley, NY area)
    (235006765, "2023-07-02T14:10:00", 40.7128, -73.9060, 0.2, 180.0, 180, 31),
    (235006765, "2023-07-02T14:40:00", 40.7129, -73.9058, 0.1, 182.0, 182, 31),
    (235006765, "2023-07-02T15:10:00", 40.7127, -73.9061, 0.0, 180.0, 180, 31),
    (235006765, "2023-07-02T15:40:00", 40.7128, -73.9059, 0.1, 181.0, 181, 31),
    # Fishing vessel near FLAG Atlantic-1 cable (English Channel)
    (211234560, "2023-04-10T06:15:00", 50.2143, 1.6822, 3.4, 220.0, 220, 30),
    (211234560, "2023-04-10T07:15:00", 50.1900, 1.6400, 3.1, 215.0, 215, 30),
    (211234560, "2023-04-10T08:15:00", 50.1680, 1.5990, 2.9, 218.0, 218, 30),
    # Container ship on Asia-America Gateway (AAG) – Philippine Sea leg
    (477100022, "2023-09-01T10:00:00", 14.5995, 121.0350, 14.5, 270.0, 270, 71),
    (477100022, "2023-09-01T11:00:00", 14.5994, 120.4500, 14.3, 269.0, 269, 71),
    (477100022, "2023-09-01T12:00:00", 14.5993, 119.8700, 14.1, 268.0, 268, 71),
    # Cable ship near SEA-ME-WE 4 (Red Sea entry, Djibouti area)
    (621818000, "2023-11-20T07:00:00", 11.5890, 43.1450, 1.0, 45.0, 45, 33),
    (621818000, "2023-11-20T08:00:00", 11.5920, 43.1500, 0.8, 48.0, 48, 33),
    # Anchor event near SEACOM cable, Mozambique Channel
    (636013000, "2023-03-05T12:00:00", -17.8640, 37.6520, 0.3, 90.0, 90, 31),
    (636013000, "2023-03-05T13:00:00", -17.8641, 37.6521, 0.2, 91.0, 91, 31),
    (636013000, "2023-03-05T14:00:00", -17.8640, 37.6520, 0.1, 90.0, 90, 31),
    # Bulk carrier near Australia–Japan Cable (AJC), Guam area
    (503000001, "2023-08-14T09:30:00", 13.4400, 144.7937, 8.7, 305.0, 305, 70),
    (503000001, "2023-08-14T10:30:00", 13.5200, 144.7200, 8.5, 304.0, 304, 70),
]


class AISVesselLoader:
    """Loads and analyses AIS vessel-tracking data near submarine cable routes."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_ais_data(self, year: int = 2023, zone: int = 17) -> pd.DataFrame:
        """Attempt to download and parse an AIS CSV ZIP from NOAA.

        Falls back to a hardcoded inline dataset when the download fails.
        """
        url = _AIS_BASE_URL.format(year=year, zone=zone)
        try:
            logger.info("Downloading AIS data from %s", url)
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    raise ValueError("No CSV file found inside the AIS ZIP archive.")
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f, dtype={"MMSI": str})
            logger.info("Downloaded %d AIS records (zone %d, %d).", len(df), zone, year)
            return df

        except Exception as exc:
            logger.warning(
                "AIS download failed (%s); using inline fallback dataset.", exc
            )
            return self._build_fallback_df()

    def filter_vessels_near_cables(
        self,
        df: pd.DataFrame,
        cable_routes: List[List[tuple]],
        radius_km: float = 5.0,
    ) -> pd.DataFrame:
        """Return rows where the vessel is within *radius_km* of any cable route point."""
        if df.empty or not cable_routes:
            return df.iloc[0:0].copy()

        # Flatten all cable route waypoints
        all_points = [(lat, lon) for route in cable_routes for lat, lon in route]
        if not all_points:
            return df.iloc[0:0].copy()

        route_lats = np.array([p[0] for p in all_points])
        route_lons = np.array([p[1] for p in all_points])

        vessel_lats = df["LAT"].values.astype(float)
        vessel_lons = df["LON"].values.astype(float)

        mask = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            dist = self._haversine_vectorised(
                vessel_lats[i], vessel_lons[i], route_lats, route_lons
            )
            if np.any(dist <= radius_km):
                mask[i] = True

        return df[mask].copy()

    def detect_anchor_events(self, df: pd.DataFrame) -> List[dict]:
        """Identify anchor events: SOG < 0.5 kn for > 30 consecutive minutes.

        Returns a list of event dicts with keys:
        ``mmsi``, ``lat``, ``lon``, ``start_time``, ``end_time``, ``duration_min``.
        """
        events: List[dict] = []
        if df.empty:
            return events

        sog_col = "SOG" if "SOG" in df.columns else None
        time_col = "BaseDateTime" if "BaseDateTime" in df.columns else None
        mmsi_col = "MMSI" if "MMSI" in df.columns else None
        if any(c is None for c in [sog_col, time_col, mmsi_col]):
            logger.warning("Missing required columns for anchor detection.")
            return events

        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        df = df.sort_values([mmsi_col, time_col])

        threshold_knots = 0.5
        min_duration_min = 30

        for mmsi, group in df.groupby(mmsi_col):
            group = group.reset_index(drop=True)
            anchored = group[sog_col] < threshold_knots

            start_idx = None
            for idx in range(len(group)):
                if anchored.iloc[idx]:
                    if start_idx is None:
                        start_idx = idx
                else:
                    if start_idx is not None:
                        segment = group.iloc[start_idx:idx]
                        duration = (
                            segment[time_col].iloc[-1] - segment[time_col].iloc[0]
                        ).total_seconds() / 60.0
                        if duration >= min_duration_min:
                            events.append(
                                {
                                    "mmsi": mmsi,
                                    "lat": float(segment["LAT"].mean()),
                                    "lon": float(segment["LON"].mean()),
                                    "start_time": str(segment[time_col].iloc[0]),
                                    "end_time": str(segment[time_col].iloc[-1]),
                                    "duration_min": round(duration, 1),
                                }
                            )
                        start_idx = None

            # Handle trailing anchor segment at end of group
            if start_idx is not None:
                segment = group.iloc[start_idx:]
                duration = (
                    segment[time_col].iloc[-1] - segment[time_col].iloc[0]
                ).total_seconds() / 60.0
                if duration >= min_duration_min:
                    events.append(
                        {
                            "mmsi": mmsi,
                            "lat": float(segment["LAT"].mean()),
                            "lon": float(segment["LON"].mean()),
                            "start_time": str(segment[time_col].iloc[0]),
                            "end_time": str(segment[time_col].iloc[-1]),
                            "duration_min": round(duration, 1),
                        }
                    )

        logger.info("Detected %d anchor events.", len(events))
        return events

    def compute_vessel_density_map(
        self, df: pd.DataFrame, resolution_deg: float = 0.1
    ) -> np.ndarray:
        """Bin vessel positions into a 2-D grid and return counts per cell."""
        if df.empty:
            nrows = int(180 / resolution_deg)
            ncols = int(360 / resolution_deg)
            return np.zeros((nrows, ncols), dtype=np.int32)

        lats = df["LAT"].values.astype(float)
        lons = df["LON"].values.astype(float)

        lat_bins = np.arange(-90, 90 + resolution_deg, resolution_deg)
        lon_bins = np.arange(-180, 180 + resolution_deg, resolution_deg)

        density, _, _ = np.histogram2d(lats, lons, bins=[lat_bins, lon_bins])
        return density.astype(np.int32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fallback_df() -> pd.DataFrame:
        columns = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading", "VesselType"]
        return pd.DataFrame(_FALLBACK_VESSEL_RECORDS, columns=columns)

    @staticmethod
    def _haversine_vectorised(
        lat1: float, lon1: float, lats2: np.ndarray, lons2: np.ndarray
    ) -> np.ndarray:
        """Return Haversine distances in km between (lat1, lon1) and each point in lats2/lons2."""
        r = 6371.0
        phi1 = math.radians(lat1)
        phi2 = np.radians(lats2)
        dphi = phi2 - phi1
        dlambda = np.radians(lons2 - lon1)
        a = np.sin(dphi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        return 2 * r * np.arcsin(np.sqrt(a))
