import logging
import os
from typing import Dict, List

import numpy as np

try:
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy.core.event import Catalog
    from obspy.core.stream import Stream

    _OBSPY_AVAILABLE = True
except ImportError:
    _OBSPY_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "obspy is not installed; IRISSeismicFetcher will run in degraded mode."
    )

logger = logging.getLogger(__name__)


class IRISSeismicFetcher:
    """Fetches seismic event data and waveforms from the IRIS FDSN service via obspy."""

    def __init__(self):
        self._client = None
        if _OBSPY_AVAILABLE:
            try:
                self._client = Client("IRIS")
                logger.info("IRIS FDSN client initialised.")
            except Exception as exc:
                logger.warning("Could not connect to IRIS FDSN: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_events_near_cables(
        self,
        cable_coords: List[tuple],
        radius_km: float = 500,
        starttime: str = "2020-01-01",
        endtime: str = "2024-12-31",
        minmagnitude: float = 4.0,
    ):
        """Fetch seismic events within the bounding box of *cable_coords*.

        Parameters
        ----------
        cable_coords : list of (lat, lon) tuples defining the cable route.
        radius_km    : not used for filtering here (bounding-box approach).
        starttime    : ISO date string for the query start.
        endtime      : ISO date string for the query end.
        minmagnitude : minimum moment magnitude.

        Returns
        -------
        obspy.Catalog or empty list when obspy is unavailable / network fails.
        """
        if not _OBSPY_AVAILABLE or self._client is None:
            logger.warning("obspy unavailable; returning empty catalog.")
            return []

        lats = [c[0] for c in cable_coords]
        lons = [c[1] for c in cable_coords]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # Expand bounding box by ~radius_km (≈ radius_km/111 degrees)
        pad = radius_km / 111.0
        min_lat -= pad
        max_lat += pad
        min_lon -= pad
        max_lon += pad

        try:
            catalog = self._client.get_events(
                starttime=UTCDateTime(starttime),
                endtime=UTCDateTime(endtime),
                minmagnitude=minmagnitude,
                maxdepth=700,
                minlatitude=min_lat,
                maxlatitude=max_lat,
                minlongitude=min_lon,
                maxlongitude=max_lon,
            )
            logger.info("Fetched %d seismic events from IRIS.", len(catalog))
            return catalog
        except Exception as exc:
            logger.error("Error fetching events from IRIS: %s", exc)
            return []

    def fetch_waveforms_for_event(self, event, network: str = "II"):
        """Retrieve broadband waveforms for a single seismic event.

        Parameters
        ----------
        event   : obspy Event object.
        network : FDSN network code (default ``"II"`` – Global Seismographic Network).

        Returns
        -------
        obspy.Stream or empty Stream on failure.
        """
        if not _OBSPY_AVAILABLE or self._client is None:
            logger.warning("obspy unavailable; returning empty Stream.")
            return Stream() if _OBSPY_AVAILABLE else None

        try:
            origin = event.preferred_origin() or event.origins[0]
            origin_time = origin.time
            st = self._client.get_waveforms(
                network=network,
                station="*",
                location="*",
                channel="BH*",
                starttime=origin_time - 60,
                endtime=origin_time + 300,
            )
            st.detrend("demean")
            st.taper(max_percentage=0.05)
            logger.info("Fetched %d traces for event %s.", len(st), event.resource_id)
            return st
        except Exception as exc:
            logger.error("Error fetching waveforms: %s", exc)
            return Stream()

    def compute_pgv_at_cable_nodes(self, stream, cable_nodes: List[tuple]) -> Dict[int, float]:
        """Estimate peak ground velocity (m/s) at each cable node.

        A simple distance-based attenuation model is applied when station
        coordinates are available in the trace metadata.

        Parameters
        ----------
        stream      : obspy Stream of velocity traces (or acceleration, integrated).
        cable_nodes : list of (lat, lon) tuples representing cable nodes.

        Returns
        -------
        dict mapping cable node index → PGV in m/s.
        """
        if not _OBSPY_AVAILABLE or stream is None or len(stream) == 0:
            return {i: 0.0 for i in range(len(cable_nodes))}

        pgv_map: Dict[int, float] = {}

        # Collect (station_lat, station_lon, pgv) from each trace
        station_pgv = []
        for tr in stream:
            data = tr.data.astype(float)
            pgv_trace = float(np.max(np.abs(data)))
            coords = tr.stats.get("coordinates", None)
            if coords:
                slat = coords.get("latitude", 0.0)
                slon = coords.get("longitude", 0.0)
            else:
                slat, slon = None, None
            station_pgv.append((slat, slon, pgv_trace))

        for node_idx, (nlat, nlon) in enumerate(cable_nodes):
            if not station_pgv:
                pgv_map[node_idx] = 0.0
                continue

            # Use distance-weighted average when coordinates are known
            weighted_sum = 0.0
            weight_total = 0.0
            for slat, slon, pgv_val in station_pgv:
                if slat is not None and slon is not None:
                    dist_deg = np.sqrt((nlat - slat) ** 2 + (nlon - slon) ** 2)
                    dist_km = dist_deg * 111.0
                    weight = 1.0 / max(dist_km, 1.0)
                else:
                    weight = 1.0
                weighted_sum += weight * pgv_val
                weight_total += weight

            pgv_map[node_idx] = weighted_sum / max(weight_total, 1e-9)

        return pgv_map

    def cache_events(self, catalog=None, path: str = "data/cache/seismic_events.xml") -> None:
        """Write the catalog to a QuakeML file.

        Parameters
        ----------
        catalog : obspy Catalog to persist.  If None, nothing is written.
        path    : destination file path.
        """
        if not _OBSPY_AVAILABLE:
            logger.warning("obspy unavailable; cannot cache seismic events.")
            return
        if catalog is None or len(catalog) == 0:
            logger.info("Empty catalog – nothing to cache.")
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        catalog.write(path, format="QUAKEML")
        logger.info("Seismic catalog (%d events) cached to %s.", len(catalog), path)
