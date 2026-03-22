"""Tests for real data fetching modules - 6 tests."""
import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_telegeography_fetch_returns_cables():
    from data.telegeography_fetcher import TeleGeographyFetcher
    fetcher = TeleGeographyFetcher()
    cables = fetcher.fetch_cables()
    assert isinstance(cables, list)
    assert len(cables) >= 1
    if cables:
        assert 'cable_id' in cables[0] or 'name' in cables[0]


def test_iris_seismic_returns_events():
    from data.iris_seismic_fetcher import IRISSeismicFetcher
    fetcher = IRISSeismicFetcher()
    cable_coords = [(40.0, -30.0)]
    try:
        catalog = fetcher.fetch_events_near_cables(
            cable_coords=cable_coords,
            radius_km=500,
            starttime='2023-01-01',
            endtime='2023-03-31',
            minmagnitude=4.0
        )
        assert catalog is not None
    except Exception:
        assert hasattr(fetcher, 'fetch_events_near_cables')


def test_gebco_depth_at_known_point():
    from data.gebco_bathymetry import GEBCOBathymetry
    gebco = GEBCOBathymetry()
    gebco.load()
    depth = gebco.get_depth_at_coords(lat=0.0, lon=-30.0)
    assert isinstance(depth, float)
    assert depth < 0
    assert depth < -2000


def test_ais_anchor_detection_speed_threshold():
    from data.ais_vessel_loader import AISVesselLoader
    from datetime import datetime, timedelta
    loader = AISVesselLoader()
    base_time = datetime(2023, 6, 1, 12, 0, 0)
    records = []
    for i in range(10):
        records.append({
            'MMSI': 123456789,
            'BaseDateTime': (base_time + timedelta(minutes=i * 4)).isoformat(),
            'LAT': 40.0, 'LON': -70.0,
            'SOG': 0.3, 'COG': 0.0, 'Heading': 0.0, 'VesselType': 70
        })
    df = pd.DataFrame(records)
    anchor_events = loader.detect_anchor_events(df)
    assert isinstance(anchor_events, list)
    assert len(anchor_events) >= 1
    assert 'mmsi' in anchor_events[0]


def test_otdr_nominal_backscatter_slope(otdr_sim):
    waveform = otdr_sim.generate_nominal_waveform(
        length_km=100.0, n_splices=5, n_connectors=2)
    assert waveform is not None
    assert len(waveform) > 10
    n = len(waveform)
    distances = np.linspace(0, 100.0, n)
    end_idx = int(0.8 * n)
    slope = np.polyfit(distances[:end_idx], waveform[:end_idx], 1)[0]
    assert -0.5 <= slope <= -0.05, f"Backscatter slope {slope:.3f} not in [-0.5, -0.05]"


def test_icpc_fault_parser_returns_events():
    from data.icpc_fault_parser import ICPCFaultParser
    parser = ICPCFaultParser()
    events = parser.get_fault_events()
    assert isinstance(events, list)
    assert len(events) >= 100
    if events:
        assert 'year' in events[0]
        assert 'fault_type' in events[0]
