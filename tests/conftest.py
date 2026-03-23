"""Shared fixtures for all AQUILA-NET tests."""
import pytest
import numpy as np
import torch
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope='session')
def cable_data():
    from data.telegeography_fetcher import TeleGeographyFetcher
    fetcher = TeleGeographyFetcher()
    try:
        cables = fetcher.fetch_cables()
        if cables:
            return cables
    except Exception:
        pass
    return [
        {'cable_id': f'cable-{i}', 'name': f'Cable-{i}', 'landing_points': [],
         'owners': [], 'length_km': 5000 + i * 100, 'rfs_year': '2015',
         'cable_type': 'fiber', 'color': '#00ff44', 'anomaly_score': 0.1}
        for i in range(20)
    ]


@pytest.fixture(scope='session')
def ekf():
    from filters.adaptive_ekf import AdaptiveEKF
    return AdaptiveEKF(phi1=0.8, phi2=0.1, Q=0.01, R=0.05)


@pytest.fixture(scope='session')
def otdr_sim():
    from data.otdr_physics_simulator import OTDRPhysicsSimulator
    return OTDRPhysicsSimulator()


@pytest.fixture(scope='session')
def pinn_gnn():
    from graph.pinn_gnn import PINNGraphNeuralNetwork
    return PINNGraphNeuralNetwork(node_dim=8, hidden_dim=64, n_classes=4)


@pytest.fixture(scope='session')
def fl_server(pinn_gnn):
    from federated.fl_server import FederatedServer
    return FederatedServer(model=pinn_gnn, n_clients=5)


@pytest.fixture(scope='session')
def ddpm():
    from diffusion.ddpm_augmentor import DDPMAugmentor
    return DDPMAugmentor(seq_len=64, T=5)


@pytest.fixture(scope='session')
def sample_graph_data():
    import torch
    from torch_geometric.data import Data
    torch.manual_seed(42)
    n_nodes = 10
    x = torch.zeros(n_nodes, 8)
    for i in range(n_nodes):
        x[i] = torch.tensor([0.1 * i, 0.05 * i, 0.02 * i, 0.03 * i,
                              0.15, 0.2, 0.1, 0.08])
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8],
        [1, 0, 2, 1, 3, 2, 4, 3, 6, 5, 8, 7]
    ], dtype=torch.long)
    edge_attr = torch.zeros(edge_index.shape[1], 6)
    for i in range(edge_index.shape[1]):
        edge_attr[i] = torch.tensor([100.0 + i * 50, 10.0, 0.0, 0.1, 0.2, 4000.0])
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
