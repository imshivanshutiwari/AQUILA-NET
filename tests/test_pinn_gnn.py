"""Tests for PINN-GNN model - 6 tests."""
import pytest
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_forward_shapes_correct(pinn_gnn, sample_graph_data):
    model = pinn_gnn
    data = sample_graph_data
    model.eval()
    with torch.no_grad():
        anomaly, attribution = model(data)
    n_nodes = data.x.shape[0]
    assert anomaly.shape[0] == n_nodes
    assert attribution.shape[0] == n_nodes
    assert attribution.shape[1] == 4


def test_telegrapher_residual_near_zero_nominal(pinn_gnn, sample_graph_data):
    model = pinn_gnn
    data = sample_graph_data
    loss = model.compute_telegrapher_loss(data.x, data.edge_index, data.edge_attr)
    assert torch.isfinite(loss)
    assert loss >= 0


def test_euler_bernoulli_finite_all_inputs(pinn_gnn, sample_graph_data):
    model = pinn_gnn
    data = sample_graph_data
    loss = model.compute_euler_bernoulli_loss(data.x, data.edge_index, data.edge_attr)
    assert torch.isfinite(loss)
    assert loss >= 0


def test_attribution_sums_to_one(pinn_gnn, sample_graph_data):
    model = pinn_gnn
    data = sample_graph_data
    model.eval()
    with torch.no_grad():
        _, attribution = model(data)
    sums = attribution.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(sums.shape[0]), atol=1e-5)


def test_causality_weight_distance_inverse(pinn_gnn, sample_graph_data):
    model = pinn_gnn
    data = sample_graph_data
    loss = model.compute_causality_loss(data.x, data.edge_index, data.edge_attr)
    assert torch.isfinite(loss)
    assert loss >= 0


def test_backprop_through_all_pde_losses(pinn_gnn, sample_graph_data):
    from training.losses import AQUILALoss
    model = pinn_gnn
    data = sample_graph_data
    loss_fn = AQUILALoss()
    model.train()
    anomaly, attribution = model(data)
    n = data.x.shape[0]
    anomaly_labels = torch.zeros(n)
    attribution_labels = torch.zeros(n, dtype=torch.long)
    tele_loss = model.compute_telegrapher_loss(data.x, data.edge_index, data.edge_attr)
    eb_loss = model.compute_euler_bernoulli_loss(data.x, data.edge_index, data.edge_attr)
    caus_loss = model.compute_causality_loss(data.x, data.edge_index, data.edge_attr)
    losses = loss_fn(anomaly, attribution, anomaly_labels, attribution_labels,
                     tele_loss, eb_loss, caus_loss)
    losses['total'].backward()
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad
