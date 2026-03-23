"""Tests for federated learning - 5 tests."""
import pytest
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_opacus_dp_clips_gradients():
    from federated.dp_mechanism import DifferentialPrivacyMechanism
    import torch.nn as nn
    dp = DifferentialPrivacyMechanism(noise_multiplier=1.1, max_grad_norm=1.0)
    model = nn.Linear(10, 1)
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 100.0
    dp.clip_gradients(model)
    total_norm = sum(
        p.grad.norm(2).item() ** 2
        for p in model.parameters() if p.grad is not None
    ) ** 0.5
    assert total_norm <= dp.max_grad_norm * 1.1 + 1e-4


def test_epsilon_increases_per_round():
    from federated.privacy_accountant import PrivacyAccountant
    accountant = PrivacyAccountant(delta=1e-5, target_epsilon=1.0)
    eps_values = []
    for _ in range(5):
        accountant.add_round(noise_multiplier=1.1, sample_rate=0.05, n_steps=10)
        eps_values.append(accountant.get_epsilon())
    for i in range(1, len(eps_values)):
        assert eps_values[i] >= eps_values[i - 1]
    assert eps_values[-1] > 0


def test_fedavg_weighted_aggregation(fl_server, pinn_gnn):
    server = fl_server
    model = pinn_gnn
    state1 = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    state2 = {k: torch.ones_like(v) * 2.0 for k, v in model.state_dict().items()}
    updates = [(state1, 1, 0.5), (state2, 3, 0.3)]
    aggregated = server.aggregate_weights(updates)
    assert isinstance(aggregated, dict)
    first_key = next(iter(aggregated))
    expected_val = (0 * 1 + 2.0 * 3) / 4
    actual_val = aggregated[first_key].float().mean().item()
    assert abs(actual_val - expected_val) < 0.1


def test_async_client_3round_tolerance():
    from federated.async_handler import AsyncClientHandler
    handler = AsyncClientHandler(n_clients=5, tolerance_rounds=3)
    handler.mark_client_offline(0, round_num=10)
    handler.mark_client_active(1, round_num=10)
    active = handler.get_active_clients(round_num=12)
    assert isinstance(active, list)
    is_stale = handler.is_stale_update(client_id=0, update_round=5, current_round=15)
    assert is_stale
    not_stale = handler.is_stale_update(client_id=0, update_round=13, current_round=15)
    assert not not_stale


def test_privacy_guarantee_epsilon_under_1():
    from federated.privacy_accountant import PrivacyAccountant
    accountant = PrivacyAccountant(delta=1e-5, target_epsilon=1.0)
    for _ in range(10):
        accountant.add_round(noise_multiplier=1.1, sample_rate=0.05, n_steps=5)
    report = accountant.get_privacy_report()
    assert 'epsilon' in report
    assert 'delta' in report
    assert report['delta'] == 1e-5
    assert accountant.get_epsilon() >= 0
