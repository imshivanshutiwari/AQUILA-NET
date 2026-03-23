"""Tests for DDPM augmentation - 4 tests."""
import sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_ddpm_qsample_shape_finite(ddpm):
    batch = 2
    x0 = torch.zeros(batch, 1, ddpm.seq_len)
    t = torch.tensor([ddpm.T - 1] * batch)
    noise = torch.randn_like(x0)
    x_noisy = ddpm.q_sample(x0, t, noise)
    assert x_noisy.shape == x0.shape
    assert torch.isfinite(x_noisy).all()


def test_conditioned_forward_shape(ddpm):
    from diffusion.cable_conditioner import CableConditioner
    batch = 2
    x = torch.zeros(batch, 1, ddpm.seq_len)
    t = torch.zeros(batch, dtype=torch.long)
    cond = CableConditioner(condition_dim=32)
    material = torch.zeros(batch, dtype=torch.long)
    depth = torch.tensor([3500.0] * batch)
    age = torch.tensor([15.0] * batch)
    sev = torch.tensor([0.8] * batch)
    pos = torch.tensor([500.0] * batch)
    condition = cond(material, depth, age, sev, pos)
    pred = ddpm(x, t, condition)
    assert pred.shape == x.shape
    assert torch.isfinite(pred).all()


def test_deep_vs_shallow_differ():
    from diffusion.cable_conditioner import CableConditioner
    c = CableConditioner(condition_dim=32)
    mat = torch.zeros(1, dtype=torch.long)
    age = torch.tensor([15.0]); sev = torch.tensor([0.8]); pos = torch.tensor([500.0])
    d1 = c(mat, torch.tensor([3500.0]), age, sev, pos)
    d2 = c(mat, torch.tensor([200.0]), age, sev, pos)
    assert (d1 - d2).abs().max().item() > 1e-6


def test_augmented_dataset_grows(ddpm):
    from diffusion.augmentation_pipeline import AugmentationPipeline
    from data.otdr_physics_simulator import OTDRPhysicsSimulator
    otdr = OTDRPhysicsSimulator()
    pipeline = AugmentationPipeline(ddpm=ddpm, otdr_sim=otdr, config={"n_sabotage_augment": 8})
    n = 16
    X = np.zeros((n, 64))
    for i in range(n):
        X[i] = otdr.generate_nominal_waveform(50.0, n_splices=2, n_connectors=1)[:64]
    y = np.array([0, 1, 2, 3] * (n // 4))
    X_aug, y_aug = pipeline.augment_dataset(X, y)
    assert len(X_aug) > len(X)
    assert len(X_aug) == len(y_aug)
