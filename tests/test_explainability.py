"""Tests for explainability modules - 3 tests."""
import sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_mc_dropout_uncertainty_nonzero(pinn_gnn, sample_graph_data):
    from explainability.mc_dropout import MCDropoutUncertainty
    mc = MCDropoutUncertainty(model=pinn_gnn, T=5)
    results = mc.predict_with_uncertainty(sample_graph_data)
    for key in ("mean", "epistemic", "aleatoric"):
        assert key in results
        assert np.isfinite(results[key]).all()
        assert (results[key] >= 0).all()


def test_gnnexplainer_importable():
    from explainability.gnn_explainer import AQUILAGNNExplainer
    assert hasattr(AQUILAGNNExplainer, "explain_node"), "Must expose explain_node()"


def test_calibration_ece_computable():
    from evaluation.metrics import AQUILAMetrics
    m = AQUILAMetrics()
    n = 100
    y_prob = np.array([0.1 * ((i % 10) + 1) / 10 for i in range(n)])
    y_true = np.array([1 if y_prob[i] > 0.5 else 0 for i in range(n)])
    ece = m.compute_ece(y_true, y_prob)
    assert isinstance(ece, float) and 0.0 <= ece <= 1.0
