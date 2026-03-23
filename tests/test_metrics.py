"""Tests for evaluation metrics - 4 tests."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_auroc_perfect_classifier():
    from evaluation.metrics import AQUILAMetrics
    m = AQUILAMetrics()
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_score = np.array([0.0, 0.1, 0.2, 0.1, 0.05, 0.9, 0.95, 0.8, 0.85, 0.99])
    assert abs(m.compute_auroc(y_true, y_score) - 1.0) < 0.01


def test_f1_macro_perfect():
    from evaluation.metrics import AQUILAMetrics
    m = AQUILAMetrics()
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    assert abs(m.compute_f1(y, y, average="macro") - 1.0) < 0.01


def test_confusion_matrix_diagonal():
    from evaluation.metrics import AQUILAMetrics
    m = AQUILAMetrics()
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    cm = m.compute_confusion_matrix(y, y, normalize="true")
    assert cm.shape == (4, 4)
    assert np.allclose(np.diag(cm), 1.0, atol=1e-5)


def test_ece_range():
    from evaluation.metrics import AQUILAMetrics
    m = AQUILAMetrics()
    y_prob = np.full(80, 0.5)
    y_true = np.array([i % 2 for i in range(80)])
    ece = m.compute_ece(y_true, y_prob)
    assert isinstance(ece, float) and 0.0 <= ece <= 1.0
