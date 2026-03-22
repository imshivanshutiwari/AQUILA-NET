"""Tests for adaptive EKF - 4 tests."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_predict_covariance_finite_positive():
    from filters.adaptive_ekf import AdaptiveEKF
    ekf = AdaptiveEKF(phi1=0.8, phi2=0.1, Q=0.01, R=0.05)
    _x_pred, P_pred = ekf.predict()
    trace = np.trace(P_pred)
    assert trace > 0 and np.isfinite(trace), f"Trace must be finite positive; got {trace}"


def test_update_returns_dict_with_keys():
    from filters.adaptive_ekf import AdaptiveEKF
    ekf = AdaptiveEKF(phi1=0.8, phi2=0.1, Q=0.01, R=0.05)
    result = ekf.update(measurement=0.5)
    assert isinstance(result, dict)
    assert "filtered_score" in result
    assert "anomaly_detected" in result


def test_ar2_stationarity_condition():
    from filters.ar2_model import AR2Model
    assert AR2Model(phi1=0.8, phi2=0.1).is_stationary()
    assert not AR2Model(phi1=0.9, phi2=0.2).is_stationary()


def test_cusum_detects_injected_anomaly():
    from filters.adaptive_ekf import AdaptiveEKF
    ekf = AdaptiveEKF(phi1=0.8, phi2=0.1, Q=0.01, R=0.05, threshold=0.7, cusum_h=2.0)
    for _ in range(10):
        ekf.update(0.1)
    detected = False
    for _ in range(40):
        result = ekf.update(0.95)
        if result.get("change_detected") or result.get("anomaly_detected"):
            detected = True
            break
    assert detected, "CUSUM must alarm after sustained anomalous scores"
