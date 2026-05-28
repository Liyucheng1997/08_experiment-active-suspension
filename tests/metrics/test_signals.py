import numpy as np
import pytest

from risk_aware_active_suspension.metrics.signals import iae, itae, peak_abs, percentile95, rmse


def test_signal_metrics_match_hand_computed_values() -> None:
    values = np.array([0.0, 3.0, 4.0])

    assert rmse(values) == pytest.approx(np.sqrt(25.0 / 3.0))
    assert peak_abs(values) == 4.0
    assert percentile95(values) == pytest.approx(3.9)


def test_integral_metrics_match_trapezoid_rule() -> None:
    t = np.array([0.0, 1.0, 2.0])
    values = np.array([0.0, -2.0, 0.0])

    assert iae(t, values) == pytest.approx(2.0)
    assert itae(t, values) == pytest.approx(2.0)


def test_integral_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes"):
        iae([0.0, 1.0], [1.0])
