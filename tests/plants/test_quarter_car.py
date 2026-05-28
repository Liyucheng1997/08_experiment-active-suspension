import numpy as np
import pytest

from risk_aware_active_suspension.plants.quarter_car import QuarterCar


def test_quarter_car_matrices_match_hand_derived(default_vehicle) -> None:
    plant = QuarterCar(default_vehicle)
    p = default_vehicle
    m_s = p.m / 4.0
    m_u = p.m_u

    expected_A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-p.k_s / m_s, -p.c_s / m_s, p.k_s / m_s, p.c_s / m_s],
            [0.0, 0.0, 0.0, 1.0],
            [p.k_s / m_u, p.c_s / m_u, -(p.k_s + p.k_t) / m_u, -p.c_s / m_u],
        ]
    )
    expected_B = np.array([[0.0], [1.0 / m_s], [0.0], [-1.0 / m_u]])
    expected_E = np.array([[0.0], [0.0], [0.0], [p.k_t / m_u]])

    np.testing.assert_allclose(plant.A, expected_A)
    np.testing.assert_allclose(plant.B, expected_B)
    np.testing.assert_allclose(plant.E, expected_E)


def test_unforced_energy_decays_monotonically(default_vehicle) -> None:
    plant = QuarterCar(default_vehicle)
    t = np.arange(0.0, 5.0, 0.001)
    states = plant.simulate(t, u_seq=0.0, w_seq=0.0, x0=[0.05, 0.0, 0.0, 0.0])
    energy = np.array([plant.energy(x) for x in states])

    assert energy[-1] < 0.02 * energy[0]
    assert np.max(np.diff(energy)) <= 1e-8


def test_eigenvalues_have_sprung_and_unsprung_modes(default_vehicle) -> None:
    plant = QuarterCar(default_vehicle)
    eigvals = np.linalg.eigvals(plant.A)
    mode_hz = sorted(abs(ev) / (2.0 * np.pi) for ev in eigvals if ev.imag > 0.0)

    assert len(mode_hz) == 2
    assert mode_hz[0] == pytest.approx(1.4, rel=0.25)
    assert mode_hz[1] == pytest.approx(10.8, rel=0.25)


def test_simulate_rejects_bad_input_shape(default_vehicle) -> None:
    plant = QuarterCar(default_vehicle)
    with pytest.raises(ValueError, match="u_seq"):
        plant.simulate([0.0, 0.1, 0.2], u_seq=[0.0, 0.0], w_seq=0.0)
