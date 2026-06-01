from __future__ import annotations

import numpy as np

from risk_aware_active_suspension.plants.carsim_fmu import CarSimFmuPlant


def test_carsim_input_mapping_uses_project_corner_order(tmp_path):
    fmu_path = tmp_path / "dummy.fmu"
    fmu_path.write_bytes(b"dummy")
    plant = CarSimFmuPlant(fmu_path)
    t = np.array([0.0, 0.005, 0.010])

    inputs = plant._build_input_array(
        t,
        speed=80.0,
        steer=np.array([0.0, 1.0, 2.0]),
        active_force=np.array([10.0, 20.0, 30.0, 40.0]),
        brake_pressure=0.0,
        road_height=np.zeros((3, 4)),
        mu_x=np.array([0.8, 0.7, 0.6, 0.5]),
        mu_y=0.9,
    )

    assert inputs["time"].tolist() == [0.0, 0.005, 0.010]
    assert inputs["IMP_STEER_SW"].tolist() == [0.0, 1.0, 2.0]
    assert inputs["IMP_SPEED"].tolist() == [80.0, 80.0, 80.0]
    assert inputs["IMP_FZEXL1"].tolist() == [10.0, 10.0, 10.0]
    assert inputs["IMP_FZEXR1"].tolist() == [20.0, 20.0, 20.0]
    assert inputs["IMP_FZEXL2"].tolist() == [30.0, 30.0, 30.0]
    assert inputs["IMP_FZEXR2"].tolist() == [40.0, 40.0, 40.0]
    assert inputs["IMP_MUX_L1"].tolist() == [0.8, 0.8, 0.8]
    assert inputs["IMP_MUX_R1"].tolist() == [0.7, 0.7, 0.7]
    assert inputs["IMP_MUX_L2"].tolist() == [0.6, 0.6, 0.6]
    assert inputs["IMP_MUX_R2"].tolist() == [0.5, 0.5, 0.5]
