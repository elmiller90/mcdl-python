import numpy as np

from banco_rho_update import banco_rho_update


def test_increase_and_rescale():
    z = np.ones(3)
    rho, z = banco_rho_update(
        10.0, primal=20.0, dual=1.0,
        scaled_dual=z,
    )
    assert rho == 20.0
    assert np.allclose(z, 0.5)


def test_decrease_and_rescale():
    z = np.ones(3)
    rho, z = banco_rho_update(
        10.0, primal=1.0, dual=20.0,
        scaled_dual=z,
    )
    assert rho == 5.0
    assert np.allclose(z, 2.0)


def test_unchanged():
    z = np.ones(3)
    rho, z = banco_rho_update(
        10.0, primal=5.0, dual=1.0,
        scaled_dual=z,
    )
    assert rho == 10.0
    assert np.allclose(z, 1.0)


def test_x_dual_floor_boost():
    z = np.ones(3)
    rho, z = banco_rho_update(
        10.0, primal=1.0, dual=0.0,
        scaled_dual=z,
        dual_floor_boost=True,
    )
    assert rho == 100.0
    assert np.allclose(z, 0.1)