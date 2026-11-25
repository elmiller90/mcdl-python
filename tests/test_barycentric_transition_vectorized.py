import time
import numpy as np

from barycentric_transition import (
    barycentric_transition_value_grad,
)
from barycentric_transition_vectorized import (
    barycentric_transition_value_grad_vectorized,
)


def test_vectorized_matches_reference():
    rng = np.random.default_rng(150)

    L, K, J, T = 17, 2, 5, 8
    X = 0.01+rng.random((L, K, J, T))

    shifts = np.arange(L, dtype=float)
    scales = np.array(
        [0.2, 0.25, 1/3, 0.5, 1.0],
        dtype=float,
    )

    kwargs = dict(
        shift_coordinates=shifts,
        scale_coordinates=scales,
        gate_mass=0.05,
        mass_floor=1e-10,
        shift_velocity=1.0,
        scale_velocity=None,
        spread_weight=0.10,
    )

    v0, g0, d0 = barycentric_transition_value_grad(
        X, **kwargs
    )
    v1, g1, d1 = (
        barycentric_transition_value_grad_vectorized(
            X, **kwargs
        )
    )

    verr = abs(v1-v0)
    gerr = np.linalg.norm(g1-g0) / max(
        np.linalg.norm(g0), 1e-12
    )

    print(f"value error: {verr:.3e}")
    print(f"gradient relative error: {gerr:.3e}")

    assert verr < 1e-10
    assert gerr < 1e-10

    for key in (
        "masses",
        "frame_gates",
        "position_centroids",
        "log_scale_centroids",
        "pair_weights",
        "pair_values",
        "frame_spread",
    ):
        assert np.allclose(
            d0[key], d1[key],
            rtol=1e-11, atol=1e-12,
        )


def test_banco_size_benchmark():
    rng = np.random.default_rng(151)

    L, K, J, T = 120, 2, 10, 80
    X = 0.001+rng.random((L, K, J, T))

    shifts = np.arange(L, dtype=float)
    scales = np.array([
        1/5, 1/4, 1/3, 2/5, 1/2,
        3/5, 2/3, 3/4, 4/5, 1.0,
    ])

    kwargs = dict(
        shift_coordinates=shifts,
        scale_coordinates=scales,
        gate_mass=0.05,
        mass_floor=1e-10,
        shift_velocity=1.0,
        scale_velocity=None,
        spread_weight=0.10,
    )

    # Warm-up.
    barycentric_transition_value_grad(X, **kwargs)
    barycentric_transition_value_grad_vectorized(
        X, **kwargs
    )

    t0 = time.perf_counter()
    for _ in range(5):
        barycentric_transition_value_grad(
            X, **kwargs
        )
    tref = (time.perf_counter()-t0)/5

    t0 = time.perf_counter()
    for _ in range(20):
        barycentric_transition_value_grad_vectorized(
            X, **kwargs
        )
    tvec = (time.perf_counter()-t0)/20

    print(f"reference:  {tref:.6f} s")
    print(f"vectorized: {tvec:.6f} s")
    print(f"speedup:    {tref/tvec:.2f}x")
