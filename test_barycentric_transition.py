import numpy as np

from barycentric_transition import (
    barycentric_transition_value_grad,
)
from local_graph import physical_scale_values


SCALES = [
    (1, 1), (4, 5), (3, 4), (2, 3), (1, 2)
]

L, K, J, T = 18, 2, 5, 6
GATE_MASS = 0.05
MASS_FLOOR = 1e-10


def make_args():
    return {
        "shift_coordinates": np.arange(L, dtype=float),
        "scale_coordinates": physical_scale_values(SCALES),
        "gate_mass": GATE_MASS,
        "mass_floor": MASS_FLOOR,
        "shift_velocity": 1.0,
    }


def gradient_test():
    rng = np.random.default_rng(1200)
    X = 0.2 + rng.random((L, K, J, T))
    D = rng.standard_normal(X.shape)
    D /= np.linalg.norm(D)

    value, grad, d = barycentric_transition_value_grad(
        X, **make_args()
    )

    analytic = float(np.sum(grad * D))

    def fun(h):
        return barycentric_transition_value_grad(
            X + h * D, **make_args()
        )[0]

    print()
    print("Barycentric tensor gradient")
    print("---------------------------")
    print(f"value:    {value:.8e}")
    print(f"analytic: {analytic:.8e}")
    print("h          numeric       rel.error")

    best = np.inf

    for h in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]:
        numeric = (fun(h) - fun(-h)) / (2 * h)
        rel = abs(numeric - analytic) / max(
            1.0, abs(numeric), abs(analytic)
        )
        best = min(best, rel)

        print(
            f"{h:9.2e} "
            f"{numeric:13.6e} "
            f"{rel:10.3e}"
        )

    print(
        "position centroid range:",
        d["position_centroids"].min(),
        d["position_centroids"].max(),
    )

    assert grad.shape == X.shape
    assert np.all(np.isfinite(grad))
    assert best < 1e-7

    print("barycentric tensor gradient: PASS")


def path_geometry_test():
    X = np.zeros((L, 1, J, 5))

    # Finite track moving one shift and one scale per frame.
    X[6, 0, 1, 1] = 1.0
    X[7, 0, 2, 2] = 1.0
    X[8, 0, 3, 3] = 1.0

    value, grad, d = barycentric_transition_value_grad(
        X, **make_args()
    )

    print()
    print("Finite barycentric track")
    print("------------------------")
    print("masses:   ", d["masses"][0])
    print("gates:    ", d["frame_gates"][0])
    print("positions:", d["position_centroids"][0])
    print("log scale:", d["log_scale_centroids"][0])
    print("weights:  ", d["pair_weights"][0])
    print("values:   ", d["pair_values"][0])
    print(f"total:     {value:.8e}")

    assert np.isclose(d["pair_weights"][0, 0], 0.0)
    assert np.isclose(d["pair_weights"][0, 3], 0.0)
    assert d["pair_weights"][0, 1] > 0.99
    assert d["pair_weights"][0, 2] > 0.99

    # Squared gate has zero derivative at zero mass.
    assert np.allclose(grad[:, 0, :, 0], 0.0, atol=1e-12)
    assert np.allclose(grad[:, 0, :, 4], 0.0, atol=1e-12)

    print("finite barycentric track: PASS")


def stationary_track_test():
    X = np.zeros((L, 1, J, 4))
    X[9, 0, 2, :] = 1.0

    value, grad, d = barycentric_transition_value_grad(
        X, **make_args()
    )

    print()
    print("Stationary track")
    print("----------------")
    print(f"value: {value:.8e}")

    assert np.isclose(value, 0.0, atol=1e-12)
    assert np.allclose(grad, 0.0, atol=1e-10)

    print("stationary track: PASS")


def no_diffusion_test():
    """
    Two frames with a single moving coefficient should remain
    one-sparse. The regularizer evaluates centroid motion only.
    """
    X = np.zeros((L, 1, J, 2))
    X[7, 0, 2, 0] = 1.0
    X[8, 0, 2, 1] = 1.0

    value, grad, d = barycentric_transition_value_grad(
        X, **make_args()
    )

    assert np.count_nonzero(X[:, 0, :, 0]) == 1
    assert np.count_nonzero(X[:, 0, :, 1]) == 1

    expected = d["pair_weights"][0, 0] * 1.0
    assert np.isclose(value, expected, rtol=1e-10)

    print()
    print("One-step sparse motion")
    print("----------------------")
    print(f"value:    {value:.8e}")
    print(f"expected: {expected:.8e}")
    print("no diffusion geometry: PASS")


if __name__ == "__main__":
    gradient_test()
    path_geometry_test()
    stationary_track_test()
    no_diffusion_test()

    print()
    print("ALL BARYCENTRIC TRANSITION TESTS PASSED")