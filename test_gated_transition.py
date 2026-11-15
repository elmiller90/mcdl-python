import numpy as np

from gated_transition import gated_transition_value_grad
from local_graph import build_local_transport_graph
from local_transition import build_transition_kernel


SCALES = np.array([
    1.0, 0.8, 0.75, 2.0/3.0, 0.5
])

L = 18
K = 2
J = len(SCALES)
T = 6

EPSILON = 0.5
GATE_MASS = 0.05


def finite_difference_test():
    graph = build_local_transport_graph(
        L=L,
        scales=SCALES,
        shift_radius=3,
        scale_radius=1,
    )

    prob = build_transition_kernel(
        graph,
        epsilon=EPSILON,
    )

    rng = np.random.default_rng(500)

    X = rng.random((L, K, J, T)) + 0.2

    value, grad, diagnostics = (
        gated_transition_value_grad(
            X,
            graph,
            prob,
            gate_mass=GATE_MASS,
        )
    )

    D = rng.standard_normal(X.shape)
    D /= np.linalg.norm(D)

    # X is safely positive, so these h values remain feasible.
    analytic = float(np.sum(grad * D))

    def fun(h):
        return gated_transition_value_grad(
            X + h*D,
            graph,
            prob,
            gate_mass=GATE_MASS,
        )[0]

    print()
    print("Gated tensor gradient")
    print("---------------------")
    print(f"value:    {value:.8e}")
    print(f"analytic: {analytic:.8e}")
    print("h          numeric       rel.error")

    best = np.inf

    for h in [
        1e-2, 3e-3, 1e-3,
        3e-4, 1e-4, 3e-5,
    ]:
        numeric = (fun(h) - fun(-h)) / (2*h)

        rel = abs(numeric - analytic) / max(
            1.0,
            abs(numeric),
            abs(analytic),
        )

        best = min(best, rel)

        print(
            f"{h:9.2e} "
            f"{numeric:13.6e} "
            f"{rel:10.3e}"
        )

    print()
    print(
        f"mass range: "
        f"{diagnostics['masses'].min():.3f} "
        f"to {diagnostics['masses'].max():.3f}"
    )
    print(
        f"gate range: "
        f"{diagnostics['frame_gates'].min():.6f} "
        f"to {diagnostics['frame_gates'].max():.6f}"
    )

    assert best < 1e-7
    assert grad.shape == X.shape

    print("gated tensor gradient: PASS")


def gating_test():
    graph = build_local_transport_graph(
        L=L,
        scales=SCALES,
        shift_radius=3,
        scale_radius=1,
    )

    prob = build_transition_kernel(
        graph,
        epsilon=EPSILON,
    )

    X = np.zeros((L, 1, J, 5))

    # Frames 1,2,3 active; frames 0 and 4 inactive.
    X[7, 0, 2, 1] = 1.0
    X[8, 0, 2, 2] = 1.0
    X[9, 0, 2, 3] = 1.0

    value, grad, d = gated_transition_value_grad(
        X,
        graph,
        prob,
        gate_mass=GATE_MASS,
        mass_floor=1e-10,
    )

    print()
    print("Finite-track smooth gating")
    print("--------------------------")
    print("masses:       ", d["masses"][0])
    print("gates:        ", d["frame_gates"][0])
    print("pair weights: ", d["pair_weights"][0])
    print(
        "weighted vals:",
        d["weighted_pair_values"][0],
    )
    print(f"value: {value:.8e}")

    assert d["number_pairs"] == 4

    # Birth/death pairs have zero weight smoothly.
    assert np.isclose(
        d["pair_weights"][0, 0],
        0.0,
    )

    assert np.isclose(
        d["pair_weights"][0, 3],
        0.0,
    )

    # Active-active pairs remain strongly weighted.
    assert d["pair_weights"][0, 1] > 0.9
    assert d["pair_weights"][0, 2] > 0.9

    # Birth/death pairs contribute zero weighted penalty.
    assert np.isclose(
        d["weighted_pair_values"][0, 0],
        0.0,
        atol=1e-14,
    )

    assert np.isclose(
        d["weighted_pair_values"][0, 3],
        0.0,
        atol=1e-14,
    )
    ####
    # The weighted penalty is exactly zero across birth/death
    # boundaries because g(0)=0.
    assert np.isclose(
        d["weighted_pair_values"][0, 0],
        0.0,
        atol=1e-14,
    )

    assert np.isclose(
        d["weighted_pair_values"][0, 3],
        0.0,
        atol=1e-14,
    )

    # In the smooth formulation the gradient at zero mass is
    # generally nonzero because g'(0)=1/gate_mass. This is what
    # allows L-BFGS to enter or leave the active state smoothly.
    assert np.all(np.isfinite(
        grad[:, 0, :, 0]
    ))

    assert np.all(np.isfinite(
        grad[:, 0, :, 4]
    ))

    print(
        "birth gradient norm:",
        np.linalg.norm(
            grad[:, 0, :, 0]
        ),
    )

    print(
        "death gradient norm:",
        np.linalg.norm(
            grad[:, 0, :, 4]
        ),
    )
    ####
    print("finite-track smooth gating: PASS")


def gate_values_test():
    graph = build_local_transport_graph(
        L=L,
        scales=SCALES,
        shift_radius=3,
        scale_radius=1,
    )

    prob = build_transition_kernel(
        graph,
        epsilon=EPSILON,
    )

    X = np.zeros((L, 1, J, 3))

    # Low-mass, full-mass, low-mass.
    X[7, 0, 2, 0] = 1e-3
    X[8, 0, 2, 1] = 1.0
    X[9, 0, 2, 2] = 1e-3

    _, _, d = gated_transition_value_grad(
        X,
        graph,
        prob,
        gate_mass=GATE_MASS,
    )

    expected_low = 1e-6 / (1e-6 + GATE_MASS**2)
    expected_full = 1.0 / (1.0 + GATE_MASS**2)

    assert np.isclose(
        d["frame_gates"][0, 0],
        expected_low,
    )

    assert np.isclose(
        d["frame_gates"][0, 1],
        expected_full,
    )

    print()
    print("Gate values")
    print("-----------")
    print(f"low mass gate:  {expected_low:.6f}")
    print(f"unit mass gate: {expected_full:.6f}")
    print("gate values: PASS")

def smooth_low_mass_gradient_test():
    graph = build_local_transport_graph(
        L=L,
        scales=SCALES,
        shift_radius=3,
        scale_radius=1,
    )

    prob = build_transition_kernel(
        graph,
        epsilon=EPSILON,
    )

    rng = np.random.default_rng(900)

    X = 1e-8 * (
        0.2 + rng.random((L, 1, J, 3))
    )

    direction = rng.standard_normal(X.shape)
    direction /= np.linalg.norm(direction)

    # Keep centered perturbations nonnegative.
    max_h = np.min(
        X / np.maximum(
            np.abs(direction),
            1e-300,
        )
    )

    value, grad, diagnostics = (
        gated_transition_value_grad(
            X,
            graph,
            prob,
            gate_mass=GATE_MASS,
            mass_floor=1e-10,
        )
    )

    analytic = float(
        np.sum(grad * direction)
    )

    def fun(h):
        return gated_transition_value_grad(
            X + h * direction,
            graph,
            prob,
            gate_mass=GATE_MASS,
            mass_floor=1e-10,
        )[0]

    print()
    print("Smooth low-mass gradient")
    print("------------------------")
    print(f"value:    {value:.8e}")
    print(f"analytic: {analytic:.8e}")
    print("h          numeric       rel.error")

    best = np.inf

    for fraction in [
        1e-1,
        3e-2,
        1e-2,
        3e-3,
        1e-3,
    ]:
        h = fraction * max_h

        numeric = (
            fun(h) - fun(-h)
        ) / (2 * h)

        rel = abs(
            numeric - analytic
        ) / max(
            1.0,
            abs(numeric),
            abs(analytic),
        )

        best = min(best, rel)

        print(
            f"{h:9.2e} "
            f"{numeric:13.6e} "
            f"{rel:10.3e}"
        )

    print(
        "mass range:",
        diagnostics["masses"].min(),
        diagnostics["masses"].max(),
    )

    print(
        "gate range:",
        diagnostics["frame_gates"].min(),
        diagnostics["frame_gates"].max(),
    )

    assert np.all(np.isfinite(grad))
    assert best < 1e-6

    print("smooth low-mass gradient: PASS")

if __name__ == "__main__":
    finite_difference_test()
    gating_test()
    gate_values_test()

    print()
    print("ALL GATED TRANSITION TESTS PASSED")