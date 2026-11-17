import numpy as np

from local_graph import (
    build_local_transport_graph,
    state_index,
)
from local_transition import (
    apply_transition,
    build_transition_kernel,
    normalized_transition_value_grad,
    transition_value_grad,
)


SCALES = np.array([
    1.0, 0.8, 0.75, 2.0/3.0, 0.5
])

L = 136
EPSILON = 0.5


def delta(P, i):
    x = np.zeros(P)
    x[i] = 1.0
    return x


def directional_derivative(
    fun,
    x,
    direction,
    h,
):
    return (
        fun(x + h*direction)
        - fun(x - h*direction)
    ) / (2*h)


def test_kernel():
    graph = build_local_transport_graph(
        L,
        SCALES,
        shift_radius=3,
        scale_radius=1,
    )

    prob = build_transition_kernel(
        graph,
        epsilon=EPSILON,
    )

    row_sum = np.bincount(
        graph.src,
        weights=prob,
        minlength=graph.P,
    )

    assert np.allclose(
        row_sum,
        1.0,
        atol=1e-12,
    )

    print("row-stochastic kernel: PASS")

    return graph, prob


def test_mass_conservation(graph, prob):
    rng = np.random.default_rng(100)

    p = rng.random(graph.P)
    p /= p.sum()

    q = apply_transition(
        graph,
        prob,
        p,
    )

    assert np.all(q >= 0)
    assert np.isclose(
        q.sum(),
        1.0,
        atol=1e-12,
    )

    print("transition mass conservation: PASS")


def test_local_geometry(graph, prob):
    J = graph.J
    n, j = 50, 2

    p = delta(
        graph.P,
        state_index(n, j, J),
    )

    q = apply_transition(
        graph,
        prob,
        p,
    )

    center = state_index(
        n, j, J
    )

    shift = state_index(
        n+1, j, J
    )

    far = state_index(
        n+4, j, J
    )

    print()
    print("Local transition geometry")
    print("-------------------------")
    print(f"center probability: {q[center]:.6f}")
    print(f"+1 shift prob:      {q[shift]:.6f}")
    print(f"+4 shift prob:      {q[far]:.6f}")

    assert q[center] > q[shift]
    assert q[far] == 0.0

    print("local geometry: PASS")


def test_probability_gradient(graph, prob):
    rng = np.random.default_rng(200)

    # Positive normalized distributions avoid boundary
    # complications in this first gradient test.
    p = rng.random(graph.P) + .2
    q = rng.random(graph.P) + .2

    p /= p.sum()
    q /= q.sum()

    value, gp, gq, _ = transition_value_grad(
        p, q, graph, prob
    )

    dp = rng.standard_normal(graph.P)
    dq = rng.standard_normal(graph.P)

    dp -= dp.mean()
    dq -= dq.mean()

    dp /= np.linalg.norm(dp)
    dq /= np.linalg.norm(dq)

    # Keep perturbations inside positive orthant.
    max_h = min(
        .25*np.min(p/np.maximum(np.abs(dp), 1e-300)),
        .25*np.min(q/np.maximum(np.abs(dq), 1e-300)),
    )

    analytic = (
        np.dot(gp, dp)
        + np.dot(gq, dq)
    )

    def fun(h):
        return transition_value_grad(
            p + h*dp,
            q + h*dq,
            graph,
            prob,
        )[0]

    print()
    print("Probability gradient test")
    print("-------------------------")
    print(f"value:    {value:.8e}")
    print(f"analytic: {analytic:.8e}")
    print("h          numeric       rel.error")

    best = np.inf

    for base_h in [
        1e-3, 3e-4, 1e-4,
        3e-5, 1e-5, 3e-6,
    ]:
        h = min(
            base_h,
            .5*max_h,
        )

        numeric = (
            fun(h) - fun(-h)
        ) / (2*h)

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

    assert best < 1e-6

    print("probability gradient: PASS")


def test_normalized_gradient(graph, prob):
    rng = np.random.default_rng(300)

    x = rng.random(graph.P) + .2
    y = rng.random(graph.P) + .2

    value, gx, gy, _ = normalized_transition_value_grad(
        x, y, graph, prob
    )

    dx = rng.standard_normal(graph.P)
    dy = rng.standard_normal(graph.P)
    dx /= np.linalg.norm(dx)
    dy /= np.linalg.norm(dy)

    analytic = np.dot(gx, dx) + np.dot(gy, dy)

    def fun(h):
        return normalized_transition_value_grad(
            x + h*dx,
            y + h*dy,
            graph,
            prob,
        )[0]

    print()
    print("Normalized coefficient gradient")
    print("-------------------------------")
    print(f"value:    {value:.8e}")
    print(f"analytic: {analytic:.8e}")
    print("h          numeric       rel.error")

    best = np.inf

    for h in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]:
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

    assert best < 1e-7

    print("normalized gradient: PASS")


if __name__ == "__main__":
    graph, prob = test_kernel()
    test_mass_conservation(graph, prob)
    test_local_geometry(graph, prob)
    test_probability_gradient(graph, prob)
    test_normalized_gradient(graph, prob)

    print()
    print("ALL LOCAL TRANSITION TESTS PASSED")
