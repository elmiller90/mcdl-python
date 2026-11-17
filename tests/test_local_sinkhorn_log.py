import numpy as np

from local_graph import (
    build_local_transport_graph,
    median_log_scale_spacing,
    state_index,
)
from local_sinkhorn_log import (
    local_sinkhorn_log,
    local_sinkhorn_log_epsilon_scaling,
)


SCALES = np.array([
    1.0, 0.8, 0.75, 2.0/3.0, 0.5
])


def delta(P, i):
    x = np.zeros(P)
    x[i] = 1.0
    return x


def test_delta_geometry():
    L, J = 136, len(SCALES)
    vs = median_log_scale_spacing(SCALES)

    graph = build_local_transport_graph(
        L, SCALES,
        shift_radius=3,
        scale_radius=1,
        scale_velocity=vs,
    )

    n, j = 50, 2
    p = state_index(n, j, J)

    cases = [
        ("self", p, p, 0.0),
        (
            "shift",
            p,
            state_index(n+1, j, J),
            1.0,
        ),
        (
            "scale",
            p,
            state_index(n, j+1, J),
            (
                (
                    np.log(SCALES[j+1])
                    - np.log(SCALES[j])
                ) / vs
            ) ** 2,
        ),
    ]

    for name, a, b, expected in cases:
        result = local_sinkhorn_log(
            delta(graph.P, a),
            delta(graph.P, b),
            graph,
            epsilon=.05,
            tol=1e-12,
            maxiter=100,
            check_every=1,
        )

        print(
            f"{name:8s} "
            f"cost={result['geometric_cost']:.12f} "
            f"iter={result['info']['iterations']} "
            f"conv={result['info']['converged']} "
            f"error={result['info']['marginal_error']:.3e}"
        )

        assert result["info"]["converged"]
        assert np.isclose(
            result["geometric_cost"],
            expected,
            atol=1e-10,
        )


def test_smooth_identity():
    graph = build_local_transport_graph(
        136,
        SCALES,
        shift_radius=3,
        scale_radius=1,
    )

    rng = np.random.default_rng(200)
    p = rng.random(graph.P) + .1
    p /= p.sum()

    result = local_sinkhorn_log_epsilon_scaling(
        p,
        p,
        graph,
        schedule=(1.0, .5, .2, .1),
        tolerances=(1e-3, 1e-4, 1e-5, 1e-6),
        maxiter=2000,
        check_every=10,
    )

    print()
    print("Log-domain epsilon scaling")
    print("--------------------------")
    print("epsilon   iter   conv   marginal")

    for h in result["epsilon_scaling_history"]:
        print(
            f"{h['epsilon']:7.3f}  "
            f"{h['iterations']:5d}  "
            f"{str(h['converged']):5s}  "
            f"{h['marginal_error']:9.2e}"
        )

    print(
        "total iterations:",
        result["epsilon_scaling_info"]["total_iterations"],
    )

    print(
        "final error:",
        result["info"]["marginal_error"],
    )


if __name__ == "__main__":
    test_delta_geometry()
    test_smooth_identity()

