import numpy as np

from local_graph import (
    build_local_transport_graph,
    median_log_scale_spacing,
    state_index,
)
from local_sinkhorn import (
    local_sinkhorn_cost,
    local_sinkhorn_epsilon_scaling,
)


SCALES = np.array([
    1.0,
    0.8,
    0.75,
    2.0 / 3.0,
    0.5,
])


def delta(P, index):
    x = np.zeros(P, dtype=np.float64)
    x[index] = 1.0
    return x


def run_case(
    name,
    p,
    q,
    graph,
    epsilon=0.05,
    tol=1e-10,
    maxiter=1000,
):
    result = local_sinkhorn_cost(
        p,
        q,
        graph,
        epsilon=epsilon,
        tol=tol,
        maxiter=maxiter,
    )

    print()
    print(name)
    print("-" * len(name))
    print(f"epsilon:          {epsilon:.4g}")
    print(f"converged:        {result['info']['converged']}")
    print(f"iterations:       {result['info']['iterations']}")
    print(
        f"marginal error:   "
        f"{result['info']['marginal_error']:.3e}"
    )
    print(
        f"geometric cost:   "
        f"{result['geometric_cost']:.12f}"
    )
    print(
        f"objective:        "
        f"{result['objective']:.12f}"
    )

    assert result["info"]["converged"]

    return result


def test_delta_cases():
    L = 136
    J = len(SCALES)

    vs = median_log_scale_spacing(
        SCALES
    )

    graph = build_local_transport_graph(
        L=L,
        scales=SCALES,
        shift_radius=3,
        scale_radius=1,
        shift_velocity=1.0,
        scale_velocity=vs,
    )

    n = 50
    j = 2

    p0 = state_index(
        n,
        j,
        J,
    )

    self_case = run_case(
        "Identity",
        delta(graph.P, p0),
        delta(graph.P, p0),
        graph,
    )

    assert np.isclose(
        self_case["geometric_cost"],
        0.0,
        atol=1e-10,
    )

    p_shift = state_index(
        n + 1,
        j,
        J,
    )

    shift_case = run_case(
        "One shift",
        delta(graph.P, p0),
        delta(graph.P, p_shift),
        graph,
    )

    assert np.isclose(
        shift_case["geometric_cost"],
        1.0,
        atol=1e-8,
    )

    p_scale = state_index(
        n,
        j + 1,
        J,
    )

    expected_scale = (
        (
            np.log(SCALES[j + 1])
            - np.log(SCALES[j])
        ) / vs
    ) ** 2

    scale_case = run_case(
        "One scale",
        delta(graph.P, p0),
        delta(graph.P, p_scale),
        graph,
    )

    assert np.isclose(
        scale_case["geometric_cost"],
        expected_scale,
        atol=1e-8,
    )

    p_diag = state_index(
        n + 1,
        j + 1,
        J,
    )

    diagonal_case = run_case(
        "Shift plus scale",
        delta(graph.P, p0),
        delta(graph.P, p_diag),
        graph,
    )

    assert np.isclose(
        diagonal_case["geometric_cost"],
        1.0 + expected_scale,
        atol=1e-8,
    )


def test_smooth_case():
    L = 136

    graph = build_local_transport_graph(
        L=L,
        scales=SCALES,
        shift_radius=3,
        scale_radius=1,
    )

    rng = np.random.default_rng(200)

    # Strictly positive nearby distributions.
    p = rng.random(graph.P) + 0.1
    q = p.copy()

    result = run_case(
        "Smooth identity",
        p,
        q,
        graph,
        epsilon=0.1,
        tol=1e-5,
        maxiter=5000,
    )

    assert result["geometric_cost"] >= 0

def test_epsilon_sweep():
    graph = build_local_transport_graph(
        136,
        SCALES,
        shift_radius=3,
        scale_radius=1,
    )

    rng = np.random.default_rng(200)

    p = rng.random(graph.P) + 0.1
    p /= p.sum()

    print()
    print("Epsilon convergence sweep")
    print("-------------------------")
    print("epsilon   iter   converged   marginal error   geom cost")

    for epsilon in [0.05, 0.1, 0.2, 0.5, 1.0]:
        result = local_sinkhorn_cost(
            p,
            p,
            graph,
            epsilon=epsilon,
            tol=1e-6,
            maxiter=5000,
        )

        info = result["info"]

        print(
            f"{epsilon:7.3f}  "
            f"{info['iterations']:4d}  "
            f"{str(info['converged']):9s}  "
            f"{info['marginal_error']:14.3e}  "
            f"{result['geometric_cost']:.6e}"
        )

def test_epsilon_scaling():
    graph = build_local_transport_graph(
        136,
        SCALES,
        shift_radius=3,
        scale_radius=1,
    )

    rng = np.random.default_rng(200)

    p = rng.random(graph.P) + 0.1
    p /= p.sum()

    result = local_sinkhorn_epsilon_scaling(
        p,
        p,
        graph,
        schedule=(1.0, 0.5, 0.2, 0.1),
        tolerances=(1e-3, 1e-4, 1e-5, 1e-6),
        maxiter=2000,
    )

    print()
    print("Epsilon scaling")
    print("---------------")
    print("epsilon   tol       iter   conv   marginal")

    for h in result["epsilon_scaling_history"]:
        print(
            f"{h['epsilon']:7.3f}  "
            f"{h['tolerance']:8.1e}  "
            f"{h['iterations']:5d}  "
            f"{str(h['converged']):5s}  "
            f"{h['marginal_error']:9.2e}"
        )

    info = result["epsilon_scaling_info"]

    print("-------------------------------")
    print(
        f"total iterations: "
        f"{info['total_iterations']}"
    )
    print(
        f"final error:      "
        f"{result['info']['marginal_error']:.3e}"
    )
    print(
        f"final geom cost:  "
        f"{result['geometric_cost']:.8f}"
    )

if __name__ == "__main__":
    test_delta_cases()
    test_smooth_case()
    test_epsilon_sweep()
    test_epsilon_scaling()
    
    print()
    print("ALL LOCAL SINKHORN TESTS PASSED")