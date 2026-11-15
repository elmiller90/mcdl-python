import numpy as np

from gaussian_experiment import SCALES, make_dictionary
from gated_transition import gated_transition_value_grad
from local_graph import (
    build_local_transport_graph,
    physical_scale_values,
)
from local_transition import build_transition_kernel
from multiscale_convolution import MultiscaleConvolutionOperator


M, N, T = 41, 48, 5
K = 1

EPSILON = 0.5
GATE_MASS = 0.05
LAMBDA_T = 0.08
RHO = 0.10

SEED = 700


def make_problem():
    rng = np.random.default_rng(SEED)

    # Full scale specification used by the convolution operator.
    full_scales = np.asarray(SCALES)

    # One scalar physical scale coordinate per scale level
    # for the local shift-scale transition graph.
    transport_scales = physical_scale_values(full_scales)

    D = make_dictionary(M, 4.0)

    A = MultiscaleConvolutionOperator(
        D=D,
        scales=full_scales,
        center=(M + 1) // 2,
        N=N,
        normalize=True,
    )

    graph = build_local_transport_graph(
        L=A.L,
        scales=transport_scales,
        shift_radius=3,
        scale_radius=1,
        shift_velocity=1.0,
    )

    prob = build_transition_kernel(
        graph,
        epsilon=EPSILON,
    )

    shape = (
        A.L,
        K,
        len(transport_scales),
        T,
    )

    # Keep X safely positive for centered finite differences.
    X = 0.1 + rng.random(shape)
    Z = 0.1 + rng.random(shape)
    U = 0.05 * rng.standard_normal(shape)

    # Arbitrary synthetic coefficients are sufficient for
    # validating the objective gradient.
    X_signal = rng.random(shape)
    S = A.forward(X_signal)

    print("A.L:", A.L)
    print("full SCALES shape:", full_scales.shape)
    print("transport scales:", transport_scales)
    print("transport scales shape:", transport_scales.shape)
    print("graph.L/J/P:", graph.L, graph.J, graph.P)
    print("src shape/dtype:", graph.src.shape, graph.src.dtype)
    print("dst shape/dtype:", graph.dst.shape, graph.dst.dtype)
    print("cost shape/dtype:", graph.cost.shape, graph.cost.dtype)

    return rng, A, graph, prob, X, Z, U, S

def objective_value_grad(
    X,
    A,
    S,
    Z,
    U,
    graph,
    prob,
):
    # -------------------------------------
    # Data fidelity
    # -------------------------------------

    residual = A.forward(X) - S

    data_value = 0.5 * np.sum(
        residual * residual
    )

    data_grad = A.adjoint(
        residual,
        structured=True,
    )

    # -------------------------------------
    # ADMM quadratic
    # -------------------------------------

    Q = X - Z + U

    admm_value = 0.5 * RHO * np.sum(
        Q * Q
    )

    admm_grad = RHO * Q

    # -------------------------------------
    # Gated temporal regularizer
    # -------------------------------------

    temporal_value, temporal_grad, diagnostics = (
        gated_transition_value_grad(
            X,
            graph,
            prob,
            gate_mass=GATE_MASS,
        )
    )

    # -------------------------------------
    # Total objective and gradient
    # -------------------------------------

    value = (
        data_value
        + admm_value
        + LAMBDA_T * temporal_value
    )

    grad = (
        data_grad
        + admm_grad
        + LAMBDA_T * temporal_grad
    )

    parts = {
        "data": float(data_value),
        "admm": float(admm_value),
        "temporal": float(temporal_value),
        "weighted_temporal":
            float(LAMBDA_T * temporal_value),
        "masses": diagnostics["masses"],
        "gates": diagnostics["frame_gates"],
    }

    return float(value), grad, parts


def test_objective_gradient():
    rng, A, graph, prob, X, Z, U, S = (
        make_problem()
    )

    value, grad, parts = objective_value_grad(
        X,
        A,
        S,
        Z,
        U,
        graph,
        prob,
    )

    direction = rng.standard_normal(
        X.shape
    )

    direction /= np.linalg.norm(
        direction
    )

    analytic = float(
        np.sum(
            grad * direction
        )
    )

    def fun(h):
        return objective_value_grad(
            X + h * direction,
            A,
            S,
            Z,
            U,
            graph,
            prob,
        )[0]

    print()
    print("Gated objective gradient")
    print("------------------------")
    print(f"objective:         {value:.10e}")
    print(f"data term:         {parts['data']:.10e}")
    print(f"ADMM term:         {parts['admm']:.10e}")
    print(f"temporal term:     {parts['temporal']:.10e}")
    print(
        f"weighted temporal: "
        f"{parts['weighted_temporal']:.10e}"
    )
    print(f"analytic dir.der.: {analytic:.10e}")
    print()
    print("h          numeric          rel.error")

    best = np.inf

    for h in [
        1e-2,
        3e-3,
        1e-3,
        3e-4,
        1e-4,
        3e-5,
        1e-5,
    ]:
        numeric = (
            fun(h) - fun(-h)
        ) / (2.0 * h)

        rel = abs(
            numeric - analytic
        ) / max(
            1.0,
            abs(numeric),
            abs(analytic),
        )

        best = min(
            best,
            rel,
        )

        print(
            f"{h:9.2e} "
            f"{numeric:16.9e} "
            f"{rel:11.3e}"
        )

    print()
    print(
        f"mass range: "
        f"{parts['masses'].min():.4f} "
        f"to {parts['masses'].max():.4f}"
    )

    print(
        f"gate range: "
        f"{parts['gates'].min():.6f} "
        f"to {parts['gates'].max():.6f}"
    )

    print(
        f"best relative error: "
        f"{best:.3e}"
    )

    assert grad.shape == X.shape
    assert np.all(np.isfinite(grad))
    assert best < 1e-6

    print()
    print("FULL GATED OBJECTIVE GRADIENT: PASS")


def test_term_gradients():
    """
    Useful diagnostic if the complete test ever fails:
    verify the three gradient contributions independently.
    """
    rng, A, graph, prob, X, Z, U, S = (
        make_problem()
    )

    direction = rng.standard_normal(
        X.shape
    )

    direction /= np.linalg.norm(
        direction
    )

    # -------------------------------------
    # Data term
    # -------------------------------------

    residual = A.forward(X) - S

    data_grad = A.adjoint(
        residual,
        structured=True,
    )

    analytic_data = float(
        np.sum(
            data_grad * direction
        )
    )

    def data_fun(h):
        R = A.forward(
            X + h * direction
        ) - S

        return 0.5 * np.sum(
            R * R
        )

    # -------------------------------------
    # ADMM term
    # -------------------------------------

    admm_grad = RHO * (
        X - Z + U
    )

    analytic_admm = float(
        np.sum(
            admm_grad * direction
        )
    )

    def admm_fun(h):
        Q = (
            X + h * direction
            - Z
            + U
        )

        return 0.5 * RHO * np.sum(
            Q * Q
        )

    # -------------------------------------
    # Temporal term
    # -------------------------------------

    temporal_value, temporal_grad, _ = (
        gated_transition_value_grad(
            X,
            graph,
            prob,
            gate_mass=GATE_MASS,
        )
    )

    analytic_temporal = float(
        np.sum(
            temporal_grad
            * direction
        )
    )

    def temporal_fun(h):
        return gated_transition_value_grad(
            X + h * direction,
            graph,
            prob,
            gate_mass=GATE_MASS,
        )[0]

    h = 1e-5

    numeric_data = (
        data_fun(h) - data_fun(-h)
    ) / (2 * h)

    numeric_admm = (
        admm_fun(h) - admm_fun(-h)
    ) / (2 * h)

    numeric_temporal = (
        temporal_fun(h)
        - temporal_fun(-h)
    ) / (2 * h)

    print()
    print("Component gradients")
    print("-------------------")
    print("term        analytic          numeric           error")

    print(
        f"data      "
        f"{analytic_data:16.9e} "
        f"{numeric_data:16.9e} "
        f"{abs(analytic_data-numeric_data):.3e}"
    )

    print(
        f"ADMM      "
        f"{analytic_admm:16.9e} "
        f"{numeric_admm:16.9e} "
        f"{abs(analytic_admm-numeric_admm):.3e}"
    )

    print(
        f"temporal  "
        f"{analytic_temporal:16.9e} "
        f"{numeric_temporal:16.9e} "
        f"{abs(analytic_temporal-numeric_temporal):.3e}"
    )

    assert np.isclose(
        analytic_data,
        numeric_data,
        rtol=1e-6,
        atol=1e-7,
    )

    assert np.isclose(
        analytic_admm,
        numeric_admm,
        rtol=1e-6,
        atol=1e-7,
    )

    assert np.isclose(
        analytic_temporal,
        numeric_temporal,
        rtol=1e-6,
        atol=1e-7,
    )

    print("component gradients: PASS")


if __name__ == "__main__":
    test_term_gradients()
    test_objective_gradient()