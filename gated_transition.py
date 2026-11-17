import numpy as np

from local_transition import (
    regularized_transition_value_grad,
)


def gate(m, mu):
    if mu <= 0:
        raise ValueError("gate_mass must be positive")

    m2 = m * m
    mu2 = mu * mu

    return m2 / (m2 + mu2)


def gate_derivative(m, mu):
    m2 = m * m
    mu2 = mu * mu

    return (
        2.0 * m * mu2
        / (m2 + mu2) ** 2
    )


def frame_to_vector(X, k, t):
    """
    X[:, k, :, t] with shape (L,J) -> vector of length L*J.

    Ordering matches state_index(n,j,J) = n*J+j.
    """
    return np.asarray(
        X[:, k, :, t],
        dtype=np.float64,
    ).reshape(-1)


def vector_to_frame(x, L, J):
    return np.asarray(
        x,
        dtype=np.float64,
    ).reshape(L, J)


def gated_transition_value_grad(
    X,
    graph,
    prob,
    gate_mass=0.05,
    mass_floor=1e-10,
    hellinger_delta=1e-12,
    background=None,
):
    """
    Smooth gated bidirectional local-transition regularizer.

    X has shape (L,K,J,T).

    No hard inactive-mass branch is used. Low-mass frames
    are regularized using a tiny background distribution,
    while the smooth gate drives their contribution to zero.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 4:
        raise ValueError(
            "X must have shape (L,K,J,T)"
        )
    if np.any(X < 0):
        raise ValueError("X must be nonnegative")
    if gate_mass <= 0:
        raise ValueError(
            "gate_mass must be positive"
        )
    if mass_floor <= 0:
        raise ValueError(
            "mass_floor must be positive"
        )

    L, K, J, T = X.shape

    if L != graph.L or J != graph.J:
        raise ValueError(
            "X shift/scale dimensions do not match graph"
        )

    P = graph.P

    if background is None:
        background_vector = np.full(
            P,
            1.0 / P,
            dtype=np.float64,
        )
    else:
        background_vector = np.asarray(
            background,
            dtype=np.float64,
        )

        if background_vector.shape != (P,):
            raise ValueError(
                f"background must have shape ({P},)"
            )
        if np.any(background_vector < 0):
            raise ValueError(
                "background must be nonnegative"
            )

        total = float(background_vector.sum())

        if total <= 0:
            raise ValueError(
                "background must have positive mass"
            )

        background_vector = (
            background_vector / total
        )

    grad = np.zeros_like(X)
    value = 0.0

    masses = np.zeros((K, T))
    frame_gates = np.zeros((K, T))

    pair_shape = (K, max(T - 1, 0))

    pair_values = np.zeros(pair_shape)
    pair_weights = np.zeros(pair_shape)
    weighted_pair_values = np.zeros(pair_shape)

    for k in range(K):
        for t in range(T):
            x = frame_to_vector(X, k, t)
            m = float(x.sum())

            masses[k, t] = m
            frame_gates[k, t] = gate(
                m,
                gate_mass,
            )

    for k in range(K):
        for t in range(T - 1):
            x = frame_to_vector(X, k, t)
            y = frame_to_vector(X, k, t + 1)

            mx = masses[k, t]
            my = masses[k, t + 1]

            gx_gate = frame_gates[k, t]
            gy_gate = frame_gates[k, t + 1]

            weight = gx_gate * gy_gate

            pair_value, grad_x, grad_y, _ = (
                regularized_transition_value_grad(
                    x,
                    y,
                    graph,
                    prob,
                    mass_floor=mass_floor,
                    hellinger_delta=hellinger_delta,
                    background=background_vector,
                )
            )

            weighted = weight * pair_value

            pair_values[k, t] = pair_value
            pair_weights[k, t] = weight
            weighted_pair_values[k, t] = weighted

            value += weighted

            # Shape contribution.
            grad_x = weight * grad_x
            grad_y = weight * grad_y

            # Smooth gate contribution.
            dw_dmx = (
                gate_derivative(mx, gate_mass)
                * gy_gate
            )

            dw_dmy = (
                gx_gate
                * gate_derivative(my, gate_mass)
            )

            grad_x += pair_value * dw_dmx
            grad_y += pair_value * dw_dmy

            grad[:, k, :, t] += vector_to_frame(
                grad_x,
                L,
                J,
            )

            grad[:, k, :, t + 1] += vector_to_frame(
                grad_y,
                L,
                J,
            )

    diagnostics = {
        "masses": masses,
        "frame_gates": frame_gates,
        "pair_weights": pair_weights,
        "pair_values": pair_values,
        "weighted_pair_values": weighted_pair_values,
        "number_pairs": int(
            K * max(T - 1, 0)
        ),
        "mass_floor": float(mass_floor),
        "gate_mass": float(gate_mass),
        "minimum_pair_weight": (
            float(pair_weights.min())
            if pair_weights.size
            else 0.0
        ),
        "maximum_pair_weight": (
            float(pair_weights.max())
            if pair_weights.size
            else 0.0
        ),
    }

    return float(value), grad, diagnostics

def make_gated_transition_regularizer(
    graph,
    prob,
    gate_mass=0.05,
    mass_floor=1e-10,
    hellinger_delta=1e-12,
    background=None,
):
    def regularizer(X):
        value, grad, _ = gated_transition_value_grad(
            X,
            graph,
            prob,
            gate_mass=gate_mass,
            mass_floor=mass_floor,
            hellinger_delta=hellinger_delta,
            background=background,
        )

        return value, grad

    return regularizer