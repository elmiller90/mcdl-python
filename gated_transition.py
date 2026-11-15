import numpy as np

from local_transition import normalized_transition_value_grad


def gate(m, mu):
    if mu <= 0:
        raise ValueError("gate_mass must be positive")
    return m / (m + mu)


def gate_derivative(m, mu):
    return mu / (m + mu) ** 2


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
    delta=1e-12,
    inactive_mass=1e-14,
):
    """
    Gated bidirectional local-transition regularizer.

    X shape: (L,K,J,T), nonnegative.

    R(X) = sum_{k,t}
        g(m_kt) g(m_k,t+1)
        R_trans(x_kt, x_k,t+1)

    Pairwise R_trans acts on normalized coefficient maps.

    Returns
    -------
    value : float
    grad : ndarray, same shape as X
    diagnostics : dict
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 4:
        raise ValueError("X must have shape (L,K,J,T)")
    if np.any(X < 0):
        raise ValueError("X must be nonnegative")
    if gate_mass <= 0:
        raise ValueError("gate_mass must be positive")

    L, K, J, T = X.shape

    if L != graph.L or J != graph.J:
        raise ValueError(
            "X shift/scale dimensions do not match graph"
        )

    grad = np.zeros_like(X)
    value = 0.0

    masses = np.zeros((K, T))
    gates = np.zeros((K, T))

    pair_value = np.zeros((K, max(T - 1, 0)))
    pair_weight = np.zeros_like(pair_value)
    weighted_value = np.zeros_like(pair_value)
    skipped = np.zeros_like(pair_value, dtype=bool)

    for k in range(K):
        for t in range(T):
            x = frame_to_vector(X, k, t)
            m = float(x.sum())

            masses[k, t] = m
            gates[k, t] = gate(m, gate_mass)

    for k in range(K):
        for t in range(T - 1):
            x = frame_to_vector(X, k, t)
            y = frame_to_vector(X, k, t + 1)

            mx = masses[k, t]
            my = masses[k, t + 1]

            gx_gate = gates[k, t]
            gy_gate = gates[k, t + 1]

            w = gx_gate * gy_gate
            pair_weight[k, t] = w

            # Pairwise normalized penalty is undefined for zero mass.
            # Its gated limit is zero when either frame is inactive.
            if mx <= inactive_mass or my <= inactive_mass:
                skipped[k, t] = True
                continue

            r, grad_x, grad_y, _ = (
                normalized_transition_value_grad(
                    x,
                    y,
                    graph,
                    prob,
                    delta=delta,
                )
            )

            pair_value[k, t] = r
            weighted_value[k, t] = w * r
            value += w * r

            # Shape derivative.
            grad_x = w * grad_x
            grad_y = w * grad_y

            # Gate derivative:
            # d[w*r]/dx =
            #   w * dr/dx
            #   + r * g'(mx)*g(my) * 1.
            dw_dmx = (
                gate_derivative(mx, gate_mass)
                * gy_gate
            )

            dw_dmy = (
                gx_gate
                * gate_derivative(my, gate_mass)
            )

            grad_x += r * dw_dmx
            grad_y += r * dw_dmy

            grad[:, k, :, t] += vector_to_frame(
                grad_x, L, J
            )

            grad[:, k, :, t + 1] += vector_to_frame(
                grad_y, L, J
            )

    diagnostics = {
        "masses": masses,
        "frame_gates": gates,
        "pair_weights": pair_weight,
        "pair_values": pair_value,
        "weighted_pair_values": weighted_value,
        "skipped_pairs": skipped,
        "number_skipped": int(np.count_nonzero(skipped)),
        "number_pairs": int(K * max(T - 1, 0)),
        "gate_mass": float(gate_mass),
    }

    return float(value), grad, diagnostics
