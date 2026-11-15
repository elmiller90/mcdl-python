import numpy as np

def regularized_normalization_value_grad(
    x,
    grad_p,
    mass_floor=1e-10,
    background=None,
):
    """
    Map a gradient with respect to

        p = (x + mass_floor*u) / (sum(x) + mass_floor)

    back to x.

    The background u must be nonnegative and sum to one.
    """
    x = np.asarray(x, dtype=np.float64)
    grad_p = np.asarray(grad_p, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    if grad_p.shape != x.shape:
        raise ValueError("grad_p shape must match x")
    if np.any(x < 0):
        raise ValueError("x must be nonnegative")
    if mass_floor <= 0:
        raise ValueError("mass_floor must be positive")

    if background is None:
        background = np.full(
            x.size,
            1.0 / x.size,
            dtype=np.float64,
        )
    else:
        background = np.asarray(
            background,
            dtype=np.float64,
        )

        if background.shape != x.shape:
            raise ValueError(
                "background shape must match x"
            )
        if np.any(background < 0):
            raise ValueError(
                "background must be nonnegative"
            )

        total = float(background.sum())

        if total <= 0:
            raise ValueError(
                "background must have positive mass"
            )

        background = background / total

    mass = float(x.sum())
    denominator = mass + mass_floor

    p = (
        x + mass_floor * background
    ) / denominator

    # J^T grad_p, where
    #
    # J = (I - p 1^T) / denominator.
    grad_x = (
        grad_p - np.dot(grad_p, p)
    ) / denominator

    return p, grad_x, mass

def build_transition_kernel(graph, epsilon=0.5):
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    src = np.asarray(graph.src)
    dst = np.asarray(graph.dst)
    cost = np.asarray(graph.cost)

    if src.ndim != 1:
        raise ValueError(
            f"graph.src must be 1-D, got shape {src.shape}"
        )

    if dst.ndim != 1:
        raise ValueError(
            f"graph.dst must be 1-D, got shape {dst.shape}"
        )

    if cost.ndim != 1:
        raise ValueError(
            f"graph.cost must be 1-D, got shape {cost.shape}"
        )

    if not (
        len(src) == len(dst) == len(cost)
    ):
        raise ValueError(
            "graph src/dst/cost lengths do not match"
        )

    if not np.issubdtype(src.dtype, np.integer):
        raise ValueError(
            f"graph.src must be integer, got {src.dtype}"
        )

    if not np.issubdtype(dst.dtype, np.integer):
        raise ValueError(
            f"graph.dst must be integer, got {dst.dtype}"
        )

    w = np.exp(
        -cost.astype(np.float64) / epsilon
    )

    row_sum = np.bincount(
        src,
        weights=w,
        minlength=graph.P,
    )

    if np.any(row_sum <= 0):
        raise ValueError(
            "Graph contains empty transition rows"
        )

    return w / row_sum[src]


def apply_transition(graph, prob, p):
    """
    q = K.T @ p

    K is row-stochastic, so normalized nonnegative
    p produces normalized nonnegative q.
    """
    p = np.asarray(p, dtype=np.float64)

    return np.bincount(
        graph.dst,
        weights=prob * p[graph.src],
        minlength=graph.P,
    )


def apply_transition_transpose(graph, prob, q):
    """
    p = K @ q
    """
    q = np.asarray(q, dtype=np.float64)

    return np.bincount(
        graph.src,
        weights=prob * q[graph.dst],
        minlength=graph.P,
    )


def normalize_distribution(x, floor=0.0):
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    if np.any(x < 0):
        raise ValueError("x must be nonnegative")

    if floor > 0:
        x = x + floor

    mass = x.sum()

    if mass <= 0:
        raise ValueError("x must have positive mass")

    return x / mass


def hellinger_value_grad(p, q, delta=1e-12):
    """
    Smoothed squared Hellinger-like distance

        H(p,q) = sum_i (sqrt(p_i+d)-sqrt(q_i+d))^2

    Returns H, dH/dp, dH/dq.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    sp = np.sqrt(p + delta)
    sq = np.sqrt(q + delta)

    d = sp - sq
    value = float(np.dot(d, d))

    grad_p = d / sp
    grad_q = -d / sq

    return value, grad_p, grad_q


def transition_value_grad(
    p,
    q,
    graph,
    prob,
    delta=1e-12,
):
    """
    Bidirectional local-transition penalty:

      R(p,q) =
        1/2 H(q, K.T p)
        + 1/2 H(p, K.T q)

    Note that the same physical transition kernel is used
    in both temporal directions by swapping p and q.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    forward = apply_transition(
        graph, prob, p
    )

    backward = apply_transition(
        graph, prob, q
    )

    # H(q, K.T p)
    h1, dh1_q, dh1_pred = hellinger_value_grad(
        q,
        forward,
        delta=delta,
    )

    # H(p, K.T q)
    h2, dh2_p, dh2_pred = hellinger_value_grad(
        p,
        backward,
        delta=delta,
    )

    value = .5 * (h1 + h2)

    # From first term:
    # d/dp H(q,K.T p) = K dh/dpred
    grad_p = .5 * apply_transition_transpose(
        graph,
        prob,
        dh1_pred,
    )

    grad_q = .5 * dh1_q

    # From second term.
    grad_p += .5 * dh2_p

    grad_q += .5 * apply_transition_transpose(
        graph,
        prob,
        dh2_pred,
    )

    diagnostics = {
        "forward_mass": float(forward.sum()),
        "backward_mass": float(backward.sum()),
        "forward_min": float(forward.min()),
        "backward_min": float(backward.min()),
        "forward_hellinger": float(h1),
        "backward_hellinger": float(h2),
    }

    return value, grad_p, grad_q, diagnostics


def normalized_transition_value_grad(
    x,
    y,
    graph,
    prob,
    delta=1e-12,
):
    """
    Same penalty for unnormalized coefficient masses.

      p = x / sum(x)
      q = y / sum(y)

    Includes the exact normalization Jacobian.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mx = float(x.sum())
    my = float(y.sum())

    if mx <= 0 or my <= 0:
        raise ValueError("Coefficient masses must be positive")

    p = x / mx
    q = y / my

    value, gp, gq, diagnostics = transition_value_grad(
        p,
        q,
        graph,
        prob,
        delta=delta,
    )

    # J^T g for p=x/sum(x):
    # (g - <g,p>1) / sum(x)
    grad_x = (
        gp - np.dot(gp, p)
    ) / mx

    grad_y = (
        gq - np.dot(gq, q)
    ) / my

    diagnostics.update({
        "source_mass": mx,
        "target_mass": my,
    })

    return value, grad_x, grad_y, diagnostics

def regularized_transition_value_grad(
    x,
    y,
    graph,
    prob,
    mass_floor=1e-10,
    hellinger_delta=1e-12,
    background=None,
):
    """
    Bidirectional transition penalty with smooth low-mass
    normalization.

        p = (x + d*u) / (sum(x) + d)
        q = (y + d*u) / (sum(y) + d)

    This is defined even when x or y has zero total mass.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D")
    if x.shape != y.shape:
        raise ValueError("x and y must have equal shape")
    if np.any(x < 0) or np.any(y < 0):
        raise ValueError("x and y must be nonnegative")
    if mass_floor <= 0:
        raise ValueError("mass_floor must be positive")

    P = x.size

    if background is None:
        u = np.full(
            P,
            1.0 / P,
            dtype=np.float64,
        )
    else:
        u = np.asarray(
            background,
            dtype=np.float64,
        )

        if u.shape != x.shape:
            raise ValueError(
                "background shape must match x"
            )
        if np.any(u < 0):
            raise ValueError(
                "background must be nonnegative"
            )

        total = float(u.sum())

        if total <= 0:
            raise ValueError(
                "background must have positive mass"
            )

        u = u / total

    mx = float(x.sum())
    my = float(y.sum())

    sx = mx + mass_floor
    sy = my + mass_floor

    p = (
        x + mass_floor * u
    ) / sx

    q = (
        y + mass_floor * u
    ) / sy

    value, gp, gq, diagnostics = (
        transition_value_grad(
            p,
            q,
            graph,
            prob,
            delta=hellinger_delta,
        )
    )

    grad_x = (
        gp - np.dot(gp, p)
    ) / sx

    grad_y = (
        gq - np.dot(gq, q)
    ) / sy

    diagnostics.update({
        "source_mass": mx,
        "target_mass": my,
        "source_regularized_mass": sx,
        "target_regularized_mass": sy,
        "mass_floor": float(mass_floor),
    })

    return value, grad_x, grad_y, diagnostics
