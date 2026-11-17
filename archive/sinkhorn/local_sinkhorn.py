import numpy as np

def scalings_to_potentials(u, v, epsilon, floor=1e-300):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    f = epsilon * np.log(np.maximum(u, floor))
    g = epsilon * np.log(np.maximum(v, floor))

    return f, g


def potentials_to_scalings(f, g, epsilon, clip=700.0):
    f = np.asarray(f, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)

    # Clipping prevents floating-point overflow.
    u = np.exp(np.clip(f / epsilon, -clip, clip))
    v = np.exp(np.clip(g / epsilon, -clip, clip))

    return u, v

def kernel_weights(graph, epsilon):
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    return np.exp(-graph.cost / epsilon)


def apply_kernel(graph, weights, v):
    """Compute K @ v using sparse graph edges."""
    return np.bincount(
        graph.src,
        weights=weights * v[graph.dst],
        minlength=graph.P,
    )


def apply_kernel_transpose(graph, weights, u):
    """Compute K.T @ u using sparse graph edges."""
    return np.bincount(
        graph.dst,
        weights=weights * u[graph.src],
        minlength=graph.P,
    )


def transport_plan_edges(graph, weights, u, v):
    """Return transport values only on graph edges."""
    return u[graph.src] * weights * v[graph.dst]


def transport_marginals(graph, plan):
    source = np.bincount(
        graph.src,
        weights=plan,
        minlength=graph.P,
    )

    target = np.bincount(
        graph.dst,
        weights=plan,
        minlength=graph.P,
    )

    return source, target


def normalize_distribution(x, floor=0.0):
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("Distribution must be 1-D")

    if np.any(x < 0):
        raise ValueError("Distribution must be nonnegative")

    if floor > 0:
        x = x + floor

    mass = np.sum(x)

    if mass <= 0:
        raise ValueError("Distribution must have positive mass")

    return x / mass

def local_sinkhorn_epsilon_scaling(
    p,
    q,
    graph,
    schedule=(1.0, 0.5, 0.2, 0.1),
    tolerances=(1e-3, 1e-4, 1e-5, 1e-6),
    maxiter=2000,
):
    if len(schedule) != len(tolerances):
        raise ValueError(
            "schedule and tolerances must have equal length"
        )

    if len(schedule) == 0:
        raise ValueError("schedule must not be empty")

    schedule = tuple(float(x) for x in schedule)
    tolerances = tuple(float(x) for x in tolerances)

    if any(e <= 0 for e in schedule):
        raise ValueError("all epsilon values must be positive")

    if any(t <= 0 for t in tolerances):
        raise ValueError("all tolerances must be positive")

    previous_potentials = None
    history = []
    result = None

    for epsilon, tol in zip(schedule, tolerances):
        if previous_potentials is None:
            warm_start = None
        else:
            f, g = previous_potentials

            warm_start = potentials_to_scalings(
                f,
                g,
                epsilon,
            )

        result = local_sinkhorn_cost(
            p,
            q,
            graph,
            epsilon=epsilon,
            tol=tol,
            maxiter=maxiter,
            warm_start=warm_start,
        )

        previous_potentials = (
            result["dual_source"].copy(),
            result["dual_target"].copy(),
        )

        history.append({
            "epsilon": epsilon,
            "tolerance": tol,
            "iterations": result["info"]["iterations"],
            "converged": result["info"]["converged"],
            "marginal_error":
                result["info"]["marginal_error"],
            "geometric_cost":
                result["geometric_cost"],
            "objective":
                result["objective"],
        })

    result["epsilon_scaling_history"] = history

    result["epsilon_scaling_info"] = {
        "stages": len(history),
        "total_iterations": int(sum(
            h["iterations"]
            for h in history
        )),
        "all_converged": bool(all(
            h["converged"]
            for h in history
        )),
        "final_epsilon": schedule[-1],
        "final_tolerance": tolerances[-1],
    }

    return result

def local_sinkhorn_cost(
    p,
    q,
    graph,
    epsilon=0.1,
    tol=1e-8,
    maxiter=500,
    floor=1e-300,
    warm_start=None,
):
    """
    Entropic balanced OT on a sparse local transport graph.

    Solves

        min_pi <C,pi>
             + epsilon * sum_e pi_e (log(pi_e) - 1)

    subject to

        pi 1 = p
        pi^T 1 = q

    over graph edges only.

    Parameters
    ----------
    p, q : ndarray, shape (P,)
        Nonnegative distributions with equal total mass.
        They are normalized internally to unit mass.

    graph : LocalTransportGraph
        Sparse local shift-scale graph.

    epsilon : float
        Entropic regularization parameter.

    tol : float
        Relative marginal-error stopping tolerance.

    maxiter : int
        Maximum Sinkhorn iterations.

    floor : float
        Numerical floor used in divisions.

    warm_start : tuple(u, v), optional
        Previous Sinkhorn scaling vectors.

    Returns
    -------
    result : dict
        Contains cost, objective, plan, marginals,
        scaling vectors, and convergence diagnostics.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    if tol <= 0:
        raise ValueError("tol must be positive")

    if maxiter <= 0:
        raise ValueError("maxiter must be positive")

    p = normalize_distribution(p)
    q = normalize_distribution(q)

    if p.size != graph.P or q.size != graph.P:
        raise ValueError(
            f"Expected distributions of length {graph.P}"
        )

    weights = kernel_weights(graph, epsilon)

    if warm_start is None:
        u = np.ones(graph.P, dtype=np.float64)
        v = np.ones(graph.P, dtype=np.float64)
    else:
        u = np.asarray(warm_start[0], dtype=np.float64).copy()
        v = np.asarray(warm_start[1], dtype=np.float64).copy()

        if u.shape != p.shape or v.shape != q.shape:
            raise ValueError("Invalid warm-start shapes")

    converged = False
    marginal_error = np.inf

    for iteration in range(1, maxiter + 1):
        Kv = apply_kernel(graph, weights, v)
        u = p / np.maximum(Kv, floor)

        KTu = apply_kernel_transpose(graph, weights, u)
        v = q / np.maximum(KTu, floor)

        # Checking every iteration is fine for validation.
        plan = transport_plan_edges(
            graph, weights, u, v
        )

        p_hat, q_hat = transport_marginals(
            graph, plan
        )

        p_error = np.linalg.norm(
            p_hat - p, ord=1
        )

        q_error = np.linalg.norm(
            q_hat - q, ord=1
        )

        marginal_error = max(
            p_error,
            q_error,
        )

        if marginal_error <= tol:
            converged = True
            break

    plan = transport_plan_edges(
        graph, weights, u, v
    )

    p_hat, q_hat = transport_marginals(
        graph, plan
    )

    geometric_cost = float(
        np.dot(
            graph.cost,
            plan,
        )
    )

    positive = plan > 0

    entropy_term = float(
        np.sum(
            plan[positive]
            * (
                np.log(plan[positive])
                - 1.0
            )
        )
    )

    objective = (
        geometric_cost
        + epsilon * entropy_term
    )

    

    info = {
        "converged": converged,
        "iterations": iteration,
        "marginal_error": float(marginal_error),
        "source_error_l1": float(
            np.linalg.norm(
                p_hat - p,
                ord=1,
            )
        ),
        "target_error_l1": float(
            np.linalg.norm(
                q_hat - q,
                ord=1,
            )
        ),
        "transport_mass": float(
            np.sum(plan)
        ),
        "epsilon": float(epsilon),

    }

    f, g = scalings_to_potentials(u,v,epsilon,)

    return {
        "geometric_cost": geometric_cost,
        "objective": float(objective),
        "entropy_term": entropy_term,
        "plan": plan,
        "source_marginal": p_hat,
        "target_marginal": q_hat,
        "u": u,
        "v": v,
        "weights": weights,
        "info": info,
        "dual_source": f,
        "dual_target": g,
    }

