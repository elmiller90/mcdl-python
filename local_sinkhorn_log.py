import numpy as np


LOG_ZERO = -np.inf


def normalize_distribution(x):
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("Distribution must be 1-D")
    if np.any(x < 0):
        raise ValueError("Distribution must be nonnegative")

    mass = x.sum()

    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Distribution must have finite positive mass")

    return x / mass


def safe_log(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, LOG_ZERO)
    mask = x > 0
    out[mask] = np.log(x[mask])
    return out


def segmented_logsumexp(values, groups, size):
    """
    Compute y[i] = log(sum(exp(values[e]))) for edges
    satisfying groups[e] == i.
    """
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)

    maxima = np.full(size, -np.inf)
    np.maximum.at(maxima, groups, values)

    finite_edge = np.isfinite(maxima[groups])
    shifted = np.zeros_like(values)

    shifted[finite_edge] = np.exp(
        values[finite_edge] - maxima[groups[finite_edge]]
    )

    sums = np.bincount(
        groups,
        weights=shifted,
        minlength=size,
    )

    result = np.full(size, -np.inf)
    valid = np.isfinite(maxima) & (sums > 0)
    result[valid] = maxima[valid] + np.log(sums[valid])

    return result


def source_log_kernel_sum(graph, g, epsilon):
    """log sum_j exp((g_j-C_ij)/epsilon)."""
    values = (
        g[graph.dst] - graph.cost
    ) / epsilon

    return segmented_logsumexp(
        values,
        graph.src,
        graph.P,
    )


def target_log_kernel_sum(graph, f, epsilon):
    """log sum_i exp((f_i-C_ij)/epsilon)."""
    values = (
        f[graph.src] - graph.cost
    ) / epsilon

    return segmented_logsumexp(
        values,
        graph.dst,
        graph.P,
    )


def log_transport_plan(graph, f, g, epsilon):
    return (
        f[graph.src]
        + g[graph.dst]
        - graph.cost
    ) / epsilon


def transport_plan_edges(graph, f, g, epsilon):
    log_plan = log_transport_plan(
        graph, f, g, epsilon
    )

    plan = np.zeros_like(log_plan)
    finite = np.isfinite(log_plan)
    plan[finite] = np.exp(
        np.clip(log_plan[finite], -745.0, 700.0)
    )

    return plan


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


def marginal_error(p, q, p_hat, q_hat):
    return max(
        np.linalg.norm(p_hat-p, ord=1),
        np.linalg.norm(q_hat-q, ord=1),
    )


def validate_local_feasibility(p, q, graph):
    """
    Basic support-level feasibility checks.

    These checks are necessary but not sufficient for arbitrary
    sparse bipartite transport problems.
    """
    source_has_edge = np.bincount(
        graph.src,
        minlength=graph.P,
    ) > 0

    target_has_edge = np.bincount(
        graph.dst,
        minlength=graph.P,
    ) > 0

    if np.any((p > 0) & ~source_has_edge):
        raise ValueError("Positive source mass has no outgoing edge")

    if np.any((q > 0) & ~target_has_edge):
        raise ValueError("Positive target mass has no incoming edge")

    reachable_target = np.zeros(graph.P, dtype=bool)
    reachable_target[graph.dst[p[graph.src] > 0]] = True

    reachable_source = np.zeros(graph.P, dtype=bool)
    reachable_source[graph.src[q[graph.dst] > 0]] = True

    if np.any((q > 0) & ~reachable_target):
        raise ValueError(
            "Positive target mass is unreachable from source support"
        )

    if np.any((p > 0) & ~reachable_source):
        raise ValueError(
            "Positive source mass cannot reach target support"
        )


def gauge_center(f, g):
    finite_f = np.isfinite(f)
    finite_g = np.isfinite(g)

    if not np.any(finite_f) or not np.any(finite_g):
        return f, g

    shift = 0.5 * (
        np.mean(f[finite_f])
        - np.mean(g[finite_g])
    )

    f = f.copy()
    g = g.copy()

    f[finite_f] -= shift
    g[finite_g] += shift

    return f, g

def local_sinkhorn_log(
    p,
    q,
    graph,
    epsilon=0.1,
    tol=1e-8,
    maxiter=5000,
    warm_start=None,
    check_every=10,
    center_every=50,
):
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if maxiter <= 0:
        raise ValueError("maxiter must be positive")
    if check_every <= 0:
        raise ValueError("check_every must be positive")

    p = normalize_distribution(p)
    q = normalize_distribution(q)

    if p.size != graph.P or q.size != graph.P:
        raise ValueError(
            f"Expected distributions of length {graph.P}"
        )

    validate_local_feasibility(p, q, graph)

    log_p = safe_log(p)
    log_q = safe_log(q)

    active_p = p > 0
    active_q = q > 0

    # Zero-mass states must have -inf potential so they
    # cannot carry transport mass.
    if warm_start is None:
        f = np.full(graph.P, -np.inf, dtype=np.float64)
        g = np.full(graph.P, -np.inf, dtype=np.float64)

        f[active_p] = 0.0
        g[active_q] = 0.0
    else:
        f = np.asarray(
            warm_start[0], dtype=np.float64
        ).copy()

        g = np.asarray(
            warm_start[1], dtype=np.float64
        ).copy()

        if f.shape != p.shape or g.shape != q.shape:
            raise ValueError("Invalid warm-start shapes")

        f[~active_p] = -np.inf
        g[~active_q] = -np.inf

        # A newly active state may have inherited -inf
        # from the previous solution.
        bad_f = active_p & ~np.isfinite(f)
        bad_g = active_q & ~np.isfinite(g)

        f[bad_f] = 0.0
        g[bad_g] = 0.0

    converged = False
    error = np.inf

    p_hat = np.zeros_like(p)
    q_hat = np.zeros_like(q)

    for iteration in range(1, maxiter + 1):

        # ---------------------------------------------
        # Source-potential update
        # ---------------------------------------------

        source_lse = source_log_kernel_sum(
            graph,
            g,
            epsilon,
        )

        if np.any(
            active_p & ~np.isfinite(source_lse)
        ):
            raise FloatingPointError(
                "Source update encountered unreachable mass"
            )

        f[active_p] = epsilon * (
            log_p[active_p]
            - source_lse[active_p]
        )

        f[~active_p] = -np.inf

        # ---------------------------------------------
        # Target-potential update
        # ---------------------------------------------

        target_lse = target_log_kernel_sum(
            graph,
            f,
            epsilon,
        )

        if np.any(
            active_q & ~np.isfinite(target_lse)
        ):
            raise FloatingPointError(
                "Target update encountered unreachable mass"
            )

        g[active_q] = epsilon * (
            log_q[active_q]
            - target_lse[active_q]
        )

        g[~active_q] = -np.inf

        # Remove dual gauge drift without changing pi.
        if (
            center_every
            and iteration % center_every == 0
        ):
            f, g = gauge_center(f, g)

        # ---------------------------------------------
        # Convergence check
        # ---------------------------------------------

        if (
            iteration % check_every == 0
            or iteration == 1
            or iteration == maxiter
        ):
            plan = transport_plan_edges(
                graph,
                f,
                g,
                epsilon,
            )

            p_hat, q_hat = transport_marginals(
                graph,
                plan,
            )

            error = marginal_error(
                p,
                q,
                p_hat,
                q_hat,
            )

            if error <= tol:
                converged = True
                break

    # ---------------------------------------------
    # Final transport plan
    # ---------------------------------------------

    plan = transport_plan_edges(
        graph,
        f,
        g,
        epsilon,
    )

    p_hat, q_hat = transport_marginals(
        graph,
        plan,
    )

    error = marginal_error(
        p,
        q,
        p_hat,
        q_hat,
    )

    # ---------------------------------------------
    # Costs
    # ---------------------------------------------

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

    objective = float(
        geometric_cost
        + epsilon * entropy_term
    )

    # ---------------------------------------------
    # Diagnostics
    # ---------------------------------------------

    info = {
        "converged": bool(converged),
        "iterations": int(iteration),
        "marginal_error": float(error),
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
        "active_source_states": int(
            np.count_nonzero(active_p)
        ),
        "active_target_states": int(
            np.count_nonzero(active_q)
        ),
    }

    return {
        "geometric_cost": geometric_cost,
        "objective": objective,
        "entropy_term": entropy_term,
        "plan": plan,
        "source_marginal": p_hat,
        "target_marginal": q_hat,
        "dual_source": f,
        "dual_target": g,
        "info": info,
    }


def local_sinkhorn_log_epsilon_scaling(
    p,
    q,
    graph,
    schedule=(1.0, 0.5, 0.2, 0.1),
    tolerances=(1e-3, 1e-4, 1e-5, 1e-6),
    maxiter=2000,
    check_every=10,
):
    if len(schedule) != len(tolerances):
        raise ValueError(
            "schedule and tolerances must have equal length"
        )

    if not schedule:
        raise ValueError("schedule must not be empty")

    warm = None
    history = []
    result = None

    for epsilon, tol in zip(schedule, tolerances):
        result = local_sinkhorn_log(
            p,
            q,
            graph,
            epsilon=float(epsilon),
            tol=float(tol),
            maxiter=maxiter,
            warm_start=warm,
            check_every=check_every,
        )

        warm = (
            result["dual_source"].copy(),
            result["dual_target"].copy(),
        )

        history.append({
            "epsilon": float(epsilon),
            "tolerance": float(tol),
            "iterations":
                result["info"]["iterations"],
            "converged":
                result["info"]["converged"],
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
            h["iterations"] for h in history
        )),
        "all_converged": bool(all(
            h["converged"] for h in history
        )),
        "final_epsilon": float(schedule[-1]),
        "final_tolerance": float(tolerances[-1]),
    }

    return result