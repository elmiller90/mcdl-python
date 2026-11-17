import numpy as np

from softmin import softmin
from x_update import solve_x, structured_to_softmin
from sparse_admm import validate_admm_problem


def log_penalty(X, lambda_s, a):
    X = np.asarray(X, dtype=float)
    if lambda_s < 0:
        raise ValueError("lambda_s must be nonnegative")
    if a <= 0:
        raise ValueError("a must be positive")
    return float((lambda_s / a) * np.sum(np.log1p(a * np.abs(X))))


def continuation_a(iteration, lambda_s, a_min=1e-4,
                   continuation_iters=100, fraction=0.95):
    if lambda_s <= 0:
        return float(a_min)
    if a_min <= 0 or continuation_iters <= 0:
        raise ValueError("a_min and continuation_iters must be positive")
    a_max = fraction / lambda_s
    exponent = min(iteration, continuation_iters) / continuation_iters
    return float(a_min * (a_max / a_min) ** exponent)


def log_threshold_nonnegative(v, threshold, a):
    v = np.asarray(v, dtype=float)
    if threshold < 0 or a <= 0:
        raise ValueError("threshold must be nonnegative and a positive")
    y = np.zeros_like(v)
    mask = v >= threshold
    vm = v[mask]
    rad = (0.5 * vm + 0.5 / a) ** 2 - threshold / a
    y[mask] = 0.5 * vm - 0.5 / a + np.sqrt(np.maximum(rad, 0.0))
    return y


def log_objective(X, S, operator, lambda_s, lambda2, tau, a,
                  temporal_regularizer=None):
    R = operator.forward(X) - S
    data = 0.5 * np.sum(R * R)
    sparsity = log_penalty(X, lambda_s, a)

    if temporal_regularizer is None:
        raw_temporal = softmin(
            structured_to_softmin(X), K=operator.K, J=operator.J,
            tau=tau, pad_mode="zero", normalize=False,
            return_gradient=False,
        )
    elif lambda2 == 0:
        raw_temporal = 0.0
    else:
        raw_temporal = float(temporal_regularizer(X)[0])

    temporal = lambda2 * raw_temporal
    return {
        "data": float(data),
        "sparsity": float(sparsity),
        "temporal": float(raw_temporal),
        "weighted_temporal": float(temporal),
        "softmin": float(raw_temporal),
        "weighted_softmin": float(temporal),
        "total": float(data + sparsity + temporal),
    }


def _copy_state_array(value, shape, name, default):
    if value is None:
        return default.copy()
    out = np.asarray(value, dtype=float).copy()
    if out.shape != shape:
        raise ValueError(f"state0['{name}'] has shape {out.shape}; expected {shape}")
    return out


def solve_log_admm_stateful(
    X0, S, operator, lambda_s, lambda2=0.0, tau=1e-2,
    rho=1.0, max_admm_iter=200, x_maxiter=50,
    abs_tol=1e-4, rel_tol=1e-3, adaptive_rho=True,
    rho_mu=10.0, rho_tau=2.0, a_min=1e-4,
    continuation_iters=100, a_fraction=0.95, verbose=True,
    temporal_regularizer=None, state0=None,
    continue_iteration=False,
):
    """Compatibility ADMM solver with optional full-state warm start.

    With state0=None this follows the original solver initialization exactly.
    With state0 supplied, X, Y, U, rho, a, and optionally the global
    continuation iteration are restored. Return order remains X, Y, U, info.
    """
    Xbase = np.asarray(X0, dtype=float).copy()
    S = np.asarray(S, dtype=float)
    validate_admm_problem(Xbase, S, operator)

    if lambda_s <= 0 or rho <= 0:
        raise ValueError("lambda_s and rho must be positive")

    state0 = {} if state0 is None else dict(state0)
    X = _copy_state_array(state0.get("X"), Xbase.shape, "X", Xbase)
    Y = _copy_state_array(state0.get("Y"), Xbase.shape, "Y", np.zeros_like(Xbase))
    U = _copy_state_array(state0.get("U"), Xbase.shape, "U", np.zeros_like(Xbase))

    rho = float(state0.get("rho", rho))
    if rho <= 0:
        raise ValueError("warm-start rho must be positive")

    iteration_offset = int(state0.get("iteration", 0)) if continue_iteration else 0
    if iteration_offset < 0:
        raise ValueError("state iteration must be nonnegative")

    # If global continuation is not requested, each parameter solve uses the
    # original continuation schedule. The prior a remains available in state.
    prior_a = float(state0.get("a", a_min))
    if prior_a <= 0:
        raise ValueError("warm-start a must be positive")

    history = []
    nvar = X.size
    converged = False

    if verbose:
        print("\nStateful log-penalty ADMM")
        print("--------------------------")
        print("iter objective data log-penalty primal dual rho a")

    for local_iteration in range(1, max_admm_iter + 1):
        global_iteration = iteration_offset + local_iteration

        X, x_result = solve_x(
            X0=X, S=S, Y=Y, U=U, operator=operator,
            rho=rho, lambda2=lambda2, tau=tau,
            maxiter=x_maxiter, ftol=1e-10, gtol=1e-7, maxls=50,
            temporal_regularizer=temporal_regularizer,
        )

        Y_old = Y.copy()
        continuation_iteration = global_iteration if continue_iteration else local_iteration
        a = continuation_a(
            continuation_iteration, lambda_s, a_min,
            continuation_iters, a_fraction,
        )
        Y = log_threshold_nonnegative(X + U, lambda_s / rho, a)

        primal_array = X - Y
        U += primal_array
        primal = np.linalg.norm(primal_array)
        dual = rho * np.linalg.norm(Y - Y_old)

        eps_primal = (
            np.sqrt(nvar) * abs_tol
            + rel_tol * max(np.linalg.norm(X), np.linalg.norm(Y))
        )
        eps_dual = (
            np.sqrt(nvar) * abs_tol
            + rel_tol * rho * np.linalg.norm(U)
        )

        terms = log_objective(
            Y, S, operator, lambda_s, lambda2, tau, a,
            temporal_regularizer=temporal_regularizer,
        )

        record = {
            "iteration": local_iteration,
            "global_iteration": global_iteration,
            "objective": terms["total"],
            "data": terms["data"],
            "sparsity": terms["sparsity"],
            "temporal": terms["weighted_temporal"],
            "softmin": terms["weighted_softmin"],
            "primal_residual": float(primal),
            "dual_residual": float(dual),
            "primal_tolerance": float(eps_primal),
            "dual_tolerance": float(eps_dual),
            "rho": float(rho),
            "a": float(a),
            "nonzero_count": int(np.count_nonzero(Y)),
            "x_success": bool(x_result.success),
            "x_iterations": int(x_result.nit),
        }
        history.append(record)

        if verbose and (local_iteration == 1 or local_iteration % 5 == 0):
            print(
                f"{local_iteration:4d} {terms['total']:11.4e} "
                f"{terms['data']:9.3e} {terms['sparsity']:11.3e} "
                f"{primal:9.3e} {dual:9.3e} {rho:8.2e} {a:8.2e}"
            )

        continuation_complete = continuation_iteration >= continuation_iters
        if continuation_complete and primal <= eps_primal and dual <= eps_dual:
            converged = True
            break

        if adaptive_rho:
            old_rho = rho
            if primal > rho_mu * dual:
                rho *= rho_tau
            elif dual > rho_mu * primal:
                rho /= rho_tau
            if rho != old_rho:
                U *= old_rho / rho

    last = history[-1]
    final_iteration = iteration_offset + local_iteration
    state = {
        "X": X.copy(),
        "Y": Y.copy(),
        "U": U.copy(),
        "rho": float(rho),
        "a": float(last["a"]),
        "iteration": int(final_iteration),
    }
    info = {
        "converged": converged,
        "iterations": local_iteration,
        "rho": float(rho),
        "a": float(last["a"]),
        "history": history,
        "continuation_complete": bool(last["global_iteration"] >= continuation_iters),
        "final_objective": last["objective"],
        "final_primal_residual": last["primal_residual"],
        "final_dual_residual": last["dual_residual"],
        "nonzero_count": last["nonzero_count"],
        "state": state,
        "prior_a": prior_a,
    }
    return X, Y, U, info


# Drop-in alias for scripts that prefer the original function name.
solve_log_admm = solve_log_admm_stateful
