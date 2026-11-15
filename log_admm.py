import numpy as np

from softmin import softmin
from x_update import solve_x, structured_to_softmin
from sparse_admm import validate_admm_problem


def log_penalty(X, lambda_s, a):
    """Equation (14): (lambda_s/a) sum log(1 + a|x|)."""
    X = np.asarray(X, dtype=np.float64)

    if lambda_s < 0:
        raise ValueError("lambda_s must be nonnegative")
    if a <= 0:
        raise ValueError("a must be positive")

    return float((lambda_s / a) * np.sum(np.log1p(a * np.abs(X))))


def continuation_a(iteration, lambda_s, a_min=1e-4,
                   continuation_iters=100, fraction=0.95):
    """Equation (25), using a_max = 0.95/lambda_s."""
    if lambda_s <= 0:
        return a_min
    if a_min <= 0:
        raise ValueError("a_min must be positive")
    if continuation_iters <= 0:
        raise ValueError("continuation_iters must be positive")

    a_max = fraction / lambda_s
    exponent = min(iteration, continuation_iters) / continuation_iters
    return float(a_min * (a_max / a_min) ** exponent)


def log_threshold_nonnegative(v, threshold, a):
    """
    Nonnegative log-penalty proximal operator, Eq. (24).

    threshold = lambda_s/rho
    """
    v = np.asarray(v, dtype=np.float64)

    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    if a <= 0:
        raise ValueError("a must be positive")

    y = np.zeros_like(v)
    mask = v >= threshold
    vm = v[mask]

    radicand = (
        (0.5 * vm + 0.5 / a) ** 2
        - threshold / a
    )

    # Protect against tiny negative roundoff.
    radicand = np.maximum(radicand, 0.0)

    y[mask] = (
        0.5 * vm
        - 0.5 / a
        + np.sqrt(radicand)
    )

    return y


def log_objective(X, S, operator, lambda_s, lambda2, tau, a):
    AX = operator.forward(X)
    data = 0.5 * np.sum((AX - S) ** 2)
    sparsity = log_penalty(X, lambda_s, a)

    X_legacy = structured_to_softmin(X)
    softmin_value = softmin(
        X_legacy,
        K=operator.K,
        J=operator.J,
        tau=tau,
        pad_mode="zero",
        normalize=False,
        return_gradient=False,
    )

    temporal = lambda2 * softmin_value

    return {
        "data": float(data),
        "sparsity": float(sparsity),
        "softmin": float(softmin_value),
        "weighted_softmin": float(temporal),
        "total": float(data + sparsity + temporal),
    }


def solve_log_admm(
    X0, S, operator, lambda_s, lambda2=0.0, tau=1e-2,
    rho=1.0, max_admm_iter=200, x_maxiter=50,
    abs_tol=1e-4, rel_tol=1e-3, adaptive_rho=True,
    rho_mu=10.0, rho_tau=2.0, a_min=1e-4,
    continuation_iters=100, a_fraction=0.95, verbose=True,
    temporal_regularizer=None,
):
    X = np.asarray(X0, dtype=np.float64).copy()
    S = np.asarray(S, dtype=np.float64)
    validate_admm_problem(X, S, operator)

    if lambda_s <= 0:
        raise ValueError("lambda_s must be positive")
    if rho <= 0:
        raise ValueError("rho must be positive")

    Y = np.zeros_like(X)
    U = np.zeros_like(X)
    history = []
    nvar = X.size
    converged = False

    if verbose:
        print("\nLog-penalty ADMM")
        print("----------------")
        print("iter   objective       data   log-penalty"
              "     primal       dual       rho         a")

    for iteration in range(1, max_admm_iter + 1):
        X, x_result = solve_x(
            X0=X, S=S, Y=Y, U=U, operator=operator,
            rho=rho, lambda2=lambda2, tau=tau,
            maxiter=x_maxiter, ftol=1e-10, gtol=1e-7, maxls=50,
            temporal_regularizer=temporal_regularizer,
        )

        Y_old = Y.copy()
        a = continuation_a(
            iteration, lambda_s, a_min,
            continuation_iters, a_fraction,
        )

        Y = log_threshold_nonnegative(
            X + U,
            threshold=lambda_s / rho,
            a=a,
        )

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
            Y, S, operator, lambda_s, lambda2, tau, a
        )

        record = {
            "iteration": iteration,
            "objective": terms["total"],
            "data": terms["data"],
            "sparsity": terms["sparsity"],
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

        if verbose and (iteration == 1 or iteration % 5 == 0):
            print(
                f"{iteration:4d}  {terms['total']:11.4e}"
                f"  {terms['data']:9.3e}  {terms['sparsity']:11.3e}"
                f"  {primal:9.3e}  {dual:9.3e}"
                f"  {rho:8.2e}  {a:8.2e}"
            )

        continuation_complete = iteration >= continuation_iters

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

    info = {
        "converged": converged,
        "iterations": iteration,
        "rho": float(rho),
        "a": float(last["a"]),
        "history": history,
        "continuation_complete": iteration >= continuation_iters,
        "final_objective": last["objective"],
        "final_primal_residual": last["primal_residual"],
        "final_dual_residual": last["dual_residual"],
        "nonzero_count": last["nonzero_count"],
    }

    return X, Y, U, info
