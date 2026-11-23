import numpy as np

from sparse_admm import (
    l1_objective,
    soft_threshold,
    validate_admm_problem,
)
from x_update import solve_x


def _state_array(value, shape, name, default):
    if value is None:
        return np.asarray(default, dtype=float).copy()

    value = np.asarray(value, dtype=float)
    if value.shape != shape:
        raise ValueError(
            f"{name} must have shape {shape}, got {value.shape}"
        )
    return value.copy()


def solve_l1_admm_stateful(
    X0,
    S,
    operator,
    lambda1,
    lambda2=0.0,
    tau=1e-2,
    rho=1.0,
    max_admm_iter=1,
    x_maxiter=50,
    abs_tol=1e-4,
    rel_tol=1e-3,
    nonnegative=True,
    adaptive_rho=True,
    rho_mu=10.0,
    rho_tau=2.0,
    verbose=False,
    state0=None,
):
    """L1 ADMM with optional persistent X,Y,U,rho state."""
    Xbase = np.asarray(X0, dtype=float)
    S = np.asarray(S, dtype=float)
    validate_admm_problem(Xbase, S, operator)

    if lambda1 < 0 or lambda2 < 0:
        raise ValueError("regularization weights must be nonnegative")
    if rho <= 0:
        raise ValueError("rho must be positive")
    if max_admm_iter <= 0:
        raise ValueError("max_admm_iter must be positive")

    state0 = {} if state0 is None else dict(state0)
    shape = Xbase.shape

    X = _state_array(
        state0.get("X"), shape, "X", Xbase
    )

    if state0.get("Y") is None:
        Ydefault = soft_threshold(
            X, lambda1/rho,
            nonnegative=nonnegative,
        )
    else:
        Ydefault = np.zeros_like(X)

    Y = _state_array(
        state0.get("Y"), shape, "Y", Ydefault
    )
    U = _state_array(
        state0.get("U"), shape, "U", np.zeros_like(X)
    )

    rho = float(state0.get("rho", rho))
    if rho <= 0:
        raise ValueError("state rho must be positive")

    history = []
    converged = False
    nvar = X.size

    for iteration in range(1, max_admm_iter + 1):
        X, x_result = solve_x(
            X0=X,
            S=S,
            Y=Y,
            U=U,
            operator=operator,
            rho=rho,
            lambda2=lambda2,
            tau=tau,
            maxiter=x_maxiter,
            ftol=1e-10,
            gtol=1e-7,
            maxls=50,
        )

        Yprev = Y.copy()

        Y = soft_threshold(
            X + U,
            lambda1/rho,
            nonnegative=nonnegative,
        )

        primal = X - Y
        U = U + primal

        r = np.linalg.norm(primal)
        s = rho*np.linalg.norm(Y-Yprev)

        eps_pri = (
            np.sqrt(nvar)*abs_tol
            + rel_tol*max(np.linalg.norm(X), np.linalg.norm(Y))
        )
        eps_dual = (
            np.sqrt(nvar)*abs_tol
            + rel_tol*rho*np.linalg.norm(U)
        )

        obj = l1_objective(
            Y, S, operator,
            lambda1, lambda2, tau,
        )

        history.append({
            "iteration": iteration,
            "objective": obj["total"],
            "data": obj["data"],
            "l1": obj["l1"],
            "primal_residual": float(r),
            "dual_residual": float(s),
            "rho": float(rho),
            "x_success": bool(x_result.success),
            "x_iterations": int(x_result.nit),
        })

        if r <= eps_pri and s <= eps_dual:
            converged = True
            break

        if adaptive_rho:
            old_rho = rho

            if r > rho_mu*s:
                rho *= rho_tau
            elif s > rho_mu*r:
                rho /= rho_tau

            if rho != old_rho:
                U *= old_rho/rho

    state = {
        "X": X.copy(),
        "Y": Y.copy(),
        "U": U.copy(),
        "rho": float(rho),
    }

    info = {
        "converged": converged,
        "iterations": iteration,
        "rho": float(rho),
        "history": history,
        "final_primal_residual": float(r),
        "final_dual_residual": float(s),
        "nonzero_count": int(np.count_nonzero(Y)),
        "state": state,
    }

    if verbose:
        print(
            f"L1 stateful: iter={iteration} "
            f"r={r:.3e} s={s:.3e} rho={rho:.3e}"
        )

    return X, Y, U, info