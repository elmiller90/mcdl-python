import numpy as np

from x_update import solve_x


def soft_threshold(x, threshold, nonnegative=False):
    """
    Proximal operator for threshold * ||x||_1.

    Parameters
    ----------
    x : ndarray
        Input array.

    threshold : float
        Nonnegative shrinkage threshold.

    nonnegative : bool
        If True, enforce nonnegative coefficients.

    Returns
    -------
    y : ndarray
        Thresholded coefficients.
    """
    x = np.asarray(x, dtype=np.float64)

    if threshold < 0:
        raise ValueError(
            "threshold must be nonnegative"
        )

    if nonnegative:
        return np.maximum(x - threshold, 0.0)

    return np.sign(x) * np.maximum(
        np.abs(x) - threshold,
        0.0,
    )


def l1_objective(
    X,
    S,
    operator,
    lambda1,
    lambda2,
    tau,
):
    """
    Evaluate the unsplit optimization objective.

        0.5 ||A X - S||^2
        + lambda1 ||X||_1
        + lambda2 R_softmin(X)
    """
    from softmin import softmin
    from x_update import structured_to_softmin

    AX = operator.forward(X)
    residual = AX - S

    data_term = 0.5 * np.sum(
        residual ** 2
    )

    l1_term = lambda1 * np.sum(
        np.abs(X)
    )

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

    softmin_term = lambda2 * softmin_value

    total = (
        data_term
        + l1_term
        + softmin_term
    )

    return {
        "data": float(data_term),
        "l1": float(l1_term),
        "softmin": float(softmin_value),
        "weighted_softmin": float(
            softmin_term
        ),
        "total": float(total),
    }


def solve_l1_admm(
    X0,
    S,
    operator,
    lambda1,
    lambda2=0.0,
    tau=1e-2,
    rho=1.0,
    max_admm_iter=100,
    x_maxiter=100,
    abs_tol=1e-4,
    rel_tol=1e-3,
    nonnegative=False,
    adaptive_rho=False,
    rho_mu=10.0,
    rho_tau=2.0,
    verbose=True,
):
    """
    Solve the softmin-regularized sparse coefficient problem.

    Parameters
    ----------
    X0 : ndarray, shape (N, K, J, T)
        Initial coefficient estimate.

    S : ndarray, shape (N, T)
        Observed signals.

    operator :
        Fixed MultiscaleConvolutionOperator.

    lambda1 : float
        L1 sparsity coefficient.

    lambda2 : float
        Softmin trajectory coefficient.

    tau : float
        Softmin temperature.

    rho : float
        Initial ADMM penalty.

    max_admm_iter : int
        Maximum outer ADMM iterations.

    x_maxiter : int
        Maximum L-BFGS iterations per X update.

    abs_tol, rel_tol : float
        ADMM stopping tolerances.

    nonnegative : bool
        Enforce Y >= 0 in the proximal update.

    adaptive_rho : bool
        Enable residual-balancing rho updates.

    Returns
    -------
    X : ndarray
        Smooth/data-fitting variable.

    Y : ndarray
        Sparse coefficient estimate.

    U : ndarray
        Scaled dual variable.

    info : dict
        Convergence diagnostic information.
    """
    X = np.asarray(
        X0,
        dtype=np.float64,
    ).copy()

    S = np.asarray(
        S,
        dtype=np.float64,
    )

    if X.ndim != 4:
        raise ValueError(
            "X0 must have shape (N, K, J, T)"
        )

    if S.shape != (X.shape[0], X.shape[3]):
        raise ValueError(
            "S must have shape (N, T)"
        )

    if lambda1 < 0:
        raise ValueError(
            "lambda1 must be nonnegative"
        )

    if lambda2 < 0:
        raise ValueError(
            "lambda2 must be nonnegative"
        )

    if rho <= 0:
        raise ValueError(
            "rho must be positive"
        )

    # A sparse initialization is often preferable to
    # setting Y equal to a dense backprojection.
    Y = soft_threshold(
        X,
        lambda1 / rho,
        nonnegative=nonnegative,
    )

    U = np.zeros_like(X)

    history = []

    n_variables = X.size

    if verbose:
        print()
        print("L1 ADMM")
        print("-------")
        print(
            "iter   objective       data"
            "          l1       primal"
            "       dual       rho"
        )

    converged = False

    for iteration in range(1, max_admm_iter + 1):
        # X update
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

        # Y update
        Y_previous = Y.copy()

        Y = soft_threshold(
            X + U,
            lambda1 / rho,
            nonnegative=nonnegative,
        )

        # Scaled dual update
        primal_array = X - Y
        U = U + primal_array

        # Residuals
        primal_residual = np.linalg.norm(
            primal_array
        )

        dual_residual = rho * np.linalg.norm(
            Y - Y_previous
        )

        primal_tolerance = (
            np.sqrt(n_variables) * abs_tol
            + rel_tol
            * max(
                np.linalg.norm(X),
                np.linalg.norm(Y),
            )
        )

        dual_tolerance = (
            np.sqrt(n_variables) * abs_tol
            + rel_tol
            * rho
            * np.linalg.norm(U)
        )

        components = l1_objective(
            X=Y,
            S=S,
            operator=operator,
            lambda1=lambda1,
            lambda2=lambda2,
            tau=tau,
        )

        nonzero_count = int(
            np.count_nonzero(Y)
        )

        record = {
            "iteration": iteration,
            "objective": components["total"],
            "data": components["data"],
            "l1": components["l1"],
            "softmin": components[
                "weighted_softmin"
            ],
            "primal_residual": float(
                primal_residual
            ),
            "dual_residual": float(
                dual_residual
            ),
            "primal_tolerance": float(
                primal_tolerance
            ),
            "dual_tolerance": float(
                dual_tolerance
            ),
            "rho": float(rho),
            "nonzero_count": nonzero_count,
            "x_success": bool(
                x_result.success
            ),
            "x_iterations": int(
                x_result.nit
            ),
        }

        history.append(record)

        if (
            verbose
            and (
                iteration == 1
                or iteration % 5 == 0
                or iteration
                == max_admm_iter
            )
        ):
            print(
                f"{iteration:4d}"
                f"  {components['total']:11.4e}"
                f"  {components['data']:11.4e}"
                f"  {components['l1']:9.3e}"
                f"  {primal_residual:9.3e}"
                f"  {dual_residual:9.3e}"
                f"  {rho:8.2e}"
            )

        if (
            primal_residual
            <= primal_tolerance
            and dual_residual
            <= dual_tolerance
        ):
            converged = True
            break

        if adaptive_rho:
            old_rho = rho

            if (
                primal_residual
                > rho_mu * dual_residual
            ):
                rho *= rho_tau

            elif (
                dual_residual
                > rho_mu * primal_residual
            ):
                rho /= rho_tau

            # Preserve the unscaled dual variable when rho
            # changes: u_new = (rho_old/rho_new) u_old.
            if rho != old_rho:
                U *= old_rho / rho

    info = {
        "converged": converged,
        "iterations": iteration,
        "rho": float(rho),
        "history": history,
        "final_objective": history[-1][
            "objective"
        ],
        "final_primal_residual": history[-1][
            "primal_residual"
        ],
        "final_dual_residual": history[-1][
            "dual_residual"
        ],
        "nonzero_count": history[-1][
            "nonzero_count"
        ],
    }

    return X, Y, U, info
