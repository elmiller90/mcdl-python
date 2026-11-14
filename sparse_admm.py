import numpy as np

from softmin import softmin
from x_update import (
    solve_x,
    structured_to_softmin,
)


def soft_threshold(
    x,
    threshold,
    nonnegative=False,
):
    """
    Proximal operator for threshold * ||x||_1.

    Works elementwise on L-domain coefficients.

    Parameters
    ----------
    x : ndarray
        Coefficients of arbitrary shape.

    threshold : float
        Nonnegative shrinkage threshold.

    nonnegative : bool
        If True, additionally enforce x >= 0.
    """
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if threshold < 0:
        raise ValueError(
            "threshold must be nonnegative"
        )

    if nonnegative:
        return np.maximum(
            x - threshold,
            0.0,
        )

    return np.sign(x) * np.maximum(
        np.abs(x) - threshold,
        0.0,
    )


def validate_admm_problem(
    X,
    S,
    operator,
):
    """
    Validate MATLAB-compatible ADMM dimensions.

    Coefficients:
        X : (L,K,J,T)

    Physical measurements:
        S : (N,T)

    where:
        L = N + M - 1.
    """
    X = np.asarray(X)
    S = np.asarray(S)

    if X.ndim != 4:
        raise ValueError(
            "X must have shape (L,K,J,T)"
        )

    L, K, J, T = X.shape

    if L != operator.L:
        raise ValueError(
            f"Coefficient length must be "
            f"L={operator.L}, got {L}"
        )

    if K != operator.K:
        raise ValueError(
            f"Expected K={operator.K}, got {K}"
        )

    if J != operator.J:
        raise ValueError(
            f"Expected J={operator.J}, got {J}"
        )

    expected_s_shape = (
        operator.N,
        T,
    )

    if S.shape != expected_s_shape:
        raise ValueError(
            f"S must have shape "
            f"{expected_s_shape}, "
            f"got {S.shape}"
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
    Evaluate the unsplit objective

        0.5 ||A_D X - S||_F^2
        + lambda1 ||X||_1
        + lambda2 R_softmin(X)

    with

        X : (L,K,J,T)
        S : (N,T).
    """
    X = np.asarray(
        X,
        dtype=np.float64,
    )

    S = np.asarray(
        S,
        dtype=np.float64,
    )

    validate_admm_problem(
        X,
        S,
        operator,
    )

    AX = operator.forward(X)

    residual = AX - S

    data_term = (
        0.5
        * np.sum(
            residual ** 2
        )
    )

    l1_term = (
        lambda1
        * np.sum(
            np.abs(X)
        )
    )

    X_legacy = structured_to_softmin(
        X
    )

    softmin_value = softmin(
        X_legacy,
        K=operator.K,
        J=operator.J,
        tau=tau,
        pad_mode="zero",
        normalize=False,
        return_gradient=False,
    )

    weighted_softmin = (
        lambda2
        * softmin_value
    )

    total = (
        data_term
        + l1_term
        + weighted_softmin
    )

    return {
        "data": float(
            data_term
        ),
        "l1": float(
            l1_term
        ),
        "softmin": float(
            softmin_value
        ),
        "weighted_softmin": float(
            weighted_softmin
        ),
        "total": float(
            total
        ),
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
    Solve

        min_X
            0.5 ||A_D X - S||_F^2
            + lambda1 ||X||_1
            + lambda2 R_softmin(X)

    using scaled-form ADMM with splitting X = Y.

    Dimensions
    ----------
    X,Y,U :
        (L,K,J,T)

    S :
        (N,T)

    L :
        N + M - 1

    Returns
    -------
    X :
        Smooth/data-fitting ADMM variable.

    Y :
        Sparse coefficient estimate.

    U :
        Scaled dual variable.

    info :
        Convergence and diagnostic information.
    """
    X = np.asarray(
        X0,
        dtype=np.float64,
    ).copy()

    S = np.asarray(
        S,
        dtype=np.float64,
    )

    validate_admm_problem(
        X,
        S,
        operator,
    )

    if lambda1 < 0:
        raise ValueError(
            "lambda1 must be nonnegative"
        )

    if lambda2 < 0:
        raise ValueError(
            "lambda2 must be nonnegative"
        )

    if tau <= 0:
        raise ValueError(
            "tau must be positive"
        )

    if rho <= 0:
        raise ValueError(
            "rho must be positive"
        )

    if max_admm_iter <= 0:
        raise ValueError(
            "max_admm_iter must be positive"
        )

    if x_maxiter <= 0:
        raise ValueError(
            "x_maxiter must be positive"
        )

    if abs_tol < 0:
        raise ValueError(
            "abs_tol must be nonnegative"
        )

    if rel_tol < 0:
        raise ValueError(
            "rel_tol must be nonnegative"
        )

    if adaptive_rho:
        if rho_mu <= 1:
            raise ValueError(
                "rho_mu must be greater than 1"
            )

        if rho_tau <= 1:
            raise ValueError(
                "rho_tau must be greater than 1"
            )

    # -------------------------------------------------
    # ADMM initialization
    # -------------------------------------------------

    Y = soft_threshold(
        X,
        lambda1 / rho,
        nonnegative=nonnegative,
    )

    U = np.zeros_like(X)

    history = []

    n_variables = X.size

    converged = False

    if verbose:
        print()
        print("L1 ADMM - extended coefficient domain")
        print("------------------------------------")

        print(
            f"N={operator.N}, "
            f"M={operator.M}, "
            f"L={operator.L}, "
            f"K={operator.K}, "
            f"J={operator.J}, "
            f"T={X.shape[3]}"
        )

        print()

        print(
            "iter   objective       data"
            "          l1       primal"
            "       dual       rho"
        )

    # -------------------------------------------------
    # ADMM iterations
    # -------------------------------------------------

    for iteration in range(
        1,
        max_admm_iter + 1,
    ):
        # ---------------------------------------------
        # X update
        #
        # X lives on the extended L-domain.
        # operator.forward(X) returns physical N-domain
        # measurements.
        # ---------------------------------------------

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

        # ---------------------------------------------
        # Y update: L1 proximal operator
        # ---------------------------------------------

        Y_previous = Y.copy()

        Y = soft_threshold(
            X + U,
            lambda1 / rho,
            nonnegative=nonnegative,
        )

        # ---------------------------------------------
        # Scaled dual update
        # ---------------------------------------------

        primal_array = X - Y

        U = U + primal_array

        # ---------------------------------------------
        # ADMM residuals
        # ---------------------------------------------

        primal_residual = np.linalg.norm(
            primal_array
        )

        dual_residual = (
            rho
            * np.linalg.norm(
                Y - Y_previous
            )
        )

        primal_tolerance = (
            np.sqrt(n_variables)
            * abs_tol
            + rel_tol
            * max(
                np.linalg.norm(X),
                np.linalg.norm(Y),
            )
        )

        dual_tolerance = (
            np.sqrt(n_variables)
            * abs_tol
            + rel_tol
            * rho
            * np.linalg.norm(U)
        )

        # Y is the sparse variable and is therefore
        # used to report the unsplit objective.
        components = l1_objective(
            X=Y,
            S=S,
            operator=operator,
            lambda1=lambda1,
            lambda2=lambda2,
            tau=tau,
        )

        exact_nonzero_count = int(
            np.count_nonzero(Y)
        )

        record = {
            "iteration": iteration,
            "objective": components[
                "total"
            ],
            "data": components[
                "data"
            ],
            "l1": components[
                "l1"
            ],
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
            "rho": float(
                rho
            ),
            "nonzero_count":
                exact_nonzero_count,
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

        # ---------------------------------------------
        # Stopping criterion
        # ---------------------------------------------

        if (
            primal_residual
            <= primal_tolerance
            and dual_residual
            <= dual_tolerance
        ):
            converged = True
            break

        # ---------------------------------------------
        # Residual-balanced adaptive rho
        # ---------------------------------------------

        if adaptive_rho:
            old_rho = rho

            if (
                primal_residual
                > rho_mu
                * dual_residual
            ):
                rho *= rho_tau

            elif (
                dual_residual
                > rho_mu
                * primal_residual
            ):
                rho /= rho_tau

            # U is the scaled dual variable.
            #
            # Preserve the corresponding unscaled
            # dual variable when rho changes.
            if rho != old_rho:
                U *= (
                    old_rho
                    / rho
                )

    # -------------------------------------------------
    # Final diagnostics
    # -------------------------------------------------

    final_record = history[-1]

    info = {
        "converged":
            converged,
        "iterations":
            iteration,
        "N":
            int(operator.N),
        "M":
            int(operator.M),
        "L":
            int(operator.L),
        "rho":
            float(rho),
        "history":
            history,
        "final_objective":
            final_record[
                "objective"
            ],
        "final_primal_residual":
            final_record[
                "primal_residual"
            ],
        "final_dual_residual":
            final_record[
                "dual_residual"
            ],
        "final_primal_tolerance":
            final_record[
                "primal_tolerance"
            ],
        "final_dual_tolerance":
            final_record[
                "dual_tolerance"
            ],
        "nonzero_count":
            final_record[
                "nonzero_count"
            ],
    }

    return (
        X,
        Y,
        U,
        info,
    )
