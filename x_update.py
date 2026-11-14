import numpy as np
from scipy.optimize import minimize

from softmin import softmin


def structured_to_softmin(X):
    """
    Convert canonical Python coefficients

        (L, K, J, T)

    to the MATLAB-compatible layout expected by softmin():

        (1, L, K*J, T)

    Channels are flattened in atom-major order:

        (k=0,j=0), (k=0,j=1), ...,
        (k=1,j=0), ...
    """
    X = np.asarray(
        X,
        dtype=np.float64,
    )

    if X.ndim != 4:
        raise ValueError(
            "X must have shape (L, K, J, T)"
        )

    L, K, J, T = X.shape

    return X.reshape(
        1,
        L,
        K * J,
        T,
    )


def softmin_to_structured(
    X_legacy,
    K,
    J,
):
    """
    Convert softmin layout

        (1, L, K*J, T)

    back to

        (L, K, J, T).
    """
    X_legacy = np.asarray(
        X_legacy,
        dtype=np.float64,
    )

    if X_legacy.ndim != 4:
        raise ValueError(
            "Legacy X must have shape "
            "(1, L, K*J, T)"
        )

    if X_legacy.shape[0] != 1:
        raise ValueError(
            "Legacy X first dimension must be 1"
        )

    _, L, KJ, T = X_legacy.shape

    if KJ != K * J:
        raise ValueError(
            f"Expected {K * J} channels, "
            f"got {KJ}"
        )

    return X_legacy.reshape(
        L,
        K,
        J,
        T,
    )


def validate_problem(
    X,
    S,
    Y,
    U,
    operator,
):
    """
    Validate the MATLAB-compatible fixed-D X problem.

    Coefficients:
        X,Y,U : (L,K,J,T)

    where
        L = N + M - 1

    Measurements:
        S : (N,T)
    """
    X = np.asarray(X)
    S = np.asarray(S)
    Y = np.asarray(Y)
    U = np.asarray(U)

    if X.ndim != 4:
        raise ValueError(
            "X must have shape (L, K, J, T)"
        )

    if Y.shape != X.shape:
        raise ValueError(
            f"Y shape {Y.shape} must match "
            f"X shape {X.shape}"
        )

    if U.shape != X.shape:
        raise ValueError(
            f"U shape {U.shape} must match "
            f"X shape {X.shape}"
        )

    L, K, J, T = X.shape

    if L != operator.L:
        raise ValueError(
            f"X has L={L}, but operator.L="
            f"{operator.L}"
        )

    if K != operator.K:
        raise ValueError(
            f"X has K={K}, but operator.K="
            f"{operator.K}"
        )

    if J != operator.J:
        raise ValueError(
            f"X has J={J}, but operator.J="
            f"{operator.J}"
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


def x_objective(
    x_flat,
    shape,
    S,
    Y,
    U,
    operator,
    rho,
    lambda2,
    tau,
    pad_mode="zero",
    normalize_softmin=False,
    return_components=False,
):
    """
    Fixed-dictionary MCDL coefficient objective.

        f(X)
        =
        0.5 ||A_D X - S||_F^2
        + 0.5 rho ||X - Y + U||_F^2
        + lambda2 R_softmin(X)

    where

        X,Y,U : (L,K,J,T)
        S     : (N,T)

    and

        L = N + M - 1.
    """
    if rho < 0:
        raise ValueError(
            "rho must be nonnegative"
        )

    if lambda2 < 0:
        raise ValueError(
            "lambda2 must be nonnegative"
        )

    if tau <= 0:
        raise ValueError(
            "tau must be positive"
        )

    X = np.asarray(
        x_flat,
        dtype=np.float64,
    ).reshape(shape)

    validate_problem(
        X=X,
        S=S,
        Y=Y,
        U=U,
        operator=operator,
    )

    # -----------------------------------------
    # Data fidelity
    #
    # A_D maps:
    #
    #   (L,K,J,T) -> (N,T)
    #
    # and A_D^* maps:
    #
    #   (N,T) -> (L,K,J,T)
    # -----------------------------------------

    AX = operator.forward(X)

    residual = AX - S

    data_term = (
        0.5
        * np.sum(
            residual ** 2
        )
    )

    data_grad = operator.adjoint(
        residual,
        structured=True,
    )

    # -----------------------------------------
    # ADMM quadratic
    # -----------------------------------------

    admm_residual = (
        X - Y + U
    )

    admm_term = (
        0.5
        * rho
        * np.sum(
            admm_residual ** 2
        )
    )

    admm_grad = (
        rho
        * admm_residual
    )

    # -----------------------------------------
    # Softmin
    #
    # Python canonical:
    #
    #   (L,K,J,T)
    #
    # softmin legacy layout:
    #
    #   (1,L,K*J,T)
    # -----------------------------------------

    X_legacy = structured_to_softmin(
        X
    )

    (
        softmin_value,
        softmin_grad_legacy,
    ) = softmin(
        X_legacy,
        K=operator.K,
        J=operator.J,
        tau=tau,
        pad_mode=pad_mode,
        normalize=normalize_softmin,
        return_gradient=True,
    )

    softmin_grad = (
        softmin_to_structured(
            softmin_grad_legacy,
            K=operator.K,
            J=operator.J,
        )
    )

    regularizer_term = (
        lambda2
        * softmin_value
    )

    regularizer_grad = (
        lambda2
        * softmin_grad
    )

    # -----------------------------------------
    # Total
    # -----------------------------------------

    value = (
        data_term
        + admm_term
        + regularizer_term
    )

    grad = (
        data_grad
        + admm_grad
        + regularizer_grad
    )

    if return_components:
        components = {
            "data": float(
                data_term
            ),
            "admm": float(
                admm_term
            ),
            "softmin": float(
                softmin_value
            ),
            "weighted_softmin": float(
                regularizer_term
            ),
            "total": float(
                value
            ),
        }

        return (
            float(value),
            grad.ravel(),
            components,
        )

    return (
        float(value),
        grad.ravel(),
    )


def solve_x(
    X0,
    S,
    Y,
    U,
    operator,
    rho=1.0,
    lambda2=1.0,
    tau=1e-2,
    pad_mode="zero",
    normalize_softmin=False,
    maxiter=100,
    ftol=1e-12,
    gtol=1e-8,
    maxls=40,
):
    """
    Solve the fixed-dictionary coefficient subproblem
    using L-BFGS-B.

    Coefficient arrays use

        (L,K,J,T),

    where

        L = N + M - 1.

    The physical signal uses

        (N,T).
    """
    X0 = np.asarray(
        X0,
        dtype=np.float64,
    )

    S = np.asarray(
        S,
        dtype=np.float64,
    )

    Y = np.asarray(
        Y,
        dtype=np.float64,
    )

    U = np.asarray(
        U,
        dtype=np.float64,
    )

    validate_problem(
        X=X0,
        S=S,
        Y=Y,
        U=U,
        operator=operator,
    )

    shape = X0.shape

    result = minimize(
        x_objective,
        X0.ravel(),
        args=(
            shape,
            S,
            Y,
            U,
            operator,
            rho,
            lambda2,
            tau,
            pad_mode,
            normalize_softmin,
            False,
        ),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(
                maxiter
            ),
            "ftol": float(
                ftol
            ),
            "gtol": float(
                gtol
            ),
            "maxls": int(
                maxls
            ),
        },
    )

    X = result.x.reshape(
        shape
    )

    return X, result


def objective_components(
    X,
    S,
    Y,
    U,
    operator,
    rho=1.0,
    lambda2=1.0,
    tau=1e-2,
    pad_mode="zero",
    normalize_softmin=False,
):
    """
    Evaluate the separate terms of the fixed-D
    coefficient objective.
    """
    X = np.asarray(
        X,
        dtype=np.float64,
    )

    _, _, components = x_objective(
        X.ravel(),
        X.shape,
        S,
        Y,
        U,
        operator,
        rho,
        lambda2,
        tau,
        pad_mode,
        normalize_softmin,
        True,
    )

    return components
