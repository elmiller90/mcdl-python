import numpy as np
from scipy.optimize import minimize

from softmin import softmin


def structured_to_softmin(X):
    """
    Convert the canonical Python coefficient layout

        (N, K, J, T)

    to the legacy MATLAB-compatible layout expected by softmin():

        (1, N, K*J, T)

    Channels are flattened in atom-major order:

        (k=0,j=0), (k=0,j=1), ...,
        (k=1,j=0), ...
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 4:
        raise ValueError(
            "X must have shape (N, K, J, T)"
        )

    N, K, J, T = X.shape

    return X.reshape(1, N, K * J, T)


def softmin_to_structured(X_legacy, K, J):
    """
    Convert the legacy softmin layout

        (1, N, K*J, T)

    back to the canonical Python layout

        (N, K, J, T).
    """
    X_legacy = np.asarray(
        X_legacy,
        dtype=np.float64,
    )

    if X_legacy.ndim != 4:
        raise ValueError(
            "Legacy X must have shape (1, N, K*J, T)"
        )

    if X_legacy.shape[0] != 1:
        raise ValueError(
            "Legacy X first dimension must be 1"
        )

    _, N, KJ, T = X_legacy.shape

    if KJ != K * J:
        raise ValueError(
            f"Expected {K * J} channels, got {KJ}"
        )

    return X_legacy.reshape(N, K, J, T)


def validate_problem(
    X,
    S,
    Y,
    U,
    operator,
):
    """
    Validate dimensions for the fixed-dictionary X problem.
    """
    X = np.asarray(X)
    S = np.asarray(S)
    Y = np.asarray(Y)
    U = np.asarray(U)

    expected_x_shape = (
        operator.N,
        operator.K,
        operator.J,
        X.shape[3] if X.ndim == 4 else None,
    )

    if X.ndim != 4:
        raise ValueError(
            "X must have shape (N, K, J, T)"
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

    N, K, J, T = X.shape

    if N != operator.N:
        raise ValueError(
            f"X has N={N}, but operator.N="
            f"{operator.N}"
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

    expected_s_shape = (N, T)

    if S.shape != expected_s_shape:
        raise ValueError(
            f"S must have shape {expected_s_shape}, "
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

    The objective is

        0.5 ||A_D X - S||_F^2
        + 0.5 rho ||X - Y + U||_F^2
        + lambda2 R_softmin(X).

    Parameters
    ----------
    x_flat : ndarray
        Flattened coefficient array.

    shape : tuple
        Canonical shape (N, K, J, T).

    S : ndarray, shape (N, T)
        Observed signals.

    Y, U : ndarray, shape (N, K, J, T)
        ADMM auxiliary and scaled-dual variables.

    operator :
        MultiscaleConvolutionOperator instance.

    rho : float
        ADMM penalty coefficient.

    lambda2 : float
        Softmin regularization coefficient.

    tau : float
        Softmin temperature.

    return_components : bool
        If True, additionally return objective components.

    Returns
    -------
    value : float
        Objective value.

    grad_flat : ndarray
        Flattened analytical gradient.

    components : dict, optional
        Unweighted and weighted objective components.
    """
    if rho < 0:
        raise ValueError("rho must be nonnegative")

    if lambda2 < 0:
        raise ValueError(
            "lambda2 must be nonnegative"
        )

    if tau <= 0:
        raise ValueError("tau must be positive")

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

    # Data-fidelity term:
    #
    #   0.5 ||A_D X - S||^2
    AX = operator.forward(X)
    residual = AX - S

    data_term = 0.5 * np.sum(
        residual**2
    )

    data_grad = operator.adjoint(
        residual,
        structured=True,
    )

    # ADMM quadratic:
    #
    #   0.5 rho ||X - Y + U||^2
    admm_residual = X - Y + U

    admm_term = 0.5 * rho * np.sum(
        admm_residual**2
    )

    admm_grad = rho * admm_residual

    # Softmin still uses the original
    # MATLAB-compatible coefficient layout.
    X_legacy = structured_to_softmin(X)

    softmin_value, softmin_grad_legacy = softmin(
        X_legacy,
        K=operator.K,
        J=operator.J,
        tau=tau,
        pad_mode=pad_mode,
        normalize=normalize_softmin,
        return_gradient=True,
    )

    softmin_grad = softmin_to_structured(
        softmin_grad_legacy,
        K=operator.K,
        J=operator.J,
    )

    regularizer_term = (
        lambda2 * softmin_value
    )

    regularizer_grad = (
        lambda2 * softmin_grad
    )

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
            "data": float(data_term),
            "admm": float(admm_term),
            "softmin": float(softmin_value),
            "weighted_softmin": float(
                regularizer_term
            ),
            "total": float(value),
        }

        return (
            float(value),
            grad.ravel(),
            components,
        )

    return float(value), grad.ravel()


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

    All coefficient arrays use shape

        (N, K, J, T).
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
            "maxiter": int(maxiter),
            "ftol": float(ftol),
            "gtol": float(gtol),
            "maxls": int(maxls),
        },
    )

    X = result.x.reshape(shape)

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
    Evaluate and return the separate objective terms.
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
