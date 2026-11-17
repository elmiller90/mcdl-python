import numpy as np
from scipy.optimize import minimize

from softmin import softmin


def structured_to_softmin(X):
    """(L,K,J,T) -> (1,L,K*J,T), atom-major."""
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 4:
        raise ValueError("X must have shape (L,K,J,T)")

    L, K, J, T = X.shape
    return X.reshape(1, L, K * J, T)


def softmin_to_structured(X_legacy, K, J):
    """(1,L,K*J,T) -> (L,K,J,T)."""
    X_legacy = np.asarray(X_legacy, dtype=np.float64)

    if X_legacy.ndim != 4:
        raise ValueError(
            "Legacy X must have shape (1,L,K*J,T)"
        )

    if X_legacy.shape[0] != 1:
        raise ValueError(
            "Legacy X first dimension must be 1"
        )

    _, L, KJ, T = X_legacy.shape

    if KJ != K * J:
        raise ValueError(
            f"Expected {K * J} channels, got {KJ}"
        )

    return X_legacy.reshape(L, K, J, T)


def validate_problem(X, S, Y, U, operator):
    """
    Validate the fixed-dictionary coefficient problem.

    X,Y,U : (L,K,J,T)
    S     : (N,T)
    """
    X = np.asarray(X)
    S = np.asarray(S)
    Y = np.asarray(Y)
    U = np.asarray(U)

    if X.ndim != 4:
        raise ValueError("X must have shape (L,K,J,T)")

    if Y.shape != X.shape:
        raise ValueError(
            f"Y shape {Y.shape} must match X shape {X.shape}"
        )

    if U.shape != X.shape:
        raise ValueError(
            f"U shape {U.shape} must match X shape {X.shape}"
        )

    L, K, J, T = X.shape

    if L != operator.L:
        raise ValueError(
            f"X has L={L}, but operator.L={operator.L}"
        )

    if K != operator.K:
        raise ValueError(
            f"X has K={K}, but operator.K={operator.K}"
        )

    if J != operator.J:
        raise ValueError(
            f"X has J={J}, but operator.J={operator.J}"
        )

    expected = (operator.N, T)

    if S.shape != expected:
        raise ValueError(
            f"S must have shape {expected}, got {S.shape}"
        )


def make_softmin_regularizer(
    K,
    J,
    tau=1e-2,
    pad_mode="zero",
    normalize=False,
):
    """
    Build a callback satisfying:

        value, grad = callback(X)

    where X and grad have shape (L,K,J,T).
    """
    if tau <= 0:
        raise ValueError("tau must be positive")

    def regularizer(X):
        X_legacy = structured_to_softmin(X)

        value, grad_legacy = softmin(
            X_legacy,
            K=K,
            J=J,
            tau=tau,
            pad_mode=pad_mode,
            normalize=normalize,
            return_gradient=True,
        )

        grad = softmin_to_structured(
            grad_legacy,
            K=K,
            J=J,
        )

        return float(value), grad

    return regularizer


def evaluate_temporal_regularizer(
    X,
    lambda2,
    temporal_regularizer,
):
    """
    Evaluate and weight a temporal callback.

    The callback must return:

        raw_value, raw_grad = temporal_regularizer(X)

    Weighting by lambda2 is performed here exactly once.
    """
    if lambda2 < 0:
        raise ValueError("lambda2 must be nonnegative")

    if lambda2 == 0 or temporal_regularizer is None:
        return 0.0, np.zeros_like(X), 0.0

    result = temporal_regularizer(X)

    if not isinstance(result, tuple) or len(result) < 2:
        raise TypeError(
            "Temporal callback must return (value, grad)"
        )

    raw_value = float(result[0])
    raw_grad = np.asarray(result[1], dtype=np.float64)

    if not np.isfinite(raw_value):
        raise FloatingPointError(
            "Non-finite temporal regularizer value"
        )

    if raw_grad.shape != X.shape:
        raise ValueError(
            "Temporal gradient shape "
            f"{raw_grad.shape} does not match X shape {X.shape}"
        )

    if not np.all(np.isfinite(raw_grad)):
        raise FloatingPointError(
            "Non-finite temporal regularizer gradient"
        )

    return (
        float(lambda2 * raw_value),
        lambda2 * raw_grad,
        raw_value,
    )


def x_objective_value_grad(
    X,
    S,
    Y,
    U,
    operator,
    rho,
    lambda2=0.0,
    temporal_regularizer=None,
    return_components=False,
):
    """
    Structured fixed-dictionary X objective:

        0.5 ||A X - S||_F^2
      + 0.5 rho ||X - Y + U||_F^2
      + lambda2 R_temporal(X)

    X,Y,U : (L,K,J,T)
    S     : (N,T)
    """
    X = np.asarray(X, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)

    if rho < 0:
        raise ValueError("rho must be nonnegative")

    if lambda2 < 0:
        raise ValueError("lambda2 must be nonnegative")

    validate_problem(
        X=X,
        S=S,
        Y=Y,
        U=U,
        operator=operator,
    )

    residual = operator.forward(X) - S

    data_term = 0.5 * np.sum(residual * residual)

    data_grad = operator.adjoint(
        residual,
        structured=True,
    )

    admm_residual = X - Y + U

    admm_term = (
        0.5
        * rho
        * np.sum(admm_residual * admm_residual)
    )

    admm_grad = rho * admm_residual

    temporal_term, temporal_grad, raw_temporal = (
        evaluate_temporal_regularizer(
            X=X,
            lambda2=lambda2,
            temporal_regularizer=temporal_regularizer,
        )
    )

    value = data_term + admm_term + temporal_term

    grad = data_grad + admm_grad + temporal_grad

    if return_components:
        components = {
            "data": float(data_term),
            "admm": float(admm_term),
            "temporal": float(raw_temporal),
            "weighted_temporal": float(temporal_term),
            "total": float(value),
        }

        return float(value), grad, components

    return float(value), grad


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
    temporal_regularizer=None,
):
    """
    Flat L-BFGS wrapper.

    If temporal_regularizer is None, the historical softmin
    regularizer is constructed automatically.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")

    X = np.asarray(
        x_flat,
        dtype=np.float64,
    ).reshape(shape)

    if temporal_regularizer is None:
        temporal_regularizer = make_softmin_regularizer(
            K=operator.K,
            J=operator.J,
            tau=tau,
            pad_mode=pad_mode,
            normalize=normalize_softmin,
        )

    result = x_objective_value_grad(
        X=X,
        S=S,
        Y=Y,
        U=U,
        operator=operator,
        rho=rho,
        lambda2=lambda2,
        temporal_regularizer=temporal_regularizer,
        return_components=return_components,
    )

    if return_components:
        value, grad, components = result
        return value, grad.ravel(), components

    value, grad = result
    return value, grad.ravel()


def solve_x(
    X0, S, Y, U, operator,
    rho=1.0, lambda2=1.0, tau=1e-2,
    pad_mode="zero", normalize_softmin=False,
    maxiter=100, ftol=1e-12, gtol=1e-8,
    maxls=40, temporal_regularizer=None,
):
    X0 = np.asarray(X0, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)

    validate_problem(X0, S, Y, U, operator)

    shape = X0.shape

    # Ensure feasible initial point.
    X0 = np.maximum(X0, 0.0)

    result = minimize(
        x_objective,
        X0.ravel(),
        args=(
            shape, S, Y, U, operator,
            rho, lambda2, tau,
            pad_mode, normalize_softmin,
            False, temporal_regularizer,
        ),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * X0.size,
        options={
            "maxiter": int(maxiter),
            "ftol": float(ftol),
            "gtol": float(gtol),
            "maxls": int(maxls),
        },
    )

    return result.x.reshape(shape), result

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
    temporal_regularizer=None,
):
    """
    Evaluate separate terms of the fixed-D coefficient objective.

    Default behavior remains softmin.
    """
    X = np.asarray(X, dtype=np.float64)

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
        temporal_regularizer,
    )

    return components