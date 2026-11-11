import numpy as np
from scipy.optimize import minimize

from softmin import softmin


def x_objective(
    x_flat,
    shape,
    S,
    Y,
    U,
    K,
    J,
    rho,
    lambda2,
    tau,
):
    X = x_flat.reshape(shape)

    # Data fidelity
    residual = X - S
    data_term = 0.5 * np.sum(residual**2)
    data_grad = residual

    # ADMM quadratic
    admm_residual = X - Y + U
    admm_term = 0.5 * rho * np.sum(admm_residual**2)
    admm_grad = rho * admm_residual

    # Softmin regularizer
    R, grad_R = softmin(
        X,
        K,
        J,
        tau=tau,
        pad_mode="zero",
        normalize=False,
    )

    reg_term = lambda2 * R
    reg_grad = lambda2 * grad_R

    value = data_term + admm_term + reg_term

    grad = (
        data_grad
        + admm_grad
        + reg_grad
    )

    return value, grad.ravel()


def solve_x(
    X0,
    S,
    Y,
    U,
    K,
    J,
    rho=1.0,
    lambda2=1.0,
    tau=1e-2,
    maxiter=100,
):
    shape = X0.shape

    result = minimize(
        x_objective,
        X0.ravel(),
        args=(
            shape,
            S,
            Y,
            U,
            K,
            J,
            rho,
            lambda2,
            tau,
        ),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": maxiter,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 40,
        },
    )

    X = result.x.reshape(shape)

    return X, result