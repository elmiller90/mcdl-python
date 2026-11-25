import numpy as np
from scipy.optimize import minimize

from banco_barycentric_spg import solve_spg
from banco_coefficient_fourier import solve_banco_x_fourier
from banco_sparse_penalty import banco_sparse_update


def _pginf(x, g):
    g = np.asarray(g).copy()
    g[(np.asarray(x) <= 1e-10) & (g > 0)] = 0
    return float(np.linalg.norm(g, np.inf))


def banco_barycentric_coefficient_step(
    X, Y, U, S, operator,
    lambda_s, lambda_t, temporal_regularizer,
    rho, iteration, x_relax=1.8,
    maxiter=400, use_fourier_init=True,
    solver="lbfgsb", gtol=1e-4,
):
    X0 = (
        solve_banco_x_fourier(operator, S, Y, U, rho)
        if use_fourier_init else X.copy()
    )
    shape = X.shape

    def fg(x):
        Z = x.reshape(shape)
        R = operator.forward(Z)-S
        Q = Z-Y+U
        f = 0.5*np.sum(R*R)+0.5*rho*np.sum(Q*Q)
        g = operator.adjoint(R, structured=True)+rho*Q

        if lambda_t > 0:
            rt, gt = temporal_regularizer(Z)
            f += lambda_t*rt
            g += lambda_t*gt

        return float(f), g.ravel()

    if solver == "spg":
        result = solve_spg(
            fg, X0.ravel(),
            maxiter=maxiter, gtol=gtol,
            alpha0=1/max(rho, 1e-12),
        )
    elif solver == "lbfgsb":
        result = minimize(
            fg, X0.ravel(), method="L-BFGS-B", jac=True,
            bounds=[(0, None)]*X0.size,
            options={
                "maxiter": maxiter,
                "ftol": 1e-10,
                "gtol": gtol,
                "maxls": 50,
            },
        )
        result.projected_grad_inf = _pginf(
            result.x, result.jac
        )
    else:
        raise ValueError("solver must be 'lbfgsb' or 'spg'")

    X = result.x.reshape(shape)
    Xr = x_relax*X+(1-x_relax)*Y

    Ynew = banco_sparse_update(
        Xr+U,
        lambda_s=lambda_s, rho=rho,
        iteration=iteration, l1_iters=20,
        nonnegative=True,
    )
    return X, Ynew, U+Xr-Ynew, result