from banco_coefficient_fourier import solve_banco_x_fourier
from banco_sparse_penalty import banco_sparse_update


def banco_coefficient_step(
    X, Y, U, S, operator,
    lambda_s, rho, iteration,
    x_relax=1.8,
):
    """One Banco/SPORCO coefficient ADMM step."""

    # MATLAB lambda2=0 X update: solvedbi_sm.
    X = solve_banco_x_fourier(
        operator, S, Y, U, rho
    )

    # XRelaxParam.
    Xr = x_relax*X + (1.0-x_relax)*Y

    # Iterations 1:20 -> L1.
    # Iterations 21+  -> fixed-a log shrink.
    Ynew = banco_sparse_update(
        Xr+U,
        lambda_s=lambda_s,
        rho=rho,
        iteration=iteration,
        l1_iters=20,
        nonnegative=True,
    )

    Unew = U + Xr-Ynew

    return X, Ynew, Unew, None