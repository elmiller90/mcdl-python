import numpy as np

from banco_coefficient_step import banco_coefficient_step
from banco_sparse_penalty import banco_sparse_update
from multiscale_convolution import MultiscaleConvolutionOperator
from banco_coefficient_fourier import solve_banco_x_fourier


def test_banco_step_matches_manual_update():
    M, N = 21, 48

    x = np.arange(M, dtype=float)
    d = np.exp(-0.5*((x-(M-1)/2)/3.0)**2)
    d /= np.linalg.norm(d)

    A = MultiscaleConvolutionOperator(
        D=d[:, None],
        scales=[(1, 1)],
        center=(M+1)//2,
        N=N,
        normalize=True,
    )

    X = np.zeros((A.L, 1, 1, 1))
    Y = np.zeros_like(X)
    U = np.zeros_like(X)

    Xtrue = np.zeros_like(X)
    Xtrue[M-1+18, 0, 0, 0] = 1.0
    S = A.forward(Xtrue)

    Xm = solve_banco_x_fourier(
        operator=A,
        S=S,
        Y=Y,
        U=U,
        rho=0.1,
    )


    Xr = 1.8*Xm - 0.8*Y
    Ym = banco_sparse_update(
        Xr+U, lambda_s=0.1, rho=0.1,
        iteration=21,
    )
    Um = U + Xr - Ym

    X1, Y1, U1, _ = banco_coefficient_step(
        X, Y, U, S, A,
        lambda_s=0.1, rho=0.1,
        iteration=21,
    )

    assert np.allclose(X1, Xm)
    assert np.allclose(Y1, Ym)
    assert np.allclose(U1, Um)