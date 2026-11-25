import numpy as np

from banco_coefficient_fourier import solve_banco_x_fourier
from multiscale_convolution import MultiscaleConvolutionOperator


def test_banco_fourier_x_update():
    rng = np.random.default_rng(90)

    M, N, T = 9, 20, 3
    scales = [(1, 2), (1, 1)]

    D = rng.random((M, 2))
    D /= np.linalg.norm(D, axis=0)

    A = MultiscaleConvolutionOperator(
        D=D, scales=scales,
        center=(M+1)//2, N=N, normalize=True,
    )

    shape = (A.L, A.K, A.J, T)
    Y = rng.standard_normal(shape)
    U = rng.standard_normal(shape)
    S = rng.standard_normal((N, T))
    rho = 2.7

    X = solve_banco_x_fourier(A, S, Y, U, rho)

    Xf = np.fft.fft(
        X.reshape(A.L, A.F, T), axis=0
    )
    Yf = np.fft.fft(
        Y.reshape(A.L, A.F, T), axis=0
    )
    Uf = np.fft.fft(
        U.reshape(A.L, A.F, T), axis=0
    )

    Sf = np.fft.fft(
        A.conv.pad_residual(S), axis=0
    )
    Af = A.conv.Df

    bf = (
        np.conj(Af)[:, :, None] * Sf[:, None, :]
        + rho*(Yf-Uf)
    )

    AXf = np.sum(
        Af[:, :, None]*Xf,
        axis=1,
        keepdims=True,
    )

    lhs = (
        np.conj(Af)[:, :, None]*AXf
        + rho*Xf
    )

    rel = np.linalg.norm(lhs-bf)/np.linalg.norm(bf)

    print(f"Fourier normal-equation residual: {rel:.3e}")

    assert rel < 1e-11