import numpy as np

from banco_solvedbi_sm import solvedbi_sm_rank1


def solve_banco_x_fourier(operator, S, Y, U, rho):
    """Banco/SPORCO lambda2=0 coefficient X-update."""
    S = np.asarray(S, dtype=float)
    Y = np.asarray(Y, dtype=float)
    U = np.asarray(U, dtype=float)

    shape = (operator.L, operator.K, operator.J, S.shape[1])
    if Y.shape != shape or U.shape != shape:
        raise ValueError(f"Y,U must have shape {shape}")

    # (L,K,J,T) -> atom-major (L,F,T).
    Yf = np.fft.fft(
        Y.reshape(operator.L, operator.F, S.shape[1]),
        axis=0,
    )
    Uf = np.fft.fft(
        U.reshape(operator.L, operator.F, S.shape[1]),
        axis=0,
    )

    # MATLAB:
    # Spad = padarray(S,[0 M-1 ...],0,'pre')
    Spad = operator.conv.pad_residual(S)
    Sf = np.fft.fft(Spad, axis=0)

    # MATLAB AGf: post-padded scaled dictionary FFT.
    Af = operator.conv.Df

    # bf = AGSf + rho*fft2(Y-U)
    # AGSf = conj(AGf).*Sfpad
    bf = (
        np.conj(Af)[:, :, None] * Sf[:, None, :]
        + rho*(Yf-Uf)
    )

    # Solve independently at every Fourier bin and frame.
    Xf = solvedbi_sm_rank1(
        Af[:, :, None],
        rho,
        bf,
    )

    Xflat = np.fft.ifft(Xf, axis=0)
    Xflat = np.real_if_close(Xflat).real

    return Xflat.reshape(
        operator.L, operator.K, operator.J, S.shape[1]
    )