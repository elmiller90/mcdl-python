import numpy as np


class ConvolutionOperator:
    """
    1-D multi-filter circular convolution operator.

    D shape:
        (N, F)

    X shape:
        (N, F, T)

    forward(X) shape:
        (N, T)

    adjoint(R) shape:
        (N, F, T)

    where F is the total number of atom/scale channels.
    """

    def __init__(self, D):
        D = np.asarray(D, dtype=np.float64)

        if D.ndim != 2:
            raise ValueError(
                "D must have shape (N, F)"
            )

        self.D = D
        self.N, self.F = D.shape

        # MATLAB-style FFT convention:
        # forward FFT is unnormalized, inverse has 1/N.
        self.Df = np.fft.fft(
            D,
            axis=0,
        )

    def forward(self, X):
        """
        Compute

            S_t = sum_f D_f * X_{f,t}

        using circular convolution.
        """
        X = np.asarray(X, dtype=np.float64)

        expected = (
            self.N,
            self.F,
            X.shape[2]
            if X.ndim == 3 else None,
        )

        if X.ndim != 3:
            raise ValueError(
                "X must have shape (N, F, T)"
            )

        if X.shape[0] != self.N:
            raise ValueError(
                "X first dimension must equal N"
            )

        if X.shape[1] != self.F:
            raise ValueError(
                "X second dimension must equal F"
            )

        Xf = np.fft.fft(
            X,
            axis=0,
        )

        Sf = np.sum(
            self.Df[:, :, None] * Xf,
            axis=1,
        )

        S = np.fft.ifft(
            Sf,
            axis=0,
        )

        return np.real_if_close(S).real

    def adjoint(self, R):
        """
        Adjoint with respect to X.

        For residual R,

            A*R

        is correlation with each dictionary filter.
        """
        R = np.asarray(R, dtype=np.float64)

        if R.ndim != 2:
            raise ValueError(
                "R must have shape (N, T)"
            )

        if R.shape[0] != self.N:
            raise ValueError(
                "R first dimension must equal N"
            )

        Rf = np.fft.fft(
            R,
            axis=0,
        )

        Xf = (
            np.conj(self.Df)[:, :, None]
            * Rf[:, None, :]
        )

        X = np.fft.ifft(
            Xf,
            axis=0,
        )

        return np.real_if_close(X).real