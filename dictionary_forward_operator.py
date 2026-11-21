import numpy as np

from multiscale_operator import resample_atom, resample_atom_adjoint
from zero_padded_convolution import ZeroPaddedConvolutionOperator


class DictionaryForwardOperator:
    """Fixed-X, fixed-alpha dictionary forward operator."""

    def __init__(self, X, alpha, scales, center, N, M):
        self.X = np.asarray(X, dtype=np.float64)
        self.alpha = np.asarray(alpha, dtype=np.float64)
        self.scales = list(scales)
        self.center = int(center)
        self.N = int(N)
        self.M = int(M)

        if self.X.ndim != 4:
            raise ValueError("X must have shape (L,K,J,T)")

        self.L, self.K, self.J, self.T = self.X.shape

        if self.L != self.N + self.M - 1:
            raise ValueError("X has inconsistent first dimension")
        if self.J != len(self.scales):
            raise ValueError("X scale dimension does not match scales")
        if self.alpha.shape != (self.K, self.J):
            raise ValueError(
                f"alpha must have shape {(self.K, self.J)}"
            )

    def scaled_dictionary(self, D):
        """Return physical scaled dictionary of shape (N,K*J)."""
        D = np.asarray(D, dtype=np.float64)

        if D.shape != (self.M, self.K):
            raise ValueError(
                f"D must have shape {(self.M, self.K)}"
            )

        A = np.empty(
            (self.N, self.K * self.J),
            dtype=np.float64,
        )

        for k in range(self.K):
            for j, (c1, c2) in enumerate(self.scales):
                f = k * self.J + j

                atom, _ = resample_atom(
                    D[:, k],
                    c1,
                    c2,
                    self.center,
                    self.N,
                )

                A[:, f] = self.alpha[k, j] * atom

        return A

    def forward(self, D):
        """Apply the fixed-X, fixed-alpha map D -> S."""
        A = self.scaled_dictionary(D)

        conv = ZeroPaddedConvolutionOperator(
            D=A,
            M=self.M,
        )

        Xflat = self.X.reshape(
            self.L,
            self.K * self.J,
            self.T,
        )

        return conv.forward(Xflat)
    
    def adjoint(self, R):
        """Apply the signal-to-base-dictionary adjoint B*R."""
        R = np.asarray(R, dtype=np.float64)

        if R.shape != (self.N, self.T):
            raise ValueError(
                f"R must have shape {(self.N, self.T)}"
            )

        Rpad = np.zeros((self.L, self.T), dtype=np.float64)
        Rpad[self.M - 1:, :] = R
        Rf = np.fft.fft(Rpad, axis=0)

        Dadj = np.zeros((self.M, self.K), dtype=np.float64)

        for k in range(self.K):
            for j, (c1, c2) in enumerate(self.scales):
                Xkj = self.X[:, k, j, :]
                Xf = np.fft.fft(Xkj, axis=0)

                # Adjoint of convolution by fixed X[:,k,j,t].
                Zf = np.conj(Xf) * Rf
                Z = np.fft.ifft(Zf, axis=0).real

                # Sum over signals T before applying M_j^*.
                z = np.sum(Z, axis=1)[:self.N]

                Dadj[:, k] += self.alpha[k, j] * resample_atom_adjoint(
                    z=z,
                    M=self.M,
                    c1=c1,
                    c2=c2,
                    center=self.center,
                    N=self.N,
                )

        return Dadj
    
    def normal(self, D, rho=0.0):
        """Apply (B*B + rho I)D."""
        D = np.asarray(D, dtype=np.float64)

        if D.shape != (self.M, self.K):
            raise ValueError(
                f"D must have shape {(self.M, self.K)}"
            )

        if rho < 0:
            raise ValueError("rho must be nonnegative")

        return self.adjoint(self.forward(D)) + rho * D

    