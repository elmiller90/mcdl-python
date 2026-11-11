import numpy as np

from multiscale_operator import resample_atom
from convolution_operator import ConvolutionOperator


class MultiscaleConvolutionOperator:
    """
    Fixed-dictionary multiscale 1-D convolution operator.

    Parameters
    ----------
    D : ndarray, shape (M, K)
        Base dictionary. One atom per column.

    scales : sequence of (c1, c2)
        Rational scales c1/c2.

    center : int
        Dictionary center using the MATLAB 1-based convention.

    N : int
        Signal/coefficient-map length.

    normalize : bool
        If True, normalize every resampled atom to unit L2 norm.

    Notes
    -----
    Internally the scaled filters are flattened in atom-major order:

        (k=0,j=0), (k=0,j=1), ...,
        (k=1,j=0), ...

    Thus F = K*J.
    """

    def __init__(
        self,
        D,
        scales,
        center,
        N,
        normalize=True,
    ):
        D = np.asarray(D, dtype=np.float64)

        if D.ndim == 1:
            D = D[:, None]

        if D.ndim != 2:
            raise ValueError(
                "D must have shape (M, K)"
            )

        if N <= 0:
            raise ValueError("N must be positive")

        self.D_base = D.copy()
        self.M, self.K = D.shape

        self.scales = list(scales)
        self.J = len(self.scales)

        if self.J == 0:
            raise ValueError(
                "At least one scale is required"
            )

        self.F = self.K * self.J
        self.center = int(center)
        self.N = int(N)
        self.normalize = bool(normalize)

        self.D_scaled = np.zeros(
            (self.N, self.F),
            dtype=np.float64,
        )

        self.norm_vals = np.ones(
            (self.K, self.J),
            dtype=np.float64,
        )

        self.shifts = np.zeros(
            (self.K, self.J),
            dtype=int,
        )

        self._build_scaled_dictionary()

        self.conv = ConvolutionOperator(
            self.D_scaled
        )

    def channel(self, k, j):
        """
        Flatten atom/scale pair to filter-channel index.
        """
        if not (0 <= k < self.K):
            raise IndexError("Invalid atom index k")

        if not (0 <= j < self.J):
            raise IndexError("Invalid scale index j")

        return k * self.J + j

    def _build_scaled_dictionary(self):
        for k in range(self.K):
            d = self.D_base[:, k]

            for j, (c1, c2) in enumerate(
                self.scales
            ):
                raw, shift = resample_atom(
                    d=d,
                    c1=c1,
                    c2=c2,
                    center=self.center,
                    N=self.N,
                )

                norm_val = np.linalg.norm(raw)

                if norm_val == 0:
                    raise ValueError(
                        "Resampling produced a zero-norm "
                        f"atom for k={k}, j={j}, "
                        f"scale={c1}/{c2}"
                    )

                f = self.channel(k, j)

                if self.normalize:
                    self.D_scaled[:, f] = (
                        raw / norm_val
                    )
                    self.norm_vals[k, j] = norm_val
                else:
                    self.D_scaled[:, f] = raw
                    self.norm_vals[k, j] = 1.0

                self.shifts[k, j] = shift

    def forward(self, X):
        """
        Compute S = A_D X.

        X may have shape:

            (N, K, J, T)

        or flattened channel form:

            (N, K*J, T)

        Returns
        -------
        S : ndarray, shape (N, T)
        """
        Xf = self._flatten_x(X)

        return self.conv.forward(Xf)

    def adjoint(self, R, structured=True):
        """
        Compute A_D^* R with D fixed.

        Parameters
        ----------
        R : ndarray, shape (N, T)

        structured : bool
            If True return shape (N,K,J,T).
            Otherwise return (N,K*J,T).
        """
        Xf = self.conv.adjoint(R)

        if not structured:
            return Xf

        return self._unflatten_x(Xf)

    def _flatten_x(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 3:
            if X.shape[0] != self.N:
                raise ValueError(
                    "X first dimension must equal N"
                )

            if X.shape[1] != self.F:
                raise ValueError(
                    f"Expected {self.F} channels"
                )

            return X

        if X.ndim != 4:
            raise ValueError(
                "X must have shape (N,K,J,T) "
                "or (N,K*J,T)"
            )

        N, K, J, T = X.shape

        if N != self.N:
            raise ValueError(
                "X first dimension must equal N"
            )

        if K != self.K or J != self.J:
            raise ValueError(
                f"Expected X shape "
                f"(N,{self.K},{self.J},T)"
            )

        # C-order flattening gives:
        # (k=0,j=0), (k=0,j=1), ...
        return X.reshape(
            self.N,
            self.F,
            T,
        )

    def _unflatten_x(self, X):
        N, F, T = X.shape

        if N != self.N or F != self.F:
            raise ValueError(
                "Invalid flattened X dimensions"
            )

        return X.reshape(
            self.N,
            self.K,
            self.J,
            T,
        )