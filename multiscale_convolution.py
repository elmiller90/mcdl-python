import numpy as np

from multiscale_operator import resample_atom
from zero_padded_convolution import ZeroPaddedConvolutionOperator


class MultiscaleConvolutionOperator:
    """
    MATLAB-compatible fixed-dictionary multiscale convolution operator.

    Base dictionary:
        D.shape == (M, K)

    Scaled dictionary:
        D_scaled.shape == (N, K*J)

    Extended coefficient domain:
        L = N + M - 1

    Structured coefficients:
        X.shape == (L, K, J, T)

    Physical signals:
        S.shape == (N, T)

    Channel order is atom-major:

        (k=0,j=0), (k=0,j=1), ...,
        (k=1,j=0), ...
    """

    def __init__(
        self,
        D,
        scales,
        center,
        N,
        normalize=True,
    ):
        D = np.asarray(
            D,
            dtype=np.float64,
        )

        if D.ndim == 1:
            D = D[:, None]

        if D.ndim != 2:
            raise ValueError(
                "D must have shape (M, K)"
            )

        if N <= 0:
            raise ValueError(
                "N must be positive"
            )

        self.D_base = D.copy()

        self.M, self.K = D.shape
        self.N = int(N)

        if self.M > self.N:
            raise ValueError(
                "Dictionary support M cannot exceed N"
            )

        self.L = self.N + self.M - 1

        self.scales = list(scales)
        self.J = len(self.scales)

        if self.J == 0:
            raise ValueError(
                "At least one scale is required"
            )

        self.F = self.K * self.J

        self.center = int(center)
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

        # MATLAB-compatible padded convolution engine.
        self.conv = ZeroPaddedConvolutionOperator(
            D=self.D_scaled,
            M=self.M,
        )

        # Sanity check: both classes must agree on L.
        if self.conv.L != self.L:
            raise RuntimeError(
                "Inconsistent extended convolution length"
            )

    def channel(self, k, j):
        """
        Flatten atom/scale pair to channel index.
        """
        if not 0 <= k < self.K:
            raise IndexError(
                "Invalid atom index k"
            )

        if not 0 <= j < self.J:
            raise IndexError(
                "Invalid scale index j"
            )

        return k * self.J + j

    def _build_scaled_dictionary(self):
        """
        Construct each scaled atom at physical length N.

        Each resampled atom is optionally normalized to
        unit L2 norm, matching the MATLAB scaled-dictionary
        normalization convention.
        """
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

                norm_val = np.linalg.norm(
                    raw
                )

                if norm_val == 0:
                    raise ValueError(
                        "Resampling produced a "
                        "zero-norm atom for "
                        f"k={k}, j={j}, "
                        f"scale={c1}/{c2}"
                    )

                f = self.channel(k, j)

                if self.normalize:
                    self.D_scaled[:, f] = (
                        raw / norm_val
                    )

                    self.norm_vals[
                        k, j
                    ] = norm_val

                else:
                    self.D_scaled[:, f] = raw

                    self.norm_vals[
                        k, j
                    ] = 1.0

                self.shifts[
                    k, j
                ] = shift

    def forward(self, X):
        """
        Compute the physical signal

            S = A_D X.

        X may be supplied as:

            structured:
                (L, K, J, T)

        or:

            flattened:
                (L, K*J, T)

        Returns
        -------
        S : ndarray
            Shape (N, T).
        """
        X_flat = self._flatten_x(X)

        return self.conv.forward(
            X_flat
        )

    def forward_padded(self, X):
        """
        Return the complete length-L convolution before
        the MATLAB pre-crop.

        Returns
        -------
        Z : ndarray
            Shape (L, T).
        """
        X_flat = self._flatten_x(X)

        return self.conv.forward_padded(
            X_flat
        )

    def adjoint(
        self,
        R,
        structured=True,
    ):
        """
        Compute the coefficient-space adjoint

            A_D^* R.

        Parameters
        ----------
        R : ndarray
            Physical signal residual of shape (N, T).

        structured : bool
            If True return

                (L, K, J, T)

            otherwise return

                (L, K*J, T).
        """
        X_flat = self.conv.adjoint(R)

        if not structured:
            return X_flat

        return self._unflatten_x(
            X_flat
        )

    def pad_residual(self, R):
        """
        Expose the MATLAB-compatible residual pre-padding
        for diagnostics.

        Input:
            (N, T)

        Output:
            (L, T)
        """
        return self.conv.pad_residual(R)

    def _flatten_x(self, X):
        """
        Convert structured coefficients to flattened
        channel form.
        """
        X = np.asarray(
            X,
            dtype=np.float64,
        )

        if X.ndim == 3:
            if X.shape[0] != self.L:
                raise ValueError(
                    "Flattened X first dimension "
                    f"must equal L={self.L}"
                )

            if X.shape[1] != self.F:
                raise ValueError(
                    "Flattened X second dimension "
                    f"must equal F={self.F}"
                )

            return X

        if X.ndim != 4:
            raise ValueError(
                "X must have shape "
                "(L,K,J,T) or (L,K*J,T)"
            )

        L, K, J, T = X.shape

        if L != self.L:
            raise ValueError(
                f"Expected coefficient length "
                f"L={self.L}, got {L}"
            )

        if K != self.K:
            raise ValueError(
                f"Expected K={self.K}, got {K}"
            )

        if J != self.J:
            raise ValueError(
                f"Expected J={self.J}, got {J}"
            )

        return X.reshape(
            self.L,
            self.F,
            T,
        )

    def _unflatten_x(self, X):
        """
        Convert flattened coefficient channels back to

            (L, K, J, T).
        """
        X = np.asarray(
            X,
            dtype=np.float64,
        )

        if X.ndim != 3:
            raise ValueError(
                "Flattened X must have "
                "shape (L,F,T)"
            )

        L, F, T = X.shape

        if L != self.L:
            raise ValueError(
                f"Expected L={self.L}, got {L}"
            )

        if F != self.F:
            raise ValueError(
                f"Expected F={self.F}, got {F}"
            )

        return X.reshape(
            self.L,
            self.K,
            self.J,
            T,
        )