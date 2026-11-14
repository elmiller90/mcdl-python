import numpy as np


class ZeroPaddedConvolutionOperator:
    """
    MATLAB-compatible zero-padded 1-D convolution operator.

    The MATLAB implementation uses

        L = N + M - 1

    where

        N : physical signal length
        M : base dictionary support length
        L : extended coefficient / FFT length

    Dictionary filters are supplied at physical length N and
    post-padded with M-1 zeros to length L.

    Coefficient maps live directly in the extended domain:

        X.shape == (L, F, T)

    The forward operator computes length-L circular convolution,
    sums across filter channels, and discards the first M-1
    samples:

        S = C A X

    where

        S.shape == (N, T).

    The adjoint reverses the crop by pre-padding a physical
    residual with M-1 zeros before Fourier correlation:

        R.shape          == (N, T)
        adjoint(R).shape == (L, F, T)
    """

    def __init__(self, D, M):
        """
        Parameters
        ----------
        D : ndarray, shape (N, F)
            Dictionary filters in the physical N-sample domain.

        M : int
            Base dictionary support length. The extended FFT
            length is L = N + M - 1.
        """
        D = np.asarray(D, dtype=np.float64)

        if D.ndim != 2:
            raise ValueError(
                "D must have shape (N, F)"
            )

        if M <= 0:
            raise ValueError(
                "M must be positive"
            )

        self.D = D
        self.N, self.F = D.shape
        self.M = int(M)

        if self.M > self.N:
            raise ValueError(
                "M cannot exceed N"
            )

        self.pad = self.M - 1
        self.L = self.N + self.pad

        # MATLAB:
        #
        #   Dpad = padarray(D,[0 M-1 ...],0,'post')
        #
        # Python:
        #
        #   |------- D: N -------|-- zeros: M-1 --|
        self.D_padded = np.zeros(
            (self.L, self.F),
            dtype=np.float64,
        )

        self.D_padded[:self.N, :] = D

        # MATLAB fft2 acts along the spatial dimension here.
        self.Df = np.fft.fft(
            self.D_padded,
            axis=0,
        )

    def forward_padded(self, X):
        """
        Compute the complete length-L convolution before
        the MATLAB pre-crop.

        Parameters
        ----------
        X : ndarray, shape (L, F, T)

        Returns
        -------
        Z : ndarray, shape (L, T)
            Full padded convolution result.
        """
        X = np.asarray(X, dtype=np.float64)

        self._validate_x(X)

        Xf = np.fft.fft(
            X,
            axis=0,
        )

        Zf = np.sum(
            self.Df[:, :, None] * Xf,
            axis=1,
        )

        Z = np.fft.ifft(
            Zf,
            axis=0,
        )

        return np.real_if_close(
            Z
        ).real

    def forward(self, X):
        """
        MATLAB-compatible physical forward operator.

        MATLAB equivalent:

            Z = ifft2(
                sum(bsxfun(@times,Df,Xf),3),
                'symmetric'
            );

            S = unpad(Z,M-1,'pre');

        Since unpad(...,M-1,'pre') removes the first
        M-1 entries, Python returns

            Z[M-1:]

        which has length N.
        """
        Z = self.forward_padded(X)

        return Z[self.pad:, :]

    def pad_residual(self, R):
        """
        Apply the adjoint of the forward crop.

        MATLAB equivalent:

            Rpad = padarray(
                R,
                [0 M-1 ...],
                0,
                'pre'
            );

        Parameters
        ----------
        R : ndarray, shape (N, T)

        Returns
        -------
        Rpad : ndarray, shape (L, T)
        """
        R = np.asarray(R, dtype=np.float64)

        if R.ndim != 2:
            raise ValueError(
                "R must have shape (N, T)"
            )

        if R.shape[0] != self.N:
            raise ValueError(
                f"Expected residual length "
                f"N={self.N}, got {R.shape[0]}"
            )

        T = R.shape[1]

        Rpad = np.zeros(
            (self.L, T),
            dtype=np.float64,
        )

        Rpad[self.pad:, :] = R

        return Rpad

    def adjoint(self, R):
        """
        Coefficient-space adjoint.

        Parameters
        ----------
        R : ndarray, shape (N, T)
            Residual in the physical signal domain.

        Returns
        -------
        Xadj : ndarray, shape (L, F, T)
            Adjoint result in the extended coefficient domain.
        """
        Rpad = self.pad_residual(R)

        Rf = np.fft.fft(
            Rpad,
            axis=0,
        )

        Xf = (
            np.conj(self.Df)[:, :, None]
            * Rf[:, None, :]
        )

        Xadj = np.fft.ifft(
            Xf,
            axis=0,
        )

        return np.real_if_close(
            Xadj
        ).real

    def _validate_x(self, X):
        if X.ndim != 3:
            raise ValueError(
                "X must have shape (L, F, T)"
            )

        if X.shape[0] != self.L:
            raise ValueError(
                f"Expected coefficient length "
                f"L={self.L}, got {X.shape[0]}"
            )

        if X.shape[1] != self.F:
            raise ValueError(
                f"Expected F={self.F} channels, "
                f"got {X.shape[1]}"
            )
