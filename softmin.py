import numpy as np

SHIFTS = (
    (0, 0),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


def shift2d(x, shift_j, shift_n, mode="zero"):
    """
    Shift a J x N array in scale (j) and position (n).

    Matches the convention used by MATLAB compute_softmin.m.
    """
    x = np.asarray(x)
    J, N = x.shape

    j = np.arange(J) - shift_j
    n = np.arange(N) - shift_n

    if mode == "reflect":
        j = _reflect_indices(j, J)
        n = _reflect_indices(n, N)
        return x[np.ix_(j, n)]

    if mode != "zero":
        raise ValueError("mode must be 'zero' or 'reflect'")

    y = np.zeros_like(x)

    valid_j = (j >= 0) & (j < J)
    valid_n = (n >= 0) & (n < N)

    dst_j = np.nonzero(valid_j)[0]
    dst_n = np.nonzero(valid_n)[0]

    src_j = j[valid_j]
    src_n = n[valid_n]

    y[np.ix_(dst_j, dst_n)] = x[np.ix_(src_j, src_n)]

    return y


def _reflect_indices(idx, length):
    """
    Reproduce MATLAB reflect_index() behavior.
    """
    idx = np.asarray(idx).copy()

    low = idx < 0
    idx[low] = -idx[low]

    high = idx >= length
    idx[high] = 2 * length - 2 - idx[high]

    return np.clip(idx, 0, length - 1)


def _to_kjnt(x, K, J):
    """
    Convert MATLAB-style coefficient layout

        (1, N, K*J, T)

    to

        (K, J, N, T).
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 4:
        raise ValueError("X must have shape (1, N, K*J, T)")

    if x.shape[0] != 1:
        raise ValueError("First dimension of X must be 1")

    _, N, KJ, T = x.shape

    if KJ != K * J:
        raise ValueError(
            f"Expected third dimension {K * J}, got {KJ}"
        )

    y = np.empty((K, J, N, T), dtype=x.dtype)

    for k in range(K):
        block = x[0, :, k * J:(k + 1) * J, :]
        y[k] = np.transpose(block, (1, 0, 2))

    return y


def _from_kjnt(x):
    """
    Convert

        (K, J, N, T)

    back to

        (1, N, K*J, T).
    """
    K, J, N, T = x.shape

    y = np.zeros((1, N, K * J, T), dtype=x.dtype)

    for k in range(K):
        block = np.transpose(x[k], (1, 0, 2))
        y[0, :, k * J:(k + 1) * J, :] = block

    return y


def softmin(X, K, J, tau=1e-3,
            pad_mode="zero",
            normalize=False,
            return_gradient=True,
            debug=False):
    """
    Soft-min temporal scale/position regularizer.

    Parameters
    ----------
    X : ndarray
        Shape (1, N, K*J, T).

    K : int
        Number of dictionary atoms.

    J : int
        Number of scale channels per atom.

    tau : float
        Soft-min temperature.

    pad_mode : {"zero", "reflect"}
        Boundary condition.

    normalize : bool
        Divide each squared residual by ||X_{t+1}||^2 + eps.

    return_gradient : bool
        If True, also return analytical gradient.

    Returns
    -------
    R : float
        Soft-min penalty.

    grad : ndarray, optional
        Same shape as X.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")

    X4 = _to_kjnt(X, K, J)

    K_, J_, N, T = X4.shape

    grad = np.zeros_like(X4) if return_gradient else None

    R = 0.0
    eps_norm = 1e-12

    for k in range(K_):
        for t in range(T - 1):

            Xt = X4[k, :, :, t]
            Xtp = X4[k, :, :, t + 1]

            residuals = []
            shifted = []

            denom = (
                np.sum(Xtp**2) + eps_norm
                if normalize else 1.0
            )

            for sj, sn in SHIFTS:

                Xs = shift2d(
                    Xt, sj, sn, mode=pad_mode
                )

                diff = Xtp - Xs

                r = np.sum(diff**2)

                if normalize:
                    r /= denom

                residuals.append(r)
                shifted.append(Xs)

            residuals = np.asarray(residuals)

            scaled = -residuals / tau

            # Stable log-sum-exp
            m = np.max(scaled)

            exp_scaled = np.exp(scaled - m)

            weights = exp_scaled / np.sum(exp_scaled)
            if debug:
                print(f"\nk={k}, t={t}")
                print(" shift       residual          weight")

                for (sj, sn), r, w in zip(SHIFTS, residuals, weights):
                    print(
                        f"({sj:+d},{sn:+d})"
                        f"    {r:12.6f}"
                        f"    {w:12.6f}"
                    )

            logsumexp = m + np.log(np.sum(exp_scaled))

            R += -tau * logsumexp

            if not return_gradient:
                continue

            grad_Xt = np.zeros_like(Xt)
            grad_Xtp = np.zeros_like(Xtp)

            for weight, (sj, sn), Xs in zip(
                    weights, SHIFTS, shifted):

                diff = Xtp - Xs

                if normalize:

                    ssum = np.sum(diff**2)

                    grad_next = (
                        2.0 * diff * denom
                        - 2.0 * ssum * Xtp
                    ) / denom**2

                    grad_shifted = (
                        -2.0 * diff / denom
                    )

                else:
                    grad_next = 2.0 * diff
                    grad_shifted = -2.0 * diff

                grad_Xtp += weight * grad_next

                grad_Xt += weight * shift2d(
                    grad_shifted,
                    -sj,
                    -sn,
                    mode=pad_mode
                )

            grad[k, :, :, t] += grad_Xt
            grad[k, :, :, t + 1] += grad_Xtp

    if return_gradient:
        return R, _from_kjnt(grad)

    return R