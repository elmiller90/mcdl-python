import numpy as np

def _upsample_indices(n, m, center):
    left = np.arange(center - m, 0, -m)[::-1]
    right = np.arange(center, m * n + 1, m)

    inds = np.concatenate([left, right]).astype(int) - 1

    if inds.size != n:
        raise ValueError(
            f"Upsampling index mismatch: "
            f"{inds.size} indices for {n} samples"
        )

    return inds


def _downsample_indices(n, m, center):
    left = np.arange(center - m, 0, -m)[::-1]
    right = np.arange(center, n + 1, m)

    inds = np.concatenate([left, right]).astype(int)

    matches = np.where(inds == center)[0]

    if matches.size != 1:
        raise ValueError(
            "Could not locate center after downsampling"
        )

    new_center = int(matches[0] + 1)

    return inds - 1, new_center


def lowpass_m(x, m):
    """
    Python equivalent of MATLAB lowpassM(x,m,0).

    Uses the normalized triangular interpolation kernel
    and 'same' convolution.
    """
    x = np.asarray(x, dtype=np.float64)

    if m == 1:
        return x.copy()

    left = np.arange(1, m, dtype=np.float64) / m

    f = np.concatenate(
        [left, [1.0], left[::-1]]
    )

    f /= np.linalg.norm(f)

    return np.convolve(x, f, mode="same")


def upsample_center(x, m, center):
    """
    Python equivalent of MATLAB uSampleCenter for 1-D data.

    center uses MATLAB's 1-based convention.
    """
    x = np.asarray(x, dtype=np.float64)

    if m == 1:
        return x.copy()

    n = x.size
    y = np.zeros(n * m, dtype=x.dtype)

    # MATLAB:
    # inds = [fliplr(center-m:-m:1), center:m:m*N]
    left = np.arange(center - m, 0, -m)[::-1]
    right = np.arange(center, m * n + 1, m)

    inds = np.concatenate([left, right])

    # MATLAB -> Python indices
    inds0 = inds.astype(int) - 1

    if inds0.size != n:
        raise ValueError(
            f"Upsampling index mismatch: "
            f"{inds0.size} indices for {n} samples"
        )

    y[inds0] = x

    return lowpass_m(y, m)


def downsample_center(x, m, center):
    """
    Python equivalent of MATLAB dSampleCenter.

    Returns
    -------
    y
        Downsampled signal.

    new_center
        Center index using MATLAB's 1-based convention.
    """
    x = np.asarray(x, dtype=np.float64)

    if m == 1:
        return x.copy(), int(center)

    n = x.size

    # MATLAB:
    # inds = [fliplr(center-m:-m:1), center:m:N]
    left = np.arange(center - m, 0, -m)[::-1]
    right = np.arange(center, n + 1, m)

    inds = np.concatenate([left, right]).astype(int)

    matches = np.where(inds == center)[0]

    if matches.size != 1:
        raise ValueError("Could not locate center after downsampling")

    # MATLAB find() is 1-based.
    new_center = int(matches[0] + 1)

    filtered = lowpass_m(x, m)

    return filtered[inds - 1], new_center

def resample_atom(d, c1, c2, center, N, norm_val=None):
    """
    Resample one base atom by rational scale c1/c2.

    Returns the unnormalized, zero-padded atom plus
    the MATLAB-style center shift.
    """
    d = np.asarray(d, dtype=np.float64)

    if c1 == c2:
        out = np.zeros(N, dtype=d.dtype)
        out[:d.size] = d
        return out, 0

    du = upsample_center(
        d,
        c1,
        c1 * center,
    )

    dd, center_down = downsample_center(
        du,
        c2,
        c1 * center,
    )

    shift = int(center - center_down)

    if dd.size > N:
        raise ValueError(
            f"Resampled atom length {dd.size} exceeds N={N}"
        )

    out = np.zeros(N, dtype=d.dtype)
    out[:dd.size] = dd

    # MATLAB circshift
    out = np.roll(out, shift)
    if norm_val is not None:
        if norm_val <= 0:
            raise ValueError(
                "norm_val must be positive"
            )
        out = out / norm_val

    return out, shift

def resample_atom_adjoint(
        z,
        M,
        c1,
        c2,
        center,
        N,
        norm_val=1.0):
    """
    Adjoint of resample_atom() with a fixed normalization.

    Parameters
    ----------
    z :
        Vector in output space, length N.

    M :
        Length of original base atom.

    c1, c2 :
        Rational scale c1/c2.

    center :
        MATLAB-style 1-based center.

    N :
        Output length used by resample_atom().

    norm_val :
        Fixed normalization used in forward operator.

    Returns
    -------
    d_adj :
        Length-M vector in base-atom space.
    """
    z = np.asarray(z, dtype=np.float64)

    if z.size != N:
        raise ValueError(
            f"Expected z length {N}, got {z.size}"
        )

    if norm_val <= 0:
        raise ValueError(
            "norm_val must be positive"
        )

    # Adjoint of normalization by norm_val.
    z = z / norm_val

    if c1 == c2:
        return z[:M].copy()

    # Determine the intermediate dimensions and shift
    # exactly as in the forward operation.
    n_up = M * c1
    center_up = c1 * center

    down_inds, center_down = _downsample_indices(
        n_up,
        c2,
        center_up,
    )

    n_down = down_inds.size

    shift = int(center - center_down)

    # Forward used:
    #
    #     padded = np.roll(padded, shift)
    #
    # Therefore the adjoint uses the inverse shift.
    z_unshift = np.roll(z, -shift)

    # Adjoint of zero padding:
    # crop back to the pre-padding size.
    z_down = z_unshift[:n_down]

    # Reverse downsampling.
    z_up = downsample_center_adjoint(
        z_down,
        n_up,
        c2,
        center_up,
    )

    # Reverse upsampling.
    d_adj = upsample_center_adjoint(
        z_up,
        M,
        c1,
        center_up,
    )

    return d_adj


def upsample_center_adjoint(y, n, m, center):
    """
    Adjoint of upsample_center().

    Parameters
    ----------
    y :
        Vector in the upsampled space.

    n :
        Length of the original input vector.
    """
    y = np.asarray(y, dtype=np.float64)

    if m == 1:
        return y[:n].copy()

    inds = _upsample_indices(
        n,
        m,
        center,
    )

    # Adjoint of the symmetric lowpass convolution.
    z = lowpass_m(y, m)

    # Adjoint of inserting samples at inds:
    # select those samples.
    return z[inds]


def downsample_center_adjoint(
        y, n_input, m, center):
    """
    Adjoint of downsample_center().

    n_input is the length of the vector before
    downsampling.
    """
    y = np.asarray(y, dtype=np.float64)

    if m == 1:
        z = np.zeros(n_input, dtype=y.dtype)
        z[:y.size] = y
        return z

    inds, _ = _downsample_indices(
        n_input,
        m,
        center,
    )

    if y.size != inds.size:
        raise ValueError(
            f"Expected {inds.size} downsampled samples, "
            f"got {y.size}"
        )

    # Adjoint of selecting indices is scatter/insertion.
    z = np.zeros(n_input, dtype=y.dtype)
    z[inds] = y

    # Adjoint of symmetric convolution.
    z = lowpass_m(z, m)

    return z

