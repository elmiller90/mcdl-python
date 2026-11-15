import numpy as np


def squared_gate(m, mu):
    if mu <= 0:
        raise ValueError("gate_mass must be positive")
    return m * m / (m * m + mu * mu)


def squared_gate_derivative(m, mu):
    if mu <= 0:
        raise ValueError("gate_mass must be positive")
    d = m * m + mu * mu
    return 2.0 * m * mu * mu / (d * d)


def barycentric_transition_value_grad(
    X,
    shift_coordinates,
    scale_coordinates,
    gate_mass=0.05,
    mass_floor=1e-10,
    shift_velocity=1.0,
    scale_velocity=None,
):
    """
    Smooth gated barycentric-motion regularizer.

    X shape: (L,K,J,T), nonnegative.

    For each atom/frame:
        p = (x + mass_floor*u) / (sum(x) + mass_floor)

    Coordinates:
        nbar = sum p(n,j) * shift_coordinates[n]
        sbar = sum p(n,j) * log(scale_coordinates[j])

    Pair penalty:
        r = ((nbar[t+1]-nbar[t])/shift_velocity)^2
          + ((sbar[t+1]-sbar[t])/scale_velocity)^2

    Weighted by:
        g(m_t) g(m_{t+1})

    Returns:
        value, gradient, diagnostics
    """
    X = np.asarray(X, dtype=np.float64)
    ncoord = np.asarray(shift_coordinates, dtype=np.float64)
    scales = np.asarray(scale_coordinates, dtype=np.float64)

    if X.ndim != 4:
        raise ValueError("X must have shape (L,K,J,T)")
    if np.any(X < 0):
        raise ValueError("X must be nonnegative")
    if mass_floor <= 0:
        raise ValueError("mass_floor must be positive")
    if shift_velocity <= 0:
        raise ValueError("shift_velocity must be positive")

    L, K, J, T = X.shape

    if ncoord.shape != (L,):
        raise ValueError(f"shift_coordinates must have shape ({L},)")
    if scales.shape != (J,):
        raise ValueError(f"scale_coordinates must have shape ({J},)")
    if np.any(scales <= 0):
        raise ValueError("scale coordinates must be positive")

    log_scales = np.log(scales)

    if scale_velocity is None:
        spacing = np.abs(np.diff(log_scales))
        spacing = spacing[spacing > 0]
        scale_velocity = (
            float(np.median(spacing))
            if spacing.size else 1.0
        )

    if scale_velocity <= 0:
        raise ValueError("scale_velocity must be positive")

    # Coordinates use the same flattening convention as
    # state_index(n,j,J) = n*J+j.
    n_grid = np.repeat(ncoord, J)
    s_grid = np.tile(log_scales, L)
    P = L * J

    # Uniform low-mass background.
    background = np.full(P, 1.0 / P)

    masses = np.zeros((K, T))
    gates = np.zeros((K, T))
    center_n = np.zeros((K, T))
    center_s = np.zeros((K, T))

    # Cache normalized distributions and coordinate gradients.
    p_cache = np.zeros((K, T, P))
    dn_dx = np.zeros((K, T, P))
    ds_dx = np.zeros((K, T, P))

    for k in range(K):
        for t in range(T):
            x = X[:, k, :, t].reshape(-1)
            m = float(x.sum())
            denom = m + mass_floor
            p = (x + mass_floor * background) / denom

            bn = float(np.dot(p, n_grid))
            bs = float(np.dot(p, s_grid))

            masses[k, t] = m
            gates[k, t] = squared_gate(m, gate_mass)
            center_n[k, t] = bn
            center_s[k, t] = bs
            p_cache[k, t] = p

            # d <p,c> / dx = (c - <p,c>) / denom
            dn_dx[k, t] = (n_grid - bn) / denom
            ds_dx[k, t] = (s_grid - bs) / denom

    grad = np.zeros_like(X)
    pair_shape = (K, max(T - 1, 0))

    pair_values = np.zeros(pair_shape)
    pair_weights = np.zeros(pair_shape)
    weighted_values = np.zeros(pair_shape)
    shift_motion = np.zeros(pair_shape)
    scale_motion = np.zeros(pair_shape)

    value = 0.0
    vn2 = shift_velocity * shift_velocity
    vs2 = scale_velocity * scale_velocity

    for k in range(K):
        for t in range(T - 1):
            mn0, mn1 = center_n[k, t], center_n[k, t + 1]
            ms0, ms1 = center_s[k, t], center_s[k, t + 1]

            dn = mn1 - mn0
            ds = ms1 - ms0

            r_shift = dn * dn / vn2
            r_scale = ds * ds / vs2
            r = r_shift + r_scale

            g0, g1 = gates[k, t], gates[k, t + 1]
            w = g0 * g1
            wr = w * r

            pair_values[k, t] = r
            pair_weights[k, t] = w
            weighted_values[k, t] = wr
            shift_motion[k, t] = r_shift
            scale_motion[k, t] = r_scale
            value += wr

            # Derivatives of r with respect to centroids.
            dr_dn0 = -2.0 * dn / vn2
            dr_dn1 = 2.0 * dn / vn2
            dr_ds0 = -2.0 * ds / vs2
            dr_ds1 = 2.0 * ds / vs2

            gx = w * (
                dr_dn0 * dn_dx[k, t]
                + dr_ds0 * ds_dx[k, t]
            )

            gy = w * (
                dr_dn1 * dn_dx[k, t + 1]
                + dr_ds1 * ds_dx[k, t + 1]
            )

            # Gate derivatives. Each frame mass derivative is 1.
            dw_dm0 = squared_gate_derivative(
                masses[k, t], gate_mass
            ) * g1

            dw_dm1 = g0 * squared_gate_derivative(
                masses[k, t + 1], gate_mass
            )

            gx += r * dw_dm0
            gy += r * dw_dm1

            grad[:, k, :, t] += gx.reshape(L, J)
            grad[:, k, :, t + 1] += gy.reshape(L, J)

    diagnostics = {
        "masses": masses,
        "frame_gates": gates,
        "position_centroids": center_n,
        "log_scale_centroids": center_s,
        "pair_weights": pair_weights,
        "pair_values": pair_values,
        "weighted_pair_values": weighted_values,
        "shift_motion": shift_motion,
        "scale_motion": scale_motion,
        "shift_velocity": float(shift_velocity),
        "scale_velocity": float(scale_velocity),
        "mass_floor": float(mass_floor),
        "gate_mass": float(gate_mass),
    }

    return float(value), grad, diagnostics


def make_barycentric_transition_regularizer(
    shift_coordinates,
    scale_coordinates,
    gate_mass=0.05,
    mass_floor=1e-10,
    shift_velocity=1.0,
    scale_velocity=None,
):
    """Create an x_update-compatible callback."""
    def regularizer(X):
        value, grad, _ = barycentric_transition_value_grad(
            X,
            shift_coordinates=shift_coordinates,
            scale_coordinates=scale_coordinates,
            gate_mass=gate_mass,
            mass_floor=mass_floor,
            shift_velocity=shift_velocity,
            scale_velocity=scale_velocity,
        )
        return value, grad

    return regularizer