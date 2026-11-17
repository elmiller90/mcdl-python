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

def variance_value_grad_p(
    p,
    coord,
):
    """
    Variance of coordinate under probability p:

        V = E[c^2] - E[c]^2

    Returns V and dV/dp.
    """
    mean = float(
        np.dot(p, coord)
    )

    second = float(
        np.dot(p, coord * coord)
    )

    value = second - mean * mean

    grad_p = (
        coord * coord
        - 2.0 * mean * coord
    )

    return float(max(value, 0.0)), grad_p

def barycentric_transition_value_grad(
    X,
    shift_coordinates,
    scale_coordinates,
    gate_mass=0.05,
    mass_floor=1e-10,
    shift_velocity=1.0,
    scale_velocity=None,
    spread_weight=0.0,
):
    X = np.asarray(X, dtype=np.float64)
    ncoord = np.asarray(shift_coordinates, dtype=np.float64)
    scales = np.asarray(scale_coordinates, dtype=np.float64)

    if X.ndim != 4:
        raise ValueError("X must have shape (L,K,J,T)")
    if np.any(X < 0):
        raise ValueError("X must be nonnegative")
    if gate_mass <= 0:
        raise ValueError("gate_mass must be positive")
    if mass_floor <= 0:
        raise ValueError("mass_floor must be positive")
    if shift_velocity <= 0:
        raise ValueError("shift_velocity must be positive")
    if spread_weight < 0:
        raise ValueError("spread_weight must be nonnegative")

    L, K, J, T = X.shape

    if ncoord.shape != (L,):
        raise ValueError(
            f"shift_coordinates must have shape ({L},)"
        )
    if scales.shape != (J,):
        raise ValueError(
            f"scale_coordinates must have shape ({J},)"
        )
    if np.any(scales <= 0):
        raise ValueError("scale_coordinates must be positive")

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

    # Flattening convention: index = n*J + j.
    n_grid = np.repeat(ncoord, J)
    s_grid = np.tile(log_scales, L)

    n_scaled = n_grid / shift_velocity
    s_scaled = s_grid / scale_velocity

    P = L * J
    background = np.full(P, 1.0 / P)

    masses = np.zeros((K, T))
    gates = np.zeros((K, T))
    center_n = np.zeros((K, T))
    center_s = np.zeros((K, T))

    frame_spread = np.zeros((K, T))
    spread_position = np.zeros((K, T))
    spread_scale = np.zeros((K, T))

    dn_dx = np.zeros((K, T, P))
    ds_dx = np.zeros((K, T, P))
    spread_frame_grad = np.zeros_like(X)

    # -----------------------------------------------------
    # Per-frame moments and gradients
    # -----------------------------------------------------

    for k in range(K):
        for t in range(T):
            x = X[:, k, :, t].reshape(-1)

            m = float(x.sum())
            denom = m + mass_floor

            p = (
                x + mass_floor * background
            ) / denom

            bn = float(np.dot(p, n_grid))
            bs = float(np.dot(p, s_grid))

            masses[k, t] = m
            gates[k, t] = squared_gate(
                m, gate_mass
            )

            center_n[k, t] = bn
            center_s[k, t] = bs

            dn_dx[k, t] = (
                n_grid - bn
            ) / denom

            ds_dx[k, t] = (
                s_grid - bs
            ) / denom

            vn, gvn_p = variance_value_grad_p(
                p, n_scaled
            )

            vs, gvs_p = variance_value_grad_p(
                p, s_scaled
            )

            v = vn + vs

            frame_spread[k, t] = v
            spread_position[k, t] = vn
            spread_scale[k, t] = vs

            gv_p = gvn_p + gvs_p

            gv_x = (
                gv_p - np.dot(gv_p, p)
            ) / denom

            spread_frame_grad[:, k, :, t] = (
                gv_x.reshape(L, J)
            )

    # -----------------------------------------------------
    # Pairwise barycentric motion
    # -----------------------------------------------------

    grad = np.zeros_like(X)

    pair_shape = (
        K,
        max(T - 1, 0),
    )

    pair_values = np.zeros(pair_shape)
    pair_weights = np.zeros(pair_shape)
    weighted_values = np.zeros(pair_shape)
    shift_motion = np.zeros(pair_shape)
    scale_motion = np.zeros(pair_shape)

    motion_value = 0.0

    vn2 = shift_velocity ** 2
    vs2 = scale_velocity ** 2

    for k in range(K):
        for t in range(T - 1):
            dn = (
                center_n[k, t + 1]
                - center_n[k, t]
            )

            ds = (
                center_s[k, t + 1]
                - center_s[k, t]
            )

            rn = dn * dn / vn2
            rs = ds * ds / vs2
            r = rn + rs

            g0 = gates[k, t]
            g1 = gates[k, t + 1]
            w = g0 * g1

            pair_values[k, t] = r
            pair_weights[k, t] = w
            weighted_values[k, t] = w * r
            shift_motion[k, t] = rn
            scale_motion[k, t] = rs

            motion_value += w * r

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

            dw_dm0 = (
                squared_gate_derivative(
                    masses[k, t],
                    gate_mass,
                )
                * g1
            )

            dw_dm1 = (
                g0
                * squared_gate_derivative(
                    masses[k, t + 1],
                    gate_mass,
                )
            )

            gx += r * dw_dm0
            gy += r * dw_dm1

            grad[:, k, :, t] += (
                gx.reshape(L, J)
            )

            grad[:, k, :, t + 1] += (
                gy.reshape(L, J)
            )

    # -----------------------------------------------------
    # Gated within-frame spread
    # -----------------------------------------------------

    spread_value = 0.0

    if spread_weight > 0:
        for k in range(K):
            for t in range(T):
                m = masses[k, t]
                g = gates[k, t]
                v = frame_spread[k, t]

                spread_value += g * v

                grad[:, k, :, t] += (
                    spread_weight
                    * g
                    * spread_frame_grad[
                        :, k, :, t
                    ]
                )

                # d[g(m)V]/dx =
                # g dV/dx + V g'(m) 1.
                grad[:, k, :, t] += (
                    spread_weight
                    * v
                    * squared_gate_derivative(
                        m,
                        gate_mass,
                    )
                )

    value = (
        motion_value
        + spread_weight * spread_value
    )

    diagnostics = {
        "masses": masses,
        "frame_gates": gates,
        "position_centroids": center_n,
        "log_scale_centroids": center_s,
        "pair_weights": pair_weights,
        "pair_values": pair_values,
        "weighted_pair_values":
            weighted_values,
        "shift_motion": shift_motion,
        "scale_motion": scale_motion,
        "motion_value":
            float(motion_value),
        "frame_spread": frame_spread,
        "spread_position":
            spread_position,
        "spread_scale":
            spread_scale,
        "spread_value":
            float(spread_value),
        "weighted_spread_value":
            float(
                spread_weight
                * spread_value
            ),
        "spread_weight":
            float(spread_weight),
        "shift_velocity":
            float(shift_velocity),
        "scale_velocity":
            float(scale_velocity),
        "mass_floor":
            float(mass_floor),
        "gate_mass":
            float(gate_mass),
    }

    return float(value), grad, diagnostics


def make_barycentric_transition_regularizer(
    shift_coordinates,
    scale_coordinates,
    gate_mass=0.05,
    mass_floor=1e-10,
    shift_velocity=1.0,
    scale_velocity=None,
    spread_weight=0.0,
):
    def regularizer(X):
        value, grad, _ = (
            barycentric_transition_value_grad(
                X,
                shift_coordinates=
                    shift_coordinates,
                scale_coordinates=
                    scale_coordinates,
                gate_mass=gate_mass,
                mass_floor=mass_floor,
                shift_velocity=
                    shift_velocity,
                scale_velocity=
                    scale_velocity,
                spread_weight=
                    spread_weight,
            )
        )

        return value, grad

    return regularizer