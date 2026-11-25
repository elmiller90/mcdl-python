import numpy as np

from barycentric_transition import (
    squared_gate,
    squared_gate_derivative,
)


def barycentric_transition_value_grad_vectorized(
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
    if gate_mass <= 0 or mass_floor <= 0:
        raise ValueError("gate_mass and mass_floor must be positive")
    if shift_velocity <= 0:
        raise ValueError("shift_velocity must be positive")
    if spread_weight < 0:
        raise ValueError("spread_weight must be nonnegative")

    L, K, J, T = X.shape
    if ncoord.shape != (L,):
        raise ValueError(f"shift_coordinates must have shape ({L},)")
    if scales.shape != (J,):
        raise ValueError(f"scale_coordinates must have shape ({J},)")
    if np.any(scales <= 0):
        raise ValueError("scale_coordinates must be positive")

    slog = np.log(scales)
    if scale_velocity is None:
        spacing = np.abs(np.diff(slog))
        spacing = spacing[spacing > 0]
        scale_velocity = (
            float(np.median(spacing)) if spacing.size else 1.0
        )
    if scale_velocity <= 0:
        raise ValueError("scale_velocity must be positive")

    # Arrange as (K,T,P), P=L*J.
    P = L*J
    Z = X.transpose(1, 3, 0, 2).reshape(K, T, P)

    n_grid = np.repeat(ncoord, J)
    s_grid = np.tile(slog, L)
    n_scaled = n_grid/shift_velocity
    s_scaled = s_grid/scale_velocity

    background = np.full(P, 1.0/P)

    # --------------------------------------------------
    # Frame masses, probabilities, gates, barycenters.
    # --------------------------------------------------
    masses = Z.sum(axis=2)
    denom = masses + mass_floor

    Pdist = (
        Z + mass_floor*background[None, None, :]
    ) / denom[:, :, None]

    gates = masses*masses / (
        masses*masses + gate_mass*gate_mass
    )

    gate_prime = (
        2.0*masses*gate_mass*gate_mass
        / (masses*masses + gate_mass*gate_mass)**2
    )

    center_n = np.einsum("ktp,p->kt", Pdist, n_grid)
    center_s = np.einsum("ktp,p->kt", Pdist, s_grid)

    dn_dx = (
        n_grid[None, None, :]
        - center_n[:, :, None]
    ) / denom[:, :, None]

    ds_dx = (
        s_grid[None, None, :]
        - center_s[:, :, None]
    ) / denom[:, :, None]

    # --------------------------------------------------
    # Within-frame spread.
    # --------------------------------------------------
    n2 = n_scaled*n_scaled
    s2 = s_scaled*s_scaled

    mean_ns = np.einsum(
        "ktp,p->kt", Pdist, n_scaled
    )
    mean_ss = np.einsum(
        "ktp,p->kt", Pdist, s_scaled
    )

    second_n = np.einsum("ktp,p->kt", Pdist, n2)
    second_s = np.einsum("ktp,p->kt", Pdist, s2)

    spread_position = np.maximum(
        second_n-mean_ns*mean_ns, 0.0
    )
    spread_scale = np.maximum(
        second_s-mean_ss*mean_ss, 0.0
    )
    frame_spread = spread_position+spread_scale

    gv_p = (
        n2[None, None, :]
        - 2.0*mean_ns[:, :, None]*n_scaled[None, None, :]
        + s2[None, None, :]
        - 2.0*mean_ss[:, :, None]*s_scaled[None, None, :]
    )

    gv_dot_p = np.sum(gv_p*Pdist, axis=2)
    spread_grad = (
        gv_p-gv_dot_p[:, :, None]
    ) / denom[:, :, None]

    grad = np.zeros_like(Z)

    spread_value = float(
        np.sum(gates*frame_spread)
    )

    if spread_weight > 0:
        grad += spread_weight * (
            gates[:, :, None]*spread_grad
            + (
                frame_spread*gate_prime
            )[:, :, None]
        )

    # --------------------------------------------------
    # Pairwise barycentric motion.
    # --------------------------------------------------
    if T > 1:
        delta_n = center_n[:, 1:]-center_n[:, :-1]
        delta_s = center_s[:, 1:]-center_s[:, :-1]

        rn = delta_n*delta_n/(shift_velocity**2)
        rs = delta_s*delta_s/(scale_velocity**2)
        pair_values = rn+rs

        pair_weights = gates[:, :-1]*gates[:, 1:]
        weighted = pair_weights*pair_values

        motion_value = float(weighted.sum())

        dr_dn0 = -2.0*delta_n/(shift_velocity**2)
        dr_dn1 = -dr_dn0
        dr_ds0 = -2.0*delta_s/(scale_velocity**2)
        dr_ds1 = -dr_ds0

        g0 = pair_weights[:, :, None] * (
            dr_dn0[:, :, None]*dn_dx[:, :-1, :]
            + dr_ds0[:, :, None]*ds_dx[:, :-1, :]
        )
        g1 = pair_weights[:, :, None] * (
            dr_dn1[:, :, None]*dn_dx[:, 1:, :]
            + dr_ds1[:, :, None]*ds_dx[:, 1:, :]
        )

        dw0 = gate_prime[:, :-1]*gates[:, 1:]
        dw1 = gates[:, :-1]*gate_prime[:, 1:]

        g0 += (
            pair_values*dw0
        )[:, :, None]
        g1 += (
            pair_values*dw1
        )[:, :, None]

        grad[:, :-1, :] += g0
        grad[:, 1:, :] += g1

    else:
        pair_values = np.zeros((K, 0))
        pair_weights = np.zeros((K, 0))
        weighted = np.zeros((K, 0))
        rn = np.zeros((K, 0))
        rs = np.zeros((K, 0))
        motion_value = 0.0

    value = motion_value + spread_weight*spread_value

    grad = (
        grad.reshape(K, T, L, J)
        .transpose(2, 0, 3, 1)
    )

    diagnostics = {
        "masses": masses,
        "frame_gates": gates,
        "position_centroids": center_n,
        "log_scale_centroids": center_s,
        "pair_weights": pair_weights,
        "pair_values": pair_values,
        "weighted_pair_values": weighted,
        "shift_motion": rn,
        "scale_motion": rs,
        "motion_value": motion_value,
        "frame_spread": frame_spread,
        "spread_position": spread_position,
        "spread_scale": spread_scale,
        "spread_value": spread_value,
        "weighted_spread_value":
            float(spread_weight*spread_value),
        "spread_weight": float(spread_weight),
        "shift_velocity": float(shift_velocity),
        "scale_velocity": float(scale_velocity),
        "mass_floor": float(mass_floor),
        "gate_mass": float(gate_mass),
    }

    return float(value), grad, diagnostics


def make_barycentric_transition_regularizer_vectorized(
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
            barycentric_transition_value_grad_vectorized(
                X,
                shift_coordinates=shift_coordinates,
                scale_coordinates=scale_coordinates,
                gate_mass=gate_mass,
                mass_floor=mass_floor,
                shift_velocity=shift_velocity,
                scale_velocity=scale_velocity,
                spread_weight=spread_weight,
            )
        )
        return value, grad

    return regularizer