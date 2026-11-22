import numpy as np

from log_admm_stateful_compat import solve_log_admm_stateful
from multiscale_convolution import MultiscaleConvolutionOperator
from barycentric_transition import make_barycentric_transition_regularizer
from local_graph import physical_scale_values

def relerr(a, b):
    return np.linalg.norm(a - b) / max(
        np.linalg.norm(b), 1e-15
    )


def make_dictionary(M):
    x = np.arange(M, dtype=float)
    c = (M - 1) / 2.0

    d = np.exp(
        -0.5 * ((x - c) / 4.0) ** 2
    )
    d /= np.linalg.norm(d)

    return d[:, None]


def make_coefficients(L, M):
    X = np.zeros((L, 1, 3, 5))

    positions = [36, 37, 38, 39, 40]
    scales = [0, 1, 2, 1, 0]

    for t, (p, j) in enumerate(
        zip(positions, scales)
    ):
        n = (M - 1) + p
        X[n, 0, j, t] = 1.0

    return X


def dominant_trajectory(X, M):
    positions = []
    scales = []
    amplitudes = []

    for t in range(X.shape[3]):
        frame = np.abs(X[:, 0, :, t])

        n, j = np.unravel_index(
            np.argmax(frame),
            frame.shape,
        )

        positions.append(
            int(n - (M - 1))
        )
        scales.append(int(j))
        amplitudes.append(
            float(X[n, 0, j, t])
        )

    return positions, scales, amplitudes


def main():
    M = 41
    N = 96
    center = 21

    scales = [
        (1, 2),
        (1, 1),
        (2, 1),
    ]

    Gtrue = make_dictionary(M)

    A = MultiscaleConvolutionOperator(
        D=Gtrue,
        scales=scales,
        center=center,
        N=N,
        normalize=True,
    )
    
    shifts = np.arange(A.L, dtype=float) - (M - 1)
    scale_coords = physical_scale_values(scales)

    reg = make_barycentric_transition_regularizer(
        shift_coordinates=shifts,
        scale_coordinates=scale_coords,
        gate_mass=0.05,
        mass_floor=1e-10,
        shift_velocity=1.0,
        spread_weight=0.10,
    )

    Xtrue = make_coefficients(
        A.L,
        M,
    )

    # Exact noiseless observations.
    S = A.forward(Xtrue)

    # Same adjoint initialization used in the
    # validated coefficient-recovery tests.
    X0 = A.adjoint(
        S,
        structured=True,
    )

    xmax = np.max(np.abs(X0))
    if xmax > 0:
        X0 /= xmax

    # -------------------------------------------------
    # Log-sparse control parameters
    # -------------------------------------------------

    rho = 0.1
    lambda_s = 0.03
    lambda_t = 3e-5

    _, Y, _, info = solve_log_admm_stateful(
        X0=X0, S=S, operator=A,
        lambda_s=lambda_s, lambda2=lambda_t, tau=1e-2,
        rho=0.1, max_admm_iter=600, x_maxiter=50,
        abs_tol=1e-4, rel_tol=1e-3, adaptive_rho=True,
        rho_mu=10.0, rho_tau=2.0, a_min=1e-4,
        continuation_iters=100, a_fraction=0.95,
        verbose=False, temporal_regularizer=reg,
        state0=None, continue_iteration=False,
    )

    recon = relerr(A.forward(Y), S)
    xerr = relerr(Y, Xtrue)
    nz = np.count_nonzero(np.abs(Y) > 1e-3)
    pos, scales_hat, amps = dominant_trajectory(Y, M)

    print("\nExact-D log + temporal/spread control")
    print("-------------------------------------")
    print(f"lambda_s={lambda_s:g} lambda_t={lambda_t:g} beta_v=0.10")
    print(f"recon={recon:.3e} Xerr={xerr:.3e} nz={nz}")
    print(f"positions={pos}")
    print(f"scales={scales_hat}")
    print(f"amplitudes={[round(a, 6) for a in amps]}")

    print("\nSignificant coefficients:")
    for n, k, j, t in np.argwhere(np.abs(Y) > 1e-3):
        print(
            f"t={t} pos={n-(M-1)} scale={j} "
            f"amp={Y[n,k,j,t]:.6f}"
        )
 


if __name__ == "__main__":
    main()