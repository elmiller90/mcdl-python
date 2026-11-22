import numpy as np

from barycentric_transition import make_barycentric_transition_regularizer
from dictionary_alpha import compute_dictionary_alpha
from dictionary_cg import solve_dictionary_cg
from dictionary_forward_operator import DictionaryForwardOperator
from dictionary_projection import project_dictionary_nonnegative
from gaussian_experiment import add_noise_snr, expected_noise_energy
from local_graph import physical_scale_values
from log_admm_stateful_compat import solve_log_admm_stateful
from multiscale_convolution import MultiscaleConvolutionOperator
from experiments.dictionary_learning.alternating_two_atom_log_noisy import (
    relerr, cosine, similarity_matrix, make_dictionary, make_coefficients,
)


def run_case(name, temporal, S, Sclean, Xtrue, Gtrue, X0, scales,
             center, N, M, expected_noise):
    G = Gtrue.copy()
    rng = np.random.default_rng(40)
    G += 0.10*rng.random(G.shape)
    G = project_dictionary_nonnegative(G)
    H = np.zeros_like(G)

    lambda_s = 0.03
    rho_x, rho_d = 0.1, 1.0
    state = None
    Xstart = X0.copy()

    for outer in range(1, 26):
        A = MultiscaleConvolutionOperator(
            D=G, scales=scales, center=center,
            N=N, normalize=True,
        )

        _, Y, _, info = solve_log_admm_stateful(
            X0=Xstart, S=S, operator=A,
            lambda_s=lambda_s,
            lambda2=temporal["lambda_t"], tau=1e-2,
            rho=rho_x, max_admm_iter=200, x_maxiter=50,
            abs_tol=1e-4, rel_tol=1e-3, adaptive_rho=True,
            rho_mu=10.0, rho_tau=2.0, a_min=1e-4,
            continuation_iters=100, a_fraction=0.95,
            verbose=False,
            temporal_regularizer=temporal["reg"],
            state0=state, continue_iteration=False,
        )

        state = info["state"]
        Xstart = Y.copy()

        alpha, _ = compute_dictionary_alpha(
            G, scales, center, N
        )

        B = DictionaryForwardOperator(
            X=Y, alpha=alpha, scales=scales,
            center=center, N=N, M=M,
        )

        D, cginfo = solve_dictionary_cg(
            op=B, S=S, G=G, H=H, rho=rho_d,
            D0=G, rtol=1e-10, maxiter=500,
        )
        if cginfo != 0:
            raise RuntimeError(f"CG failed: {cginfo}")

        Gnew = project_dictionary_nonnegative(D+H)
        H = H+D-Gnew
        G = Gnew

    A = MultiscaleConvolutionOperator(
        D=G, scales=scales, center=center,
        N=N, normalize=True,
    )

    Shat = A.forward(Y)
    C = similarity_matrix(G, Gtrue)

    return {
        "name": name,
        "clean": relerr(Shat, Sclean),
        "disc": 0.5*np.sum((Shat-S)**2)/expected_noise,
        "xerr": relerr(Y, Xtrue),
        "cos1": C[0, 0],
        "cos2": C[1, 1],
        "nz": int(np.count_nonzero(np.abs(Y) > 1e-3)),
        "Y": Y,
    }


def main():
    M, N, center = 41, 96, 21
    scales = [(1, 2), (3, 4), (1, 1)]

    Gtrue = make_dictionary(M)
    Atrue = MultiscaleConvolutionOperator(
        D=Gtrue, scales=scales, center=center,
        N=N, normalize=True,
    )

    Xtrue = make_coefficients(Atrue.L, M)
    Sclean = Atrue.forward(Xtrue)

    S, _, achieved_snr, sigma = add_noise_snr(
        Sclean, 20.0, np.random.default_rng(100),
        return_sigma=True,
    )
    expected_noise = expected_noise_energy(Sclean, sigma)

    # Identical perturbed dictionary gives identical X0 for both cases.
    G0 = Gtrue.copy()
    G0 += 0.10*np.random.default_rng(40).random(G0.shape)
    G0 = project_dictionary_nonnegative(G0)

    A0 = MultiscaleConvolutionOperator(
        D=G0, scales=scales, center=center,
        N=N, normalize=True,
    )
    X0 = A0.adjoint(S, structured=True)
    X0 /= np.max(np.abs(X0))

    shifts = np.arange(Atrue.L, dtype=float)-(M-1)
    scale_coords = physical_scale_values(scales)

    reg = make_barycentric_transition_regularizer(
        shift_coordinates=shifts,
        scale_coordinates=scale_coords,
        gate_mass=0.05,
        mass_floor=1e-10,
        shift_velocity=1.0,
        spread_weight=0.10,
    )

    cases = [
        ("non-temporal", {"lambda_t": 0.0, "reg": None}),
        ("temporal", {"lambda_t": 3e-5, "reg": reg}),
    ]

    print(f"true atom cosine: {cosine(Gtrue[:,0], Gtrue[:,1]):.6f}")
    print(f"achieved SNR: {achieved_snr:.2f} dB\n")

    results = [
        run_case(
            name, cfg, S, Sclean, Xtrue, Gtrue, X0,
            scales, center, N, M, expected_noise,
        )
        for name, cfg in cases
    ]

    print("case          cleanerr   disc    Xerr      cos1      cos2      nz")
    for r in results:
        print(
            f"{r['name']:12s} {r['clean']:9.3e} {r['disc']:6.3f} "
            f"{r['xerr']:9.3e} {r['cos1']:9.6f} "
            f"{r['cos2']:9.6f} {r['nz']:4d}"
        )

    for r in results:
        print(f"\n{r['name']} significant coefficients:")
        for n, k, j, t in np.argwhere(np.abs(r["Y"]) > 1e-3):
            print(
                f"t={t} atom={k} pos={n-(M-1)} "
                f"scale={j} amp={r['Y'][n,k,j,t]:.6f}"
            )


if __name__ == "__main__":
    main()

    