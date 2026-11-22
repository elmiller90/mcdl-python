import argparse
import json
from pathlib import Path

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


def run_case(name, lambda_t, reg, S, Sclean, Xtrue, Gtrue, G0, X0,
             scales, center, N, M, expected_noise):
    G = G0.copy()
    H = np.zeros_like(G)
    Xstart = X0.copy()
    state = None

    for _ in range(25):
        A = MultiscaleConvolutionOperator(
            D=G, scales=scales, center=center, N=N, normalize=True,
        )

        _, Y, _, info = solve_log_admm_stateful(
            X0=Xstart, S=S, operator=A,
            lambda_s=0.03, lambda2=lambda_t, tau=1e-2,
            rho=0.1, max_admm_iter=200, x_maxiter=50,
            abs_tol=1e-4, rel_tol=1e-3, adaptive_rho=True,
            rho_mu=10.0, rho_tau=2.0, a_min=1e-4,
            continuation_iters=100, a_fraction=0.95,
            verbose=False, temporal_regularizer=reg,
            state0=state, continue_iteration=False,
        )

        state = info["state"]
        Xstart = Y.copy()

        alpha, _ = compute_dictionary_alpha(G, scales, center, N)

        B = DictionaryForwardOperator(
            X=Y, alpha=alpha, scales=scales,
            center=center, N=N, M=M,
        )

        D, cginfo = solve_dictionary_cg(
            op=B, S=S, G=G, H=H, rho=1.0,
            D0=G, rtol=1e-10, maxiter=500,
        )
        if cginfo != 0:
            raise RuntimeError(f"dictionary CG failed: {cginfo}")

        Gnew = project_dictionary_nonnegative(D + H)
        H = H + D - Gnew
        G = Gnew

    A = MultiscaleConvolutionOperator(
        D=G, scales=scales, center=center, N=N, normalize=True,
    )

    Shat = A.forward(Y)
    C = similarity_matrix(G, Gtrue)

    return {
        "name": name,
        "clean_error": relerr(Shat, Sclean),
        "discrepancy": 0.5*np.sum((Shat-S)**2)/expected_noise,
        "coefficient_error": relerr(Y, Xtrue),
        "atom_cosine_0": float(C[0, 0]),
        "atom_cosine_1": float(C[1, 1]),
        "significant_count": int(np.count_nonzero(np.abs(Y) > 1e-3)),
        "admm_converged": bool(info["converged"]),
        "admm_iterations": int(info["iterations"]),
        "final_rho": float(info["rho"]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snr-db", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    M, N, center = 41, 96, 21
    scales = [(1, 2), (3, 4), (1, 1)]

    Gtrue = make_dictionary(M)

    Atrue = MultiscaleConvolutionOperator(
        D=Gtrue, scales=scales, center=center, N=N, normalize=True,
    )

    Xtrue = make_coefficients(Atrue.L, M)
    Sclean = Atrue.forward(Xtrue)

    S, _, achieved_snr, sigma = add_noise_snr(
        Sclean, args.snr_db, np.random.default_rng(args.seed),
        return_sigma=True,
    )
    expected_noise = expected_noise_energy(Sclean, sigma)

    # Fixed dictionary perturbation across all SNR/seed trials.
    G0 = Gtrue.copy()
    G0 += 0.10*np.random.default_rng(40).random(G0.shape)
    G0 = project_dictionary_nonnegative(G0)

    A0 = MultiscaleConvolutionOperator(
        D=G0, scales=scales, center=center, N=N, normalize=True,
    )

    X0 = A0.adjoint(S, structured=True)
    xmax = np.max(np.abs(X0))
    if xmax > 0:
        X0 /= xmax

    shifts = np.arange(Atrue.L, dtype=float) - (M-1)
    scale_coords = physical_scale_values(scales)

    reg = make_barycentric_transition_regularizer(
        shift_coordinates=shifts,
        scale_coordinates=scale_coords,
        gate_mass=0.05,
        mass_floor=1e-10,
        shift_velocity=1.0,
        spread_weight=0.10,
    )

    non_temporal = run_case(
        "non-temporal", 0.0, None,
        S, Sclean, Xtrue, Gtrue, G0, X0,
        scales, center, N, M, expected_noise,
    )

    temporal = run_case(
        "temporal", 3e-5, reg,
        S, Sclean, Xtrue, Gtrue, G0, X0,
        scales, center, N, M, expected_noise,
    )

    result = {
        "requested_snr_db": args.snr_db,
        "achieved_snr_db": achieved_snr,
        "sigma": sigma,
        "seed": args.seed,
        "true_atom_cosine": cosine(Gtrue[:, 0], Gtrue[:, 1]),
        "lambda_s": 0.03,
        "lambda_t": 3e-5,
        "beta_v": 0.10,
        "non_temporal": non_temporal,
        "temporal": temporal,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: partial jobs never masquerade as completed trials.
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(args.output)

    print(
        f"SNR={args.snr_db:g} seed={args.seed} "
        f"achieved={achieved_snr:.2f}"
    )
    print(
        "case          cleanerr   disc    Xerr      "
        "cos1      cos2      nz"
    )

    for r in (non_temporal, temporal):
        print(
            f"{r['name']:12s} "
            f"{r['clean_error']:9.3e} "
            f"{r['discrepancy']:6.3f} "
            f"{r['coefficient_error']:9.3e} "
            f"{r['atom_cosine_0']:9.6f} "
            f"{r['atom_cosine_1']:9.6f} "
            f"{r['significant_count']:4d}"
        )

    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()