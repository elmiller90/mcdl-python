from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from logging import config
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from multiscale_convolution import MultiscaleConvolutionOperator
from sparse_admm import solve_l1_admm


@dataclass
class ExperimentConfig:
    seed: int = 100
    M: int = 41
    N: int = 96
    T: int = 7
    gaussian_width: float = 4.0
    snr_db: float = 20.0
    lambda1: float = 0.05
    lambda2: float = 0.10
    tau: float = 1e-2
    rho: float = 0.10
    max_admm_iter: int = 200
    x_maxiter: int = 50
    abs_tol: float = 1e-4
    rel_tol: float = 1e-3
    nonnegative: bool = True
    adaptive_rho: bool = True


SCALES = [
    (1, 1),
    (4, 5),
    (3, 4),
    (2, 3),
    (1, 2),
]


TRUE_PHYSICAL_POSITIONS = [
    28,
    29,
    30,
    31,
    32,
    33,
    34,
]


TRUE_SCALE_INDICES = [
    0,
    1,
    2,
    3,
    2,
    1,
    0,
]
def json_default(obj):
    """
    Convert NumPy scalar/array objects to JSON-safe
    Python objects.
    """
    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(
        f"Object of type {type(obj).__name__} "
        "is not JSON serializable"
    )

def make_dictionary(
    M: int,
    width: float,
) -> np.ndarray:
    x = np.arange(M, dtype=np.float64)
    center = (M - 1) / 2.0

    atom = np.exp(
        -0.5 * ((x - center) / width) ** 2
    )

    atom /= np.linalg.norm(atom)

    return atom[:, None]


def physical_to_coefficient(
    physical_position: int,
    M: int,
) -> int:
    return (M - 1) + physical_position


def coefficient_to_physical(
    coefficient_position: int,
    M: int,
) -> int:
    return coefficient_position - (M - 1)


def make_true_coefficients(
    L: int,
    M: int,
    J: int,
    T: int,
) -> tuple[np.ndarray, list[int]]:
    if T != len(TRUE_PHYSICAL_POSITIONS):
        raise ValueError(
            "The canonical trajectory requires T=7"
        )

    X_true = np.zeros(
        (L, 1, J, T),
        dtype=np.float64,
    )

    coefficient_positions = [
        physical_to_coefficient(n, M)
        for n in TRUE_PHYSICAL_POSITIONS
    ]

    for t in range(T):
        X_true[
            coefficient_positions[t],
            0,
            TRUE_SCALE_INDICES[t],
            t,
        ] = 1.0

    return X_true, coefficient_positions


def add_noise_snr(
    clean: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Add Gaussian noise with exact realized Frobenius-norm SNR.

        SNR_dB = 20 log10(||clean||_F / ||noise||_F)
    """
    clean = np.asarray(
        clean,
        dtype=np.float64,
    )

    clean_norm = np.linalg.norm(clean)

    if clean_norm == 0:
        raise ValueError(
            "Cannot add SNR-controlled noise "
            "to a zero signal"
        )

    noise = rng.normal(
        size=clean.shape
    )

    noise_norm = np.linalg.norm(noise)

    if noise_norm == 0:
        raise RuntimeError(
            "Generated a zero noise realization"
        )

    target_noise_norm = (
        clean_norm
        / (10.0 ** (snr_db / 20.0))
    )

    noise *= target_noise_norm / noise_norm

    noisy = clean + noise

    achieved_snr_db = 20.0 * np.log10(
        clean_norm / np.linalg.norm(noise)
    )

    return noisy, noise, achieved_snr_db


def relative_error(
    estimate: np.ndarray,
    truth: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(estimate - truth)
        / max(1e-15, np.linalg.norm(truth))
    )

def reconstruction_snr_db(
    clean: np.ndarray,
    recovered: np.ndarray,
) -> float:
    """
    SNR of the recovered signal relative to clean truth.

        SNR_out =
            20 log10(
                ||clean||_F /
                ||clean - recovered||_F
            )
    """
    clean = np.asarray(
        clean,
        dtype=np.float64,
    )

    recovered = np.asarray(
        recovered,
        dtype=np.float64,
    )

    signal_norm = np.linalg.norm(clean)

    error_norm = np.linalg.norm(
        clean - recovered
    )

    if signal_norm == 0:
        raise ValueError(
            "Cannot compute SNR for a zero signal"
        )

    if error_norm == 0:
        return float("inf")

    return float(
        20.0
        * np.log10(
            signal_norm / error_norm
        )
    )

def dominant_trajectory(
    X: np.ndarray,
    M: int,
) -> dict:
    L, K, J, T = X.shape

    if K != 1:
        raise ValueError(
            "Canonical experiment assumes K=1"
        )

    coefficient_positions = []
    physical_positions = []
    scale_indices = []
    amplitudes = []

    for t in range(T):
        frame = np.abs(
            X[:, 0, :, t]
        )

        flat_index = int(
            np.argmax(frame)
        )

        coefficient_position, scale_index = (
            np.unravel_index(
                flat_index,
                frame.shape,
            )
        )

        coefficient_positions.append(
            int(coefficient_position)
        )

        physical_positions.append(
            int(
                coefficient_to_physical(
                    coefficient_position,
                    M,
                )
            )
        )

        scale_indices.append(
            int(scale_index)
        )

        amplitudes.append(
            float(
                X[
                    coefficient_position,
                    0,
                    scale_index,
                    t,
                ]
            )
        )

    return {
        "coefficient_positions":
            coefficient_positions,
        "physical_positions":
            physical_positions,
        "scale_indices":
            scale_indices,
        "amplitudes":
            amplitudes,
    }


def save_cost_plot(
    info: dict,
    output_path: Path,
) -> None:
    history = info["history"]

    iteration = np.array([
        row["iteration"]
        for row in history
    ])

    total = np.array([
        row["objective"]
        for row in history
    ])

    data = np.array([
        row["data"]
        for row in history
    ])

    l1 = np.array([
        row["l1"]
        for row in history
    ])

    softmin = np.array([
        row["softmin"]
        for row in history
    ])

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        iteration,
        total,
        label="Total objective",
        linewidth=2,
    )

    ax.plot(
        iteration,
        data,
        label="Data fidelity",
    )

    ax.plot(
        iteration,
        l1,
        label="L1 penalty",
    )

    ax.plot(
        iteration,
        softmin,
        label="Weighted softmin",
    )

    ax.set_xlabel("ADMM iteration")
    ax.set_ylabel("Reported unsplit objective")
    ax.set_title("Objective components")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)


def save_residual_plot(
    info: dict,
    output_path: Path,
) -> None:
    history = info["history"]

    iteration = np.array([
        row["iteration"]
        for row in history
    ])

    primal = np.array([
        row["primal_residual"]
        for row in history
    ])

    dual = np.array([
        row["dual_residual"]
        for row in history
    ])

    primal_tol = np.array([
        row["primal_tolerance"]
        for row in history
    ])

    dual_tol = np.array([
        row["dual_tolerance"]
        for row in history
    ])

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.semilogy(
        iteration,
        primal,
        label="Primal residual",
        linewidth=2,
    )

    ax.semilogy(
        iteration,
        dual,
        label="Dual residual",
        linewidth=2,
    )

    ax.semilogy(
        iteration,
        primal_tol,
        "--",
        label="Primal tolerance",
    )

    ax.semilogy(
        iteration,
        dual_tol,
        "--",
        label="Dual tolerance",
    )

    ax.set_xlabel("ADMM iteration")
    ax.set_ylabel("Residual norm")
    ax.set_title("ADMM convergence")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)


def save_rho_plot(
    info: dict,
    output_path: Path,
) -> None:
    history = info["history"]

    iteration = np.array([
        row["iteration"]
        for row in history
    ])

    rho = np.array([
        row["rho"]
        for row in history
    ])

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.step(
        iteration,
        rho,
        where="post",
        linewidth=2,
    )

    ax.set_xlabel("ADMM iteration")
    ax.set_ylabel("rho")
    ax.set_title("Adaptive ADMM penalty")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)


def save_data_heatmaps(
    clean: np.ndarray,
    noisy: np.ndarray,
    recovered: np.ndarray,
    output_path: Path,
) -> None:
    common_min = min(
        float(clean.min()),
        float(noisy.min()),
        float(recovered.min()),
    )

    common_max = max(
        float(clean.max()),
        float(noisy.max()),
        float(recovered.max()),
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13, 4),
        sharex=True,
        sharey=True,
    )

    data_sets = [
        ("Exact data", clean),
        ("Noisy data", noisy),
        ("Recovered data", recovered),
    ]

    image = None

    for ax, (title, data) in zip(
        axes,
        data_sets,
    ):
        image = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            vmin=common_min,
            vmax=common_max,
        )

        ax.set_title(title)
        ax.set_xlabel("Time frame")

    axes[0].set_ylabel(
        "Physical signal position"
    )

    fig.colorbar(
        image,
        ax=axes,
        shrink=0.85,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.90,
        bottom=0.15,
        top=0.88,
        wspace=0.12,
    )

    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)


def save_frame_profiles(
    clean: np.ndarray,
    noisy: np.ndarray,
    recovered: np.ndarray,
    output_path: Path,
) -> None:
    T = clean.shape[1]

    frames = [
        0,
        T // 2,
        T - 1,
    ]

    x = np.arange(
        clean.shape[0]
    )

    fig, axes = plt.subplots(
        len(frames),
        1,
        figsize=(9, 9),
        sharex=True,
    )

    for ax, frame in zip(
        axes,
        frames,
    ):
        ax.plot(
            x,
            clean[:, frame],
            label="Exact",
            linewidth=2,
        )

        ax.plot(
            x,
            noisy[:, frame],
            label="Noisy",
            alpha=0.65,
        )

        ax.plot(
            x,
            recovered[:, frame],
            label="Recovered",
            linewidth=2,
        )

        ax.set_ylabel("Amplitude")
        ax.set_title(
            f"Frame {frame}"
        )
        ax.grid(True)

    axes[-1].set_xlabel(
        "Physical signal position"
    )

    axes[0].legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)


def save_trajectory_plot(
    true_physical: list[int],
    true_scales: list[int],
    recovered: dict,
    scale_values: list[float],
    output_path: Path,
) -> None:
    time = np.arange(
        len(true_physical)
    )

    true_scale_values = [
        scale_values[j]
        for j in true_scales
    ]

    recovered_scale_values = [
        scale_values[j]
        for j in recovered["scale_indices"]
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
    )

    axes[0].plot(
        time,
        true_physical,
        "o-",
        label="True",
        linewidth=2,
    )

    axes[0].plot(
        time,
        recovered["physical_positions"],
        "s--",
        label="Recovered",
        linewidth=2,
    )

    axes[0].set_ylabel("Physical position")
    axes[0].set_title(
        "Recovered trajectory"
    )
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        time,
        true_scale_values,
        "o-",
        label="True",
        linewidth=2,
    )

    axes[1].plot(
        time,
        recovered_scale_values,
        "s--",
        label="Recovered",
        linewidth=2,
    )

    axes[1].set_xlabel("Time frame")
    axes[1].set_ylabel("Scale")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)


def run_experiment(
    config: ExperimentConfig,
    output_root: Path,
) -> dict:
    rng = np.random.default_rng(
        config.seed
    )

    if config.T != len(
        TRUE_PHYSICAL_POSITIONS
    ):
        raise ValueError(
            "Canonical experiment requires T=7"
        )

    D = make_dictionary(
        config.M,
        config.gaussian_width,
    )

    operator = MultiscaleConvolutionOperator(
        D=D,
        scales=SCALES,
        center=(config.M + 1) // 2,
        N=config.N,
        normalize=True,
    )

    X_true, true_coeff_positions = (
        make_true_coefficients(
            L=operator.L,
            M=config.M,
            J=len(SCALES),
            T=config.T,
        )
    )

    S_clean = operator.forward(
        X_true
    )

    (
        S_noisy,
        noise,
        achieved_snr_db,
    ) = add_noise_snr(
        S_clean,
        config.snr_db,
        rng,
    )

    X0 = operator.adjoint(
        S_noisy,
        structured=True,
    )

    max_abs = np.max(
        np.abs(X0)
    )

    if max_abs > 0:
        X0 /= max_abs

    (
        X_admm,
        Y_admm,
        U_admm,
        info,
    ) = solve_l1_admm(
        X0=X0,
        S=S_noisy,
        operator=operator,
        lambda1=config.lambda1,
        lambda2=config.lambda2,
        tau=config.tau,
        rho=config.rho,
        max_admm_iter=config.max_admm_iter,
        x_maxiter=config.x_maxiter,
        abs_tol=config.abs_tol,
        rel_tol=config.rel_tol,
        nonnegative=config.nonnegative,
        adaptive_rho=config.adaptive_rho,
        rho_mu=10.0,
        rho_tau=2.0,
        verbose=True,
    )

    X_recovered = Y_admm

    S_recovered = operator.forward(
        X_recovered
    )

    recovered_trajectory = (
        dominant_trajectory(
            X_recovered,
            config.M,
        )
    )

    scale_values = [
        p / q
        for p, q in SCALES
    ]

    #####
    output_snr_db = reconstruction_snr_db(
        S_clean,
        S_recovered,
    )

    snr_improvement_db = (
        output_snr_db
        - achieved_snr_db
    )

    position_matches = sum(
        recovered == truth
        for recovered, truth in zip(
         recovered_trajectory[
             "physical_positions"
            ],
            TRUE_PHYSICAL_POSITIONS,
        )
    )

    scale_matches = sum(
        recovered == truth
        for recovered, truth in zip(
            recovered_trajectory[
                "scale_indices"
            ],
            TRUE_SCALE_INDICES,
        )
    )

    joint_matches = sum(
        recovered_position == true_position
        and recovered_scale == true_scale
        for (
            recovered_position,
            recovered_scale,
            true_position,
            true_scale,
        ) in zip (
            recovered_trajectory[
                "physical_positions"
            ],
            recovered_trajectory[
                "scale_indices"
            ],
        TRUE_PHYSICAL_POSITIONS,
        TRUE_SCALE_INDICES,
        )
    )

    position_accuracy = (
        position_matches / config.T
    )

    scale_accuracy = (
    scale_matches / config.T
)

    trajectory_accuracy = (
        joint_matches / config.T
    )

    exact_trajectory = (
        joint_matches == config.T
    )

    metrics = {
        "requested_snr_db": float(config.snr_db),
        
        "achieved_input_snr_db": float(achieved_snr_db),
        
        "output_snr_db": float(output_snr_db),
        
        "snr_improvement_db": float(snr_improvement_db),
        
        "signal_error_clean": relative_error(S_recovered,S_clean,),                     
        
        "signal_error_noisy": relative_error(S_recovered,S_noisy,),
        
        "coefficient_error": relative_error(X_recovered,X_true,),

        "significant_coefficients": int(np.count_nonzero(np.abs(X_recovered) > 1e-3)),

        "exact_nonzero_coefficients":
            int(np.count_nonzero(X_recovered)),

        "converged":bool(info["converged"]),

        "admm_iterations":int(info["iterations"]),

        "final_rho": float(info["rho"]),

        "final_objective": float(info["final_objective"] ),

        "final_primal_residual": float(info["final_primal_residual"]),

        "final_dual_residual": float(info["final_dual_residual"]),

        "position_matches": int(position_matches),

        "scale_matches": int(scale_matches),

        "joint_trajectory_matches": int(joint_matches),

        "position_accuracy": float(position_accuracy),

        "scale_accuracy": float(scale_accuracy),

        "trajectory_accuracy": float(trajectory_accuracy),

        "exact_trajectory": bool(exact_trajectory),
    }

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        output_root
        / f"gaussian_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    metadata = {
        "config": asdict(config),
        "dimensions": {
            "M": config.M,
            "N": config.N,
            "L": operator.L,
            "K": operator.K,
            "J": operator.J,
            "T": config.T,
        },
        "scales": [
            {
                "numerator": p,
                "denominator": q,
                "value": p / q,
            }
            for p, q in SCALES
        ],
        "truth": {
            "physical_positions":
                TRUE_PHYSICAL_POSITIONS,
            "coefficient_positions":
                true_coeff_positions,
            "scale_indices":
                TRUE_SCALE_INDICES,
            "scale_values": [
                scale_values[j]
                for j in TRUE_SCALE_INDICES
            ],
        },
        "recovered_trajectory":
            recovered_trajectory,
        "metrics":
            metrics,
    }

    with (
        output_dir / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            default=json_default,
        )

    np.savez_compressed(
        output_dir / "arrays.npz",
        dictionary=D,
        scaled_dictionary=
            operator.D_scaled,
        X_true=X_true,
        X_initial=X0,
        X_smooth=X_admm,
        X_recovered=X_recovered,
        U=U_admm,
        S_clean=S_clean,
        S_noisy=S_noisy,
        S_recovered=S_recovered,
        noise=noise,
    )

    save_cost_plot(
        info,
        output_dir / "cost_history.png",
    )

    save_residual_plot(
        info,
        output_dir / "admm_residuals.png",
    )

    save_rho_plot(
        info,
        output_dir / "rho_history.png",
    )

    save_data_heatmaps(
        S_clean,
        S_noisy,
        S_recovered,
        output_dir / "data_comparison.png",
    )

    save_frame_profiles(
        S_clean,
        S_noisy,
        S_recovered,
        output_dir / "frame_profiles.png",
    )

    save_trajectory_plot(
        TRUE_PHYSICAL_POSITIONS,
        TRUE_SCALE_INDICES,
        recovered_trajectory,
        scale_values,
        output_dir / "trajectory.png",
    )

    print()
    print("Canonical Gaussian experiment")
    print("-----------------------------")

    print(
        f"M={config.M}, "
        f"N={config.N}, "
        f"L={operator.L}, "
        f"K={operator.K}, "
        f"J={operator.J}, "
        f"T={config.T}"
    )

    print(f"requested SNR: "f"{config.snr_db:.3f} dB")

    print(f"input SNR:     "f"{achieved_snr_db:.3f} dB")

    print(f"output SNR:    "f"{output_snr_db:.3f} dB")

    print(f"SNR gain:      "f"{snr_improvement_db:+.3f} dB")

    print(f"converged:     "f"{metrics['converged']}")

    print(f"iterations:    "f"{metrics['admm_iterations']}")

    print(f"signal error:  "f"{metrics['signal_error_clean']:.6e}")

    print(F"coeff. error:  "f"{metrics['coefficient_error']:.6e}")

    print(f"position acc.: "f"{100.0 * position_accuracy:.1f}% "f"({position_matches}/{config.T})")

    print(f"scale acc.:    "f"{100.0 * scale_accuracy:.1f}% "f"({scale_matches}/{config.T})")

    print(f"trajectory acc.: "f"{100.0 * trajectory_accuracy:.1f}% "f"({joint_matches}/{config.T})")

    print(f"exact path:    "f"{exact_trajectory}")

    return {
        "output_dir": output_dir,
        "metrics": metrics,
        "info": info,
        "metadata": metadata,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical Gaussian multiscale "
            "softmin/L1-ADMM experiment"
        )
    )

    parser.add_argument(
        "--snr-db",
        type=float,
        default=20.0,
        help="Requested Gaussian-noise SNR in dB",
    )

    parser.add_argument(
        "--lambda1",
        type=float,
        default=0.05,
        help="L1 sparsity weight",
    )

    parser.add_argument(
        "--lambda2",
        type=float,
        default=0.10,
        help="Softmin regularization weight",
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=1e-2,
        help="Softmin temperature",
    )

    parser.add_argument(
        "--rho",
        type=float,
        default=0.10,
        help="Initial ADMM penalty",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Random seed",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Root output directory",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ExperimentConfig(
        seed=args.seed,
        snr_db=args.snr_db,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        tau=args.tau,
        rho=args.rho,
    )

    run_experiment(
        config,
        args.output,
    )


if __name__ == "__main__":
    main()
