import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaussian_experiment import ExperimentConfig, run_experiment


LAMBDA1_VALUES = [
    0.001, 0.002, 0.005,
    0.010, 0.020, 0.030,
    0.050, 0.075, 0.100,
    0.150, 0.200,
]

SNR_DB = 14.0
OUTPUT_DIR = Path("results/lambda1_sweep")


def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def plot_metrics(results):
    lam = np.array([r["lambda1"] for r in results])

    fig, ax = plt.subplots()
    ax.semilogx(lam, [r["output_snr_db"] for r in results], "o-")
    ax.set(xlabel="lambda1", ylabel="Output SNR (dB)",
           title=f"Reconstruction quality at {SNR_DB:.0f} dB input SNR")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "output_snr_vs_lambda1.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.semilogx(lam, [r["significant_coefficients"] for r in results], "o-")
    ax.set(xlabel="lambda1", ylabel="Significant coefficients",
           title="Sparsity vs lambda1")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sparsity_vs_lambda1.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.semilogx(lam, 100*np.array([r["position_accuracy"] for r in results]),
                "o-", label="Position")
    ax.semilogx(lam, 100*np.array([r["scale_accuracy"] for r in results]),
                "s-", label="Scale")
    ax.semilogx(lam, 100*np.array([r["trajectory_accuracy"] for r in results]),
                "^-", label="Joint")
    ax.set(xlabel="lambda1", ylabel="Accuracy (%)",
           title="Trajectory accuracy vs lambda1", ylim=(-5, 105))
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_vs_lambda1.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.semilogx(lam, [r["signal_error_clean"] for r in results],
                "o-", label="Signal")
    ax.semilogx(lam, [r["coefficient_error"] for r in results],
                "s-", label="Coefficients")
    ax.set(xlabel="lambda1", ylabel="Relative error",
           title="Recovery error vs lambda1")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_vs_lambda1.png", dpi=180)
    plt.close(fig)


def choose_lambda1(results):
    """
    Selection rule:
      1. solver must converge;
      2. joint trajectory accuracy must be 100%;
      3. among qualifying values, choose the smallest lambda1.

    This finds the onset of reliable trajectory identification,
    rather than the sparsest possible fit.
    """
    valid = [
        r for r in results
        if r["converged"] and r["trajectory_accuracy"] == 1.0
    ]

    if not valid:
        return None

    return min(valid, key=lambda r: r["lambda1"])


def run_sweep():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for lambda1 in LAMBDA1_VALUES:
        print(f"\n{'='*64}")
        print(f"lambda1 = {lambda1:.4g}, SNR = {SNR_DB:.1f} dB")
        print("="*64)

        config = ExperimentConfig(
            seed=100,
            snr_db=SNR_DB,
            lambda1=lambda1,
            lambda2=0.10,
            tau=1e-2,
            rho=0.10,
        )

        run = run_experiment(config, OUTPUT_DIR)
        m = run["metrics"]

        results.append({
            "lambda1": lambda1,
            "converged": m["converged"],
            "output_snr_db": m["output_snr_db"],
            "snr_improvement_db": m["snr_improvement_db"],
            "signal_error_clean": m["signal_error_clean"],
            "coefficient_error": m["coefficient_error"],
            "significant_coefficients": m["significant_coefficients"],
            "position_accuracy": m["position_accuracy"],
            "scale_accuracy": m["scale_accuracy"],
            "trajectory_accuracy": m["trajectory_accuracy"],
            "exact_trajectory": m["exact_trajectory"],
            "admm_iterations": m["admm_iterations"],
            "final_rho": m["final_rho"],
            "run_directory": str(run["output_dir"]),
        })

    selected = choose_lambda1(results)

    summary = {
        "snr_db": SNR_DB,
        "lambda2": 0.10,
        "tau": 1e-2,
        "selection_rule": (
            "Smallest converged lambda1 with 100% joint trajectory accuracy"
        ),
        "selected_lambda1": None if selected is None else selected["lambda1"],
        "selected_result": selected,
        "results": results,
    }

    with (OUTPUT_DIR / "lambda1_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)

    plot_metrics(results)

    print("\nLambda1 sweep summary")
    print("---------------------")
    print(" lambda1  outSNR  sig.err  coef.err  nz   pos scale joint iter")

    for r in results:
        print(
            f"{r['lambda1']:8.3g}  "
            f"{r['output_snr_db']:6.2f}  "
            f"{r['signal_error_clean']:7.4f}  "
            f"{r['coefficient_error']:8.4f}  "
            f"{r['significant_coefficients']:3d}  "
            f"{100*r['position_accuracy']:3.0f}% "
            f"{100*r['scale_accuracy']:4.0f}% "
            f"{100*r['trajectory_accuracy']:4.0f}% "
            f"{r['admm_iterations']:4d}"
        )

    print()
    if selected is None:
        print("No lambda1 achieved exact trajectory recovery.")
    else:
        print(f"Selected lambda1: {selected['lambda1']:.4g}")
        print(f"Output SNR:       {selected['output_snr_db']:.2f} dB")
        print(f"Signal error:     {selected['signal_error_clean']:.4f}")
        print(f"Significant coeffs: {selected['significant_coefficients']}")


if __name__ == "__main__":
    run_sweep()
    