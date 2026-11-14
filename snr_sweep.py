import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaussian_experiment import ExperimentConfig, run_experiment


SNR_VALUES = [20.0, 17.0, 14.0, 11.0, 8.0]
OUTPUT_DIR = Path("results/snr_sweep")


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


def plot_snr(results):
    x = np.array([r["input_snr_db"] for r in results])
    y = np.array([r["output_snr_db"] for r in results])

    fig, ax = plt.subplots()
    ax.plot(x, y, "o-", label="Recovered")
    ax.plot(x, x, "--", label="No improvement")
    ax.set(xlabel="Input SNR (dB)", ylabel="Output SNR (dB)",
           title="Reconstruction SNR")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "snr_input_output.png", dpi=180)
    plt.close(fig)


def plot_gain(results):
    x = np.array([r["input_snr_db"] for r in results])
    y = np.array([r["snr_improvement_db"] for r in results])

    fig, ax = plt.subplots()
    ax.plot(x, y, "o-")
    ax.axhline(0.0, linestyle="--")
    ax.set(xlabel="Input SNR (dB)", ylabel="SNR improvement (dB)",
           title="SNR improvement")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "snr_gain.png", dpi=180)
    plt.close(fig)


def plot_accuracy(results):
    x = np.array([r["input_snr_db"] for r in results])

    fig, ax = plt.subplots()
    ax.plot(x, 100*np.array([r["position_accuracy"] for r in results]),
            "o-", label="Position")
    ax.plot(x, 100*np.array([r["scale_accuracy"] for r in results]),
            "s-", label="Scale")
    ax.plot(x, 100*np.array([r["trajectory_accuracy"] for r in results]),
            "^-", label="Joint trajectory")

    ax.set(xlabel="Input SNR (dB)", ylabel="Accuracy (%)",
           title="Trajectory accuracy vs SNR", ylim=(-5, 105))
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "trajectory_accuracy.png", dpi=180)
    plt.close(fig)


def plot_errors(results):
    x = np.array([r["input_snr_db"] for r in results])

    fig, ax = plt.subplots()
    ax.plot(x, [r["signal_error_clean"] for r in results],
            "o-", label="Signal")
    ax.plot(x, [r["coefficient_error"] for r in results],
            "s-", label="Coefficients")

    ax.set(xlabel="Input SNR (dB)", ylabel="Relative error",
           title="Recovery error vs SNR")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "recovery_errors.png", dpi=180)
    plt.close(fig)


def run_sweep():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for snr_db in SNR_VALUES:
        print(f"\n{'='*60}\nSNR = {snr_db:.1f} dB\n{'='*60}")

        config = ExperimentConfig(
            seed=100,
            snr_db=snr_db,
            lambda1=0.05,
            lambda2=0.10,
            tau=1e-2,
            rho=0.10,
        )

        run = run_experiment(config, OUTPUT_DIR)
        m = run["metrics"]

        results.append({
            "requested_snr_db": snr_db,
            "input_snr_db": m["achieved_input_snr_db"],
            "output_snr_db": m["output_snr_db"],
            "snr_improvement_db": m["snr_improvement_db"],
            "signal_error_clean": m["signal_error_clean"],
            "coefficient_error": m["coefficient_error"],
            "position_accuracy": m["position_accuracy"],
            "scale_accuracy": m["scale_accuracy"],
            "trajectory_accuracy": m["trajectory_accuracy"],
            "exact_trajectory": m["exact_trajectory"],
            "significant_coefficients": m["significant_coefficients"],
            "admm_iterations": m["admm_iterations"],
            "final_rho": m["final_rho"],
            "run_directory": str(run["output_dir"]),
        })

    with (OUTPUT_DIR / "sweep_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=json_default)

    plot_snr(results)
    plot_gain(results)
    plot_accuracy(results)
    plot_errors(results)

    print("\nSNR sweep summary")
    print("-----------------")
    print(" input  output   gain   sig.err  coef.err  pos  scale joint iter")

    for r in results:
        print(
            f"{r['input_snr_db']:5.1f}  "
            f"{r['output_snr_db']:6.2f}  "
            f"{r['snr_improvement_db']:+6.2f}  "
            f"{r['signal_error_clean']:7.4f}  "
            f"{r['coefficient_error']:8.4f}  "
            f"{100*r['position_accuracy']:3.0f}%  "
            f"{100*r['scale_accuracy']:4.0f}%  "
            f"{100*r['trajectory_accuracy']:4.0f}%  "
            f"{r['admm_iterations']:4d}"
        )


if __name__ == "__main__":
    run_sweep()
