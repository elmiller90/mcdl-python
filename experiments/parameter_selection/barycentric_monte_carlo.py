import json, subprocess, sys, time
from pathlib import Path
import numpy as np

SNR_VALUES = [20, 17, 14, 11, 8]
N_TRIALS = 10
BASE_SEED = 1000

ROOT = Path("results/barycentric_monte_carlo")
MODULE = "experiments.parameter_selection.adaptive_barycentric_selector"


def trial_seed(i_snr, trial):
    return BASE_SEED + 1000*i_snr + trial


def trial_dir(snr, trial):
    return ROOT / f"snr_{snr:g}" / f"trial_{trial:02d}"


def summary_path(snr, trial):
    return trial_dir(snr, trial) / "adaptive_selector_summary.json"


def run_trial(snr, i_snr, trial):
    out = trial_dir(snr, trial)
    summary = summary_path(snr, trial)

    if summary.exists():
        print(f"SNR={snr:2g} trial={trial:02d}: cached")
        return

    seed = trial_seed(i_snr, trial)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", MODULE,
        "--snr", str(snr),
        "--seed", str(seed),
        "--out", str(out),
    ]

    print(
        f"SNR={snr:2g} trial={trial:02d} "
        f"seed={seed}: running"
    )

    start = time.perf_counter()

    with (out / "run.log").open("w") as log:
        subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    print(
        f"  completed in "
        f"{time.perf_counter()-start:.1f}s"
    )


def posthoc(row):
    return row["posthoc_validation"]


def selected_metrics(summary, selector):
    p = posthoc(summary)

    if selector == "stability":
        selected = summary["stability_selection"]
        metrics = p["stability_selection"]
    elif selector == "lcurve":
        selected = summary["lcurve_selection"]
        metrics = p["lcurve_selection"]
    else:
        raise ValueError(selector)

    return {
        "lambda_s": selected["lambda_s"],
        "lambda_t": selected["lambda_t"],
        "beta_v": selected["beta_v"],
        "support": selected["support"],
        "discrepancy":
            selected["discrepancy_expected"],
        "xerr": metrics["coefficient_error"],
        "output_snr": metrics["output_snr_db"],
        "position_accuracy":
            metrics["position_accuracy"],
        "scale_accuracy":
            metrics["scale_accuracy"],
        "joint_accuracy":
            metrics["joint_accuracy"],
    }


def oracle_metrics(summary):
    p = posthoc(summary)

    return {
        **p["oracle_parameters"],
        **p["oracle_metrics"],
    }


def aggregate_selector(rows):
    def a(name):
        return np.array(
            [r[name] for r in rows],
            dtype=float,
        )

    xerr = a("xerr")
    joint = a("joint_accuracy")
    position = a("position_accuracy")
    scale = a("scale_accuracy")
    support = a("support")
    disc = a("discrepancy")

    return {
        "n": len(rows),
        "xerr_median": float(np.median(xerr)),
        "xerr_q25": float(np.quantile(xerr, .25)),
        "xerr_q75": float(np.quantile(xerr, .75)),
        "position_mean": float(position.mean()),
        "scale_mean": float(scale.mean()),
        "joint_mean": float(joint.mean()),
        "exact_trajectory_rate":
            float(np.mean(joint == 1.0)),
        "exact_8_support_rate":
            float(np.mean(support == 8)),
        "discrepancy_mean": float(disc.mean()),
        "discrepancy_std": float(disc.std(ddof=1))
            if len(rows) > 1 else 0.0,
        "selected_lambda_s":
            [r["lambda_s"] for r in rows],
        "selected_lambda_t":
            [r["lambda_t"] for r in rows],
        "selected_beta_v":
            [r["beta_v"] for r in rows],
    }


def aggregate_snr(snr, i_snr):
    stability, lcurve, oracle = [], [], []
    agreement = 0

    for trial in range(N_TRIALS):
        path = summary_path(snr, trial)

        with path.open(encoding="utf-8") as f:
            summary = json.load(f)

        s = selected_metrics(summary, "stability")
        l = selected_metrics(summary, "lcurve")
        o = oracle_metrics(summary)

        stability.append(s)
        lcurve.append(l)
        oracle.append(o)

        agreement += int(
            s["lambda_s"] == l["lambda_s"]
            and s["lambda_t"] == l["lambda_t"]
            and s["beta_v"] == l["beta_v"]
        )

    oracle_xerr = np.array([
        r["coefficient_error"]
        for r in oracle
    ])

    oracle_joint = np.array([
        r["joint_accuracy"] for r in oracle
    ])

    oracle_exact_rate = float(
        np.mean(oracle_joint == 1.0)
    )

    stab_ratio = np.array([
        r["xerr"] for r in stability
    ]) / oracle_xerr

    lcurve_ratio = np.array([
        r["xerr"] for r in lcurve
    ]) / oracle_xerr

    return {
        "snr": snr,
        "stability":
            aggregate_selector(stability),
        "lcurve":
            aggregate_selector(lcurve),
        "selector_agreement_rate":
            agreement / N_TRIALS,
        "stability_oracle_ratio_median":
            float(np.median(stab_ratio)),
        "lcurve_oracle_ratio_median":
            float(np.median(lcurve_ratio)),
            "oracle_exact_trajectory_rate": oracle_exact_rate,
        "trials": [
            {
                "trial": t,
                "seed": trial_seed(i_snr, t),
                "stability": stability[t],
                "lcurve": lcurve[t],
                "oracle": oracle[t],
            }
            for t in range(N_TRIALS)
        ],
    }


def print_summary(results):
    print()
    print("Monte Carlo summary")
    print("-------------------")
    print(
        "SNR selector  Xerr_med   IQR        "
        "pos   scale joint exact8 agree oracle-ratio"
    )

    for r in results:
        for name in ["stability", "lcurve"]:
            s = r[name]

            print(
                f"{r['snr']:3g} "
                f"{name:9s} "
                f"{s['xerr_median']:8.4f} "
                f"[{s['xerr_q25']:.4f},{s['xerr_q75']:.4f}] "
                f"{100*s['position_mean']:5.1f}% "
                f"{100*s['scale_mean']:5.1f}% "
                f"{100*s['exact_trajectory_rate']:5.1f}% "
                f"{100*s['exact_8_support_rate']:5.1f}% "
                f"{100*r['selector_agreement_rate']:5.1f}% "
                f"{r[name+'_oracle_ratio_median']:.2f}"
            )


def main():
    ROOT.mkdir(parents=True, exist_ok=True)

    total = len(SNR_VALUES) * N_TRIALS
    done = 0

    for i_snr, snr in enumerate(SNR_VALUES):
        for trial in range(N_TRIALS):
            run_trial(snr, i_snr, trial)
            done += 1
            print(f"progress: {done}/{total}")

    results = [
        aggregate_snr(snr, i)
        for i, snr in enumerate(SNR_VALUES)
    ]

    output = {
        "snr_values": SNR_VALUES,
        "n_trials": N_TRIALS,
        "base_seed": BASE_SEED,
        "results": results,
    }

    path = ROOT / "monte_carlo_summary.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print_summary(results)
    print()
    print("Saved:", path)


if __name__ == "__main__":
    main()