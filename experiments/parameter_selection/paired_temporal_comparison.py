"""Paired temporal vs non-temporal comparison using completed MC trials.

Run from project root:
  python -m experiments.parameter_selection.paired_temporal_comparison

The non-temporal baseline reuses Stage-A candidates (lambda_t=beta=0) from
exactly the same noise realizations. No new inverse solves are required.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

from gaussian_experiment import SCALES, make_dictionary, relative_error, reconstruction_snr_db
from multiscale_convolution import MultiscaleConvolutionOperator

ROOT = Path("results/barycentric_monte_carlo")
OUT = ROOT / "paired_temporal_comparison.json"
SNR_VALUES = [20, 17, 14, 11, 8]
N_TRIALS = 10
DISC_LO, DISC_HI = 0.95, 1.05
COEFF_THRESHOLD = 1e-3
ACTIVITY_THRESHOLD = 0.05
M, N, T = 41, 96, 12
ACTIVE_FRAMES = np.arange(2, 10)
TRUE_POSITIONS = np.array([28, 29, 30, 31, 32, 33, 34, 35])
TRUE_SCALES = np.array([0, 1, 2, 3, 3, 2, 1, 0])


def trial_dir(snr, trial): return ROOT / f"snr_{snr:g}" / f"trial_{trial:02d}"
def summary_path(snr, trial): return trial_dir(snr, trial) / "adaptive_selector_summary.json"


def make_truth(A):
    X = np.zeros((A.L, A.K, A.J, T))
    for t, pos, j in zip(ACTIVE_FRAMES, TRUE_POSITIONS, TRUE_SCALES):
        X[int(pos + M - 1), 0, int(j), int(t)] = 1.0
    return X


def signature(r):
    return tuple(r["activity"]), tuple(r["positions"]), tuple(r["scales"]), int(r["support"])


def feasible(rows):
    return [r for r in rows if r["converged"] and DISC_LO <= r["discrepancy_expected"] <= DISC_HI]


def curvature_select(rows):
    rows = sorted(rows, key=lambda r: r["lambda_s"])
    if len(rows) < 3: return min(rows, key=lambda r: abs(np.log(r["discrepancy_expected"])))
    pts = np.array([[np.log(max(r["data"], 1e-300)), np.log(max(r["log_penalty_raw"], 1e-300))] for r in rows])
    kappa = np.full(len(rows), -np.inf)
    for i in range(1, len(rows)-1):
        a, b, c = pts[i-1], pts[i], pts[i+1]
        u, v = b-a, c-a
        area2 = abs(u[0]*v[1] - u[1]*v[0])
        den = np.linalg.norm(u)*np.linalg.norm(c-b)*np.linalg.norm(c-a)
        kappa[i] = 2*area2/den if den > 0 else 0.0
    return rows[int(np.argmax(kappa))]


def stability_select(rows):
    rows = sorted(feasible(rows) or [r for r in rows if r["converged"]], key=lambda r: r["lambda_s"])
    runs, run = [], [rows[0]]
    for r in rows[1:]:
        if signature(r) == signature(run[-1]): run.append(r)
        else: runs.append(run); run = [r]
    runs.append(run)
    runs.sort(key=lambda q: (-len(q), np.mean([abs(np.log(r["discrepancy_expected"])) for r in q]), q[0]["support"]))
    run = runs[0]
    center = 0.5*(run[0]["lambda_s"] + run[-1]["lambda_s"])
    return min(run, key=lambda r: (abs(r["lambda_s"]-center), abs(np.log(r["discrepancy_expected"]))))


def cache_path(tdir, row):
    return tdir / "cache" / f"A_ls_{row['lambda_s']:.6f}_lt_0.00000000_beta_0.0000.npz"


def baseline_metrics(tdir, row, A, Xtrue, Sclean):
    path = cache_path(tdir, row)
    if not path.exists(): raise FileNotFoundError(path)
    z = np.load(path)
    Y = z["Y"]
    Shat = A.forward(Y)
    masses = np.sum(Y, axis=(0, 1, 2))
    activity = masses > ACTIVITY_THRESHOLD
    pos, scale = [], []
    for t in ACTIVE_FRAMES:
        frame = np.abs(Y[:, :, :, int(t)])
        n, _, j = np.unravel_index(np.argmax(frame), frame.shape)
        pos.append(int(n-(M-1))); scale.append(int(j))
    pos, scale = np.array(pos), np.array(scale)
    true_activity = np.zeros(T, bool); true_activity[ACTIVE_FRAMES] = True
    return {
        "lambda_s": row["lambda_s"], "support": int(np.count_nonzero(np.abs(Y) > COEFF_THRESHOLD)),
        "discrepancy": row["discrepancy_expected"], "xerr": float(relative_error(Y, Xtrue)),
        "output_snr": float(reconstruction_snr_db(Sclean, Shat)),
        "activity_exact": bool(np.array_equal(activity, true_activity)),
        "position_accuracy": float(np.mean(pos == TRUE_POSITIONS)),
        "scale_accuracy": float(np.mean(scale == TRUE_SCALES)),
        "joint_accuracy": float(np.mean((pos == TRUE_POSITIONS) & (scale == TRUE_SCALES))),
    }


def temporal_metrics(summary, selector):
    selected = summary[f"{selector}_selection"]
    p = summary["posthoc_validation"][f"{selector}_selection"]
    return {
        "lambda_s": selected["lambda_s"], "lambda_t": selected["lambda_t"], "beta_v": selected["beta_v"],
        "support": selected["support"], "discrepancy": selected["discrepancy_expected"],
        "xerr": p["coefficient_error"], "output_snr": p["output_snr_db"],
        "activity_exact": selected["activity"] == [0,0,1,1,1,1,1,1,1,1,0,0],
        "position_accuracy": p["position_accuracy"], "scale_accuracy": p["scale_accuracy"],
        "joint_accuracy": p["joint_accuracy"],
    }

def baseline_oracle(tdir, stage_a, A, Xtrue, Sclean):
    candidates = [baseline_metrics(tdir, r, A, Xtrue, Sclean) for r in stage_a]
    return min(candidates, key=lambda r: r["xerr"])


def temporal_oracle(summary):
    p = summary["posthoc_validation"]
    pars, m = p["oracle_parameters"], p["oracle_metrics"]

    matches = [
        r for r in summary["candidates"]
        if r["lambda_s"] == pars["lambda_s"]
        and r["lambda_t"] == pars["lambda_t"]
        and r["beta_v"] == pars["beta_v"]
    ]
    if not matches:
        raise RuntimeError(f"Could not locate temporal oracle candidate: {pars}")

    r = matches[0]
    return {
        "lambda_s": pars["lambda_s"],
        "lambda_t": pars["lambda_t"],
        "beta_v": pars["beta_v"],
        "support": r["support"],
        "discrepancy": r["discrepancy_expected"],
        "xerr": m["coefficient_error"],
        "output_snr": m["output_snr_db"],
        "activity_exact": r["activity"] == [0,0,1,1,1,1,1,1,1,1,0,0],
        "position_accuracy": m["position_accuracy"],
        "scale_accuracy": m["scale_accuracy"],
        "joint_accuracy": m["joint_accuracy"],
    }


def oracle_stats(base, temp):
    eb = np.array([r["xerr"] for r in base])
    et = np.array([r["xerr"] for r in temp])
    sb = np.array([r["scale_accuracy"] for r in base])
    st = np.array([r["scale_accuracy"] for r in temp])

    tab = exact_table(base, temp)
    discordant = tab["temporal_fixes"] + tab["temporal_breaks"]

    try:
        wp = float(wilcoxon(eb, et, alternative="greater").pvalue)
    except ValueError:
        wp = 1.0

    mp = (
        float(binomtest(
            tab["temporal_fixes"], discordant, 0.5,
            alternative="greater"
        ).pvalue)
        if discordant else 1.0
    )

    return {
        "baseline_xerr_median": float(np.median(eb)),
        "temporal_xerr_median": float(np.median(et)),
        "median_xerr_improvement": float(np.median(eb-et)),
        "temporal_better_xerr_rate": float(np.mean(et < eb)),
        "mean_scale_accuracy_improvement": float(np.mean(st-sb)),
        "baseline_exact_rate": float(np.mean([r["joint_accuracy"] == 1 for r in base])),
        "temporal_exact_rate": float(np.mean([r["joint_accuracy"] == 1 for r in temp])),
        "baseline_exact8_rate": float(np.mean([r["support"] == 8 for r in base])),
        "temporal_exact8_rate": float(np.mean([r["support"] == 8 for r in temp])),
        "exact_transition_table": tab,
        "wilcoxon_one_sided_p": wp,
        "mcnemar_exact_one_sided_p": mp,
    }


def exact_table(base, temp):
    bf = np.array([r["joint_accuracy"] == 1 for r in base]); tf = np.array([r["joint_accuracy"] == 1 for r in temp])
    return {"both_fail": int(np.sum(~bf & ~tf)), "temporal_fixes": int(np.sum(~bf & tf)),
            "temporal_breaks": int(np.sum(bf & ~tf)), "both_exact": int(np.sum(bf & tf))}


def paired_stats(base, temp):
    eb = np.array([r["xerr"] for r in base]); et = np.array([r["xerr"] for r in temp])
    ds = np.array([t["scale_accuracy"]-b["scale_accuracy"] for b, t in zip(base, temp)])
    tab = exact_table(base, temp); discordant = tab["temporal_fixes"] + tab["temporal_breaks"]
    try: wp = float(wilcoxon(eb, et, alternative="greater").pvalue)
    except ValueError: wp = 1.0
    mp = float(binomtest(tab["temporal_fixes"], discordant, 0.5, alternative="greater").pvalue) if discordant else 1.0
    return {
        "baseline_xerr_median": float(np.median(eb)), "temporal_xerr_median": float(np.median(et)),
        "median_xerr_improvement": float(np.median(eb-et)), "temporal_better_xerr_rate": float(np.mean(et < eb)),
        "mean_scale_accuracy_improvement": float(np.mean(ds)),
        "baseline_exact_rate": float(np.mean([r["joint_accuracy"] == 1 for r in base])),
        "temporal_exact_rate": float(np.mean([r["joint_accuracy"] == 1 for r in temp])),
        "baseline_exact8_rate": float(np.mean([r["support"] == 8 for r in base])),
        "temporal_exact8_rate": float(np.mean([r["support"] == 8 for r in temp])),
        "exact_transition_table": tab, "wilcoxon_one_sided_p": wp, "mcnemar_exact_one_sided_p": mp,
    }


def main():
    D = make_dictionary(M, 4.0)
    A = MultiscaleConvolutionOperator(D=D, scales=SCALES, center=(M+1)//2, N=N, normalize=True)
    Xtrue = make_truth(A); Sclean = A.forward(Xtrue)
    output = {"snr_values": SNR_VALUES, "n_trials": N_TRIALS, "results": []}

    for snr in SNR_VALUES:
        bs, bl, ts, tl = [], [], [], []
        bo, to, trials = [], [], []
        for trial in range(N_TRIALS):
            tdir = trial_dir(snr, trial)
            with summary_path(snr, trial).open(encoding="utf-8") as f: summary = json.load(f)
            stage_a = [r for r in summary["candidates"] if r["stage"] == "A"]
            bom = baseline_oracle(tdir, stage_a, A, Xtrue, Sclean)
            tom = temporal_oracle(summary)
            bo.append(bom)
            to.append(tom)
            bsr = stability_select(stage_a); blr = curvature_select(stage_a)
            bsm = baseline_metrics(tdir, bsr, A, Xtrue, Sclean); blm = baseline_metrics(tdir, blr, A, Xtrue, Sclean)
            tsm = temporal_metrics(summary, "stability"); tlm = temporal_metrics(summary, "lcurve")
            bs.append(bsm); bl.append(blm); ts.append(tsm); tl.append(tlm)
            trials.append({"trial": trial, "baseline_stability": bsm, "baseline_lcurve": blm,
                "temporal_stability": tsm, "temporal_lcurve": tlm, "baseline_oracle": bom,  "temporal_oracle": tom,})
        result = {"snr": snr, "stability": paired_stats(bs, ts), "lcurve": paired_stats(bl, tl), 
                  "oracle": oracle_stats(bo, to), "trials": trials}
        output["results"].append(result)
        for name in ("stability", "lcurve"):
            r = result["oracle"]
            print(
                f"SNR={snr:2d} {'oracle':9s} "
                f"Xerr {r['baseline_xerr_median']:.4f}->{r['temporal_xerr_median']:.4f} "
                f"exact {100*r['baseline_exact_rate']:.0f}%->{100*r['temporal_exact_rate']:.0f}% "
                f"fix/break={r['exact_transition_table']['temporal_fixes']}/"
                f"{r['exact_transition_table']['temporal_breaks']}"
            )

    with OUT.open("w", encoding="utf-8") as f: json.dump(output, f, indent=2)
    print("Saved:", OUT)


if __name__ == "__main__": main()
