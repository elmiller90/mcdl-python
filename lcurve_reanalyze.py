import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaussian_experiment import (
    SCALES, add_noise_snr, make_dictionary,
    make_true_coefficients, relative_error,
)
from multiscale_convolution import MultiscaleConvolutionOperator


LAMBDAS = np.array([
    .005, .0075, .01, .015, .02, .03, .04, .05, .06,
    .075, .09, .105, .12, .135, .15, .175, .20
])

SNR_DB, SEED = 14.0, 100
ORACLE = .135

ROOT = Path("results/lambda_s_lcurve")
CACHE = ROOT / "cache"
OUT = ROOT / "reanalyze"

CACHE_VERSION = "v2_warm_desc_snr14_lt0_tau1e-2_seed100"


def cache_path(lam):
    return CACHE / f"{CACHE_VERSION}_lambda_{lam:.6f}.npz"


def load_cache(lam):
    p = cache_path(lam)
    if not p.exists():
        return None

    z = np.load(p)
    return {
        "Y": z["Y"],
        "converged": bool(z["converged"].item()),
        "iterations": int(z["iterations"].item()),
        "a": float(z["a"].item()),
    }


def data_fidelity(Y, S, A):
    r = A.forward(Y) - S
    return float(.5*np.sum(r*r))


def measures(Y, a):
    y = np.abs(Y)
    return {
        "changing_a": float(np.sum(np.log1p(a*y))/a),
        "l1": float(np.sum(y)),
        "log_a1": float(np.sum(np.log1p(y))),
        "log_a10": float(np.sum(np.log1p(10*y))),
        "active": int(np.count_nonzero(y > 1e-3)),
    }


def normalize_log(v):
    v = np.log10(np.asarray(v, dtype=float))
    lo, hi = v.min(), v.max()
    return np.zeros_like(v) if hi == lo else (v-lo)/(hi-lo)


def smooth3(x):
    y = np.array(x, dtype=float, copy=True)
    if len(y) >= 3:
        y[1:-1] = (x[:-2] + x[1:-1] + x[2:])/3
    return y


def curvature(x, y):
    k = np.full(len(x), np.nan)

    for i in range(1, len(x)-1):
        xp, yp = (x[i+1]-x[i-1])/2, (y[i+1]-y[i-1])/2
        xpp = x[i+1] - 2*x[i] + x[i-1]
        ypp = y[i+1] - 2*y[i] + y[i-1]
        den = (xp*xp + yp*yp)**1.5
        if den > 0:
            k[i] = abs(xp*ypp - yp*xpp)/den

    return k


def segments(rows):
    idx = {round(float(v), 10): i for i, v in enumerate(LAMBDAS)}
    rows = sorted(rows, key=lambda r: r["lambda_s"])

    out, current, prev = [], [], None

    for r in rows:
        i = idx[round(r["lambda_s"], 10)]

        if prev is None or i == prev + 1:
            current.append(r)
        else:
            if current:
                out.append(current)
            current = [r]

        prev = i

    if current:
        out.append(current)

    return out


def analyze(rows, key):
    points = []

    for seg_id, seg in enumerate(segments(rows)):
        if len(seg) < 3:
            continue

        x = normalize_log([r["data"] for r in seg])
        y = normalize_log([r[key] for r in seg])
        xs, ys = smooth3(x), smooth3(y)
        k = curvature(xs, ys)

        for i, r in enumerate(seg):
            points.append({
                "lambda_s": r["lambda_s"],
                "segment": seg_id,
                "xs": float(xs[i]),
                "ys": float(ys[i]),
                "curvature": None if not np.isfinite(k[i]) else float(k[i]),
            })

    finite = [p for p in points if p["curvature"] is not None]
    selected = max(finite, key=lambda p: p["curvature"]) if finite else None
    return selected, points


def plot_curve(name, selected, points):
    fig, ax = plt.subplots(figsize=(7, 6))

    for s in sorted(set(p["segment"] for p in points)):
        q = [p for p in points if p["segment"] == s]
        ax.plot([p["xs"] for p in q], [p["ys"] for p in q], "o-")

        for p in q:
            ax.annotate(f"{p['lambda_s']:.3g}", (p["xs"], p["ys"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)

    if selected:
        ax.plot(selected["xs"], selected["ys"], "r*", ms=16,
                label=f"Selected={selected['lambda_s']:.3g}")

    ax.set(xlabel="Normalized log data fidelity",
           ylabel=f"Normalized log {name}",
           title=f"L-curve: {name}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"lcurve_{name}.png", dpi=180)
    plt.close(fig)


def plot_curvatures(results):
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, result in results.items():
        p = [x for x in result["points"] if x["curvature"] is not None]
        ax.plot([x["lambda_s"] for x in p],
                [x["curvature"] for x in p], "o-", label=name)

    ax.axvline(ORACLE, color="k", ls="--", label=f"Oracle={ORACLE:.3f}")
    ax.set(xlabel="lambda_s", ylabel="Curvature",
           title="L-curve corner comparison")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "curvature_comparison.png", dpi=180)
    plt.close(fig)


def run():
    OUT.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    M, N, T = 41, 96, 7

    D = make_dictionary(M, 4.0)
    A = MultiscaleConvolutionOperator(
        D=D, scales=SCALES, center=(M+1)//2, N=N, normalize=True)

    X_true, _ = make_true_coefficients(
        L=A.L, M=M, J=len(SCALES), T=T)

    S_clean = A.forward(X_true)
    S_noisy, _, _ = add_noise_snr(S_clean, SNR_DB, rng)

    rows = []

    for lam in LAMBDAS:
        c = load_cache(float(lam))

        if c is None:
            print(f"Missing cache: {lam:.4g}")
            continue

        if not c["converged"]:
            print(f"Skipping unconverged: {lam:.4g}")
            continue

        Y = c["Y"]
        row = {
            "lambda_s": float(lam),
            "data": data_fidelity(Y, S_noisy, A),
            "coefficient_error": relative_error(Y, X_true),
            "iterations": c["iterations"],
            **measures(Y, c["a"]),
        }
        rows.append(row)

    if len(rows) < 3:
        raise RuntimeError("Not enough valid cached solutions.")

    keys = {
        "changing_a": "changing_a",
        "L1": "l1",
        "fixed_log_a1": "log_a1",
        "fixed_log_a10": "log_a10",
    }

    results = {}

    for name, key in keys.items():
        selected, points = analyze(rows, key)
        results[name] = {"selected": selected, "points": points}
        plot_curve(name, selected, points)

    plot_curvatures(results)

    with (OUT / "reanalyze_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "oracle_lambda_s": ORACLE,
            "rows": rows,
            "analyses": results,
        }, f, indent=2)

    print("\nL-curve reanalysis")
    print("------------------")
    print("measure             selected    curvature")

    for name, result in results.items():
        s = result["selected"]
        if s:
            print(f"{name:18s}  {s['lambda_s']:8.3f}  {s['curvature']:10.5f}")
        else:
            print(f"{name:18s}  none")

    print(f"\nOracle benchmark: {ORACLE:.3f}")

    print("\nCached solution diagnostics")
    print("---------------------------")
    print("lambda     data      L1     log1    log10   active   Xerr")

    for r in rows:
        print(
            f"{r['lambda_s']:6.3f}  {r['data']:8.4e}  "
            f"{r['l1']:6.3f}  {r['log_a1']:6.3f}  "
            f"{r['log_a10']:7.3f}  {r['active']:6d}  "
            f"{r['coefficient_error']:.4f}"
        )


if __name__ == "__main__":
    run()
