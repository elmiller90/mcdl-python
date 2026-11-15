import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT = Path("results/joint_grid_snr14/joint_summary.json")
OUT = Path("results/joint_grid_snr14/reanalysis")

JACCARD_MIN = .90

# Log-discrepancy tolerances around the best attainable noise match.
LOG_TOLS = [.005, .01, .02, .03]


def load_data():
    with INPUT.open("r", encoding="utf-8") as f:
        return json.load(f)


def log_discrepancy(r):
    return abs(np.log(r["discrepancy_ratio"]))


def stable_rows(rows):
    return [
        r for r in rows
        if (
            r["converged"]
            and r["neighbor_jaccard"] is not None
            and r["neighbor_jaccard"] >= JACCARD_MIN
        )
    ]


def normalize(v):
    v = np.asarray(v, float)
    lo, hi = v.min(), v.max()
    return np.zeros_like(v) if hi == lo else (v-lo)/(hi-lo)


def pareto_filter(rows):
    out = []

    for i, a in enumerate(rows):
        dominated = False

        for j, b in enumerate(rows):
            if i == j:
                continue

            better = (
                b["sparsity"] <= a["sparsity"]
                and b["temporal"] <= a["temporal"]
            )

            strict = (
                b["sparsity"] < a["sparsity"]
                or b["temporal"] < a["temporal"]
            )

            if better and strict:
                dominated = True
                break

        if not dominated:
            out.append(a)

    return out


def closest_discrepancy(rows):
    return min(rows, key=log_discrepancy)


def discrepancy_neighborhood(rows, tol):
    best = min(log_discrepancy(r) for r in rows)

    return [
        r for r in rows
        if log_discrepancy(r) <= best + tol
    ]


def minimum_active_then_temporal(rows):
    qmin = min(r["active"] for r in rows)
    q = [r for r in rows if r["active"] == qmin]

    return min(q, key=lambda r: r["temporal"])


def pareto_ideal(rows):
    p = pareto_filter(rows)

    s = normalize([r["sparsity"] for r in p])
    t = normalize([r["temporal"] for r in p])

    for r, sn, tn in zip(p, s, t):
        r["_distance"] = float(np.sqrt(sn*sn + tn*tn))

    selected = min(p, key=lambda r: r["_distance"])

    return selected, p


def discrepancy_then_pareto(rows, tol):
    near = discrepancy_neighborhood(rows, tol)
    selected, pareto = pareto_ideal(near)

    return selected, near, pareto


def discrepancy_then_active(rows, tol):
    near = discrepancy_neighborhood(rows, tol)
    selected = minimum_active_then_temporal(near)

    return selected, near


def score_row(name, r, oracle):
    return {
        "rule": name,
        "lambda_s": r["lambda_s"],
        "lambda_t": r["lambda_t"],
        "discrepancy_ratio": r["discrepancy_ratio"],
        "log_discrepancy": log_discrepancy(r),
        "sparsity": r["sparsity"],
        "temporal": r["temporal"],
        "active": r["active"],
        "neighbor_jaccard": r["neighbor_jaccard"],
        "coefficient_error": r["coefficient_error"],
        "signal_error": r["signal_error"],
        "output_snr_db": r["output_snr_db"],
        "trajectory_accuracy": r["trajectory_accuracy"],
        "distance_to_oracle_parameters": float(np.sqrt(
            (r["lambda_s"]-oracle["lambda_s"])**2
            + (r["lambda_t"]-oracle["lambda_t"])**2
        )),
    }


def plot_grid(rows, choices, oracle):
    ls = sorted(set(r["lambda_s"] for r in rows))
    lt = sorted(set(r["lambda_t"] for r in rows))

    Z = np.full((len(ls), len(lt)), np.nan)

    for r in rows:
        i = ls.index(r["lambda_s"])
        j = lt.index(r["lambda_t"])
        Z[i, j] = r["discrepancy_ratio"]

    fig, ax = plt.subplots(figsize=(9, 6))

    im = ax.imshow(
        Z, origin="lower", aspect="auto",
        extent=[min(lt), max(lt), min(ls), max(ls)],
    )

    ax.scatter(
        oracle["lambda_t"], oracle["lambda_s"],
        marker="*", s=180, c="white",
        edgecolor="black", label="Oracle",
    )

    markers = ["o", "s", "^", "D", "P", "X"]

    for marker, (name, r) in zip(markers, choices.items()):
        ax.scatter(
            r["lambda_t"], r["lambda_s"],
            marker=marker, s=90, label=name,
        )

    ax.set(
        xlabel="lambda_t",
        ylabel="lambda_s",
        title="Known-noise joint parameter reanalysis",
    )

    fig.colorbar(im, ax=ax, label="Discrepancy ratio")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "selection_comparison.png", dpi=180)
    plt.close(fig)


def plot_errors(results):
    names = [r["rule"] for r in results]
    errors = [r["coefficient_error"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(names)), errors)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Relative coefficient error")
    ax.set_title("Post-hoc comparison of selection rules")
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "selection_errors.png", dpi=180)
    plt.close(fig)


def run():
    OUT.mkdir(parents=True, exist_ok=True)

    data = load_data()
    rows = data["results"]
    oracle = data["oracle"]

    stable = stable_rows(rows)

    if not stable:
        raise RuntimeError("No stable converged grid points.")

    print(f"Grid points:           {len(rows)}")
    print(f"Stable converged:      {len(stable)}")

    best_disc = closest_discrepancy(stable)

    choices = {
        "closest discrepancy": best_disc,
    }

    details = {
        "closest_discrepancy": {
            "selected": best_disc
        }
    }

    # Compare several discrepancy-neighborhood widths.
    for tol in LOG_TOLS:
        name = f"active tol={tol:g}"
        selected, near = discrepancy_then_active(stable, tol)

        choices[name] = selected

        details[f"active_{tol:g}"] = {
            "tolerance": tol,
            "n_candidates": len(near),
            "selected": selected,
        }

        name = f"pareto tol={tol:g}"
        selected, near, pareto = discrepancy_then_pareto(
            stable, tol
        )

        choices[name] = selected

        details[f"pareto_{tol:g}"] = {
            "tolerance": tol,
            "n_candidates": len(near),
            "n_pareto": len(pareto),
            "selected": selected,
        }

    scored = [
        score_row(name, row, oracle)
        for name, row in choices.items()
    ]

    summary = {
        "oracle": oracle,
        "jaccard_min": JACCARD_MIN,
        "log_discrepancy_tolerances": LOG_TOLS,
        "stable_count": len(stable),
        "rules": scored,
        "details": details,
    }

    with (OUT / "joint_reanalysis.json").open(
            "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_grid(rows, choices, oracle)
    plot_errors(scored)

    print()
    print("Joint parameter reanalysis")
    print("--------------------------")
    print(
        "rule                  ls      lt      "
        "D/Dn     nz    J      Xerr     outSNR"
    )

    for r in scored:
        print(
            f"{r['rule'][:20]:20s}  "
            f"{r['lambda_s']:5.3f}  "
            f"{r['lambda_t']:5.3f}  "
            f"{r['discrepancy_ratio']:6.3f}  "
            f"{r['active']:3d}  "
            f"{r['neighbor_jaccard']:5.3f}  "
            f"{r['coefficient_error']:7.4f}  "
            f"{r['output_snr_db']:7.2f}"
        )

    print()
    print(
        f"Oracle: ls={oracle['lambda_s']:.3f}, "
        f"lt={oracle['lambda_t']:.3f}, "
        f"Xerr={oracle['coefficient_error']:.5f}"
    )


if __name__ == "__main__":
    run()
