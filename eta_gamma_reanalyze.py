import json
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT = Path("results/eta_gamma_snr14/eta_gamma_summary.json")
OUT = Path("results/eta_gamma_snr14/reanalysis")

DISC_LO, DISC_HI = .90, 1.10
ACTIVE_TOL = 0
PARAM_STABILITY_MAX = .10


def load_data():
    with INPUT.open("r", encoding="utf-8") as f:
        return json.load(f)


def grid_index(values, x):
    return int(np.argmin(np.abs(np.asarray(values)-x)))


def build_grid(rows, eta_values, gamma_values):
    lookup = {}

    for r in rows:
        i = grid_index(eta_values, r["eta"])
        j = grid_index(gamma_values, r["gamma"])
        lookup[(i, j)] = r

    return lookup


def neighbors(i, j, n_eta, n_gamma):
    out = []

    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ii, jj = i+di, j+dj

        if 0 <= ii < n_eta and 0 <= jj < n_gamma:
            out.append((ii, jj))

    return out


def support_signature(r):
    tr = r.get("trajectory")

    if tr is None:
        return None

    return (
        tuple(tr["coefficient_positions"]),
        tuple(tr["scale_indices"]),
    )


def pair_stable(a, b):
    if not (a["converged"] and b["converged"]):
        return False

    if abs(a["active"]-b["active"]) > ACTIVE_TOL:
        return False

    sa = support_signature(a)
    sb = support_signature(b)

    if sa is not None and sb is not None and sa != sb:
        return False

    # Synthetic trajectory fields are deliberately NOT
    # used beyond support topology recorded in the result.
    return True


def feasible(r):
    return (
        r["converged"]
        and DISC_LO <= r["discrepancy_ratio"] <= DISC_HI
    )


def connected_components(
    lookup,
    eta_values,
    gamma_values,
):
    n_eta = len(eta_values)
    n_gamma = len(gamma_values)

    eligible = {
        key
        for key, r in lookup.items()
        if feasible(r)
    }

    components = []
    visited = set()

    for start in eligible:
        if start in visited:
            continue

        q = deque([start])
        visited.add(start)
        comp = []

        while q:
            i, j = q.popleft()
            r = lookup[(i, j)]
            comp.append((i, j))

            for nb in neighbors(
                i, j, n_eta, n_gamma
            ):
                if (
                    nb in eligible
                    and nb not in visited
                    and pair_stable(
                        r,
                        lookup[nb],
                    )
                ):
                    visited.add(nb)
                    q.append(nb)

        components.append(comp)

    return components


def normalized_coordinates(
    component,
    eta_values,
    gamma_values,
):
    coords = np.array([
        [
            eta_values[i],
            gamma_values[j],
        ]
        for i, j in component
    ], dtype=float)

    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.maximum(hi-lo, 1e-12)

    return (coords-lo)/span


def component_center(
    component,
    lookup,
    eta_values,
    gamma_values,
):
    """
    Select the actual grid point nearest the geometric
    center of the connected feasible region.

    Tie-breakers:
      1. smaller discrepancy error;
      2. greater number of feasible/stable neighbors.
    """
    coords = normalized_coordinates(
        component,
        eta_values,
        gamma_values,
    )

    centroid = coords.mean(axis=0)
    distance = np.linalg.norm(
        coords-centroid,
        axis=1,
    )

    min_dist = distance.min()

    candidate_ids = np.where(
        np.isclose(distance, min_dist)
    )[0]

    candidates = []

    member_set = set(component)

    for idx in candidate_ids:
        i, j = component[idx]
        r = lookup[(i, j)]

        degree = sum(
            nb in member_set
            for nb in neighbors(
                i, j,
                len(eta_values),
                len(gamma_values),
            )
        )

        candidates.append((
            abs(np.log(
                r["discrepancy_ratio"]
            )),
            -degree,
            r,
        ))

    return min(
        candidates,
        key=lambda x: (x[0], x[1]),
    )[2]


def component_stats(
    component,
    lookup,
    eta_values,
    gamma_values,
):
    rows = [
        lookup[key]
        for key in component
    ]

    return {
        "size": len(rows),
        "eta_min": min(r["eta"] for r in rows),
        "eta_max": max(r["eta"] for r in rows),
        "gamma_min": min(r["gamma"] for r in rows),
        "gamma_max": max(r["gamma"] for r in rows),
        "mean_discrepancy": float(np.mean([
            r["discrepancy_ratio"] for r in rows
        ])),
        "mean_log_discrepancy": float(np.mean([
            abs(np.log(r["discrepancy_ratio"]))
            for r in rows
        ])),
        "mean_active": float(np.mean([
            r["active"] for r in rows
        ])),
    }


def choose_component(
    components,
    lookup,
    eta_values,
    gamma_values,
):
    if not components:
        raise RuntimeError(
            "No feasible connected eta-gamma regions."
        )

    stats = [
        component_stats(
            c,
            lookup,
            eta_values,
            gamma_values,
        )
        for c in components
    ]

    # Primary criterion: largest connected region.
    # Tie-breaker: best average discrepancy match.
    order = sorted(
        range(len(components)),
        key=lambda i: (
            -stats[i]["size"],
            stats[i]["mean_log_discrepancy"],
        ),
    )

    k = order[0]

    return components[k], stats[k], stats


def clean(r):
    return {
        k: v
        for k, v in r.items()
        if not k.startswith("_")
    }


def plot_regions(
    lookup,
    eta_values,
    gamma_values,
    components,
    selected_component,
    selected,
    oracle,
):
    fig, ax = plt.subplots(figsize=(10, 7))

    # All converged grid points.
    for (i, j), r in lookup.items():
        if r["converged"]:
            ax.scatter(
                gamma_values[j],
                eta_values[i],
                c="lightgray",
                s=35,
                zorder=1,
            )

    colors = plt.cm.tab10(
        np.linspace(
            0,
            1,
            max(len(components), 1),
        )
    )

    for k, comp in enumerate(components):
        xy = np.array([
            [
                gamma_values[j],
                eta_values[i],
            ]
            for i, j in comp
        ])

        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=65,
            color=colors[k],
            label=f"Region {k+1} (n={len(comp)})",
            zorder=2,
        )

    xy = np.array([
        [
            gamma_values[j],
            eta_values[i],
        ]
        for i, j in selected_component
    ])

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=120,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        zorder=3,
    )

    ax.scatter(
        selected["gamma"],
        selected["eta"],
        marker="*",
        s=260,
        c="red",
        label="Selected center",
        zorder=5,
    )

    ax.scatter(
        oracle["gamma"],
        oracle["eta"],
        marker="x",
        s=130,
        linewidths=2.5,
        c="black",
        label="Oracle",
        zorder=5,
    )

    ax.set(
        xlabel="gamma = lambda_t / lambda_s",
        ylabel="eta = lambda_s",
        title=(
            "Connected noise-consistent "
            "eta-gamma regions"
        ),
    )

    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        OUT / "eta_gamma_regions.png",
        dpi=180,
    )
    plt.close(fig)


def plot_discrepancy(
    lookup,
    eta_values,
    gamma_values,
    selected,
    oracle,
):
    Z = np.full(
        (len(eta_values), len(gamma_values)),
        np.nan,
    )

    for (i, j), r in lookup.items():
        if r["converged"]:
            Z[i, j] = r["discrepancy_ratio"]

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[
            min(gamma_values),
            max(gamma_values),
            min(eta_values),
            max(eta_values),
        ],
        vmin=.90,
        vmax=1.10,
    )

    ax.scatter(
        selected["gamma"],
        selected["eta"],
        marker="*",
        s=220,
        c="red",
        label="Selected",
    )

    ax.scatter(
        oracle["gamma"],
        oracle["eta"],
        marker="x",
        s=120,
        linewidths=2.5,
        c="white",
        label="Oracle",
    )

    ax.set(
        xlabel="gamma",
        ylabel="eta",
        title="Discrepancy ratio D / D_noise",
    )

    fig.colorbar(
        im,
        ax=ax,
        label="D / D_noise",
    )

    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUT / "eta_gamma_discrepancy.png",
        dpi=180,
    )
    plt.close(fig)


def run():
    OUT.mkdir(parents=True, exist_ok=True)

    data = load_data()

    rows = data["results"]
    oracle = data["oracle"]

    eta_values = sorted(
        float(x)
        for x in data["eta_grid"]
    )

    gamma_values = sorted(
        float(x)
        for x in data["gamma_grid"]
        if float(x) > 0
    )

    # Exclude gamma=0 from region selection.
    positive_rows = [
        r for r in rows
        if r["gamma"] > 0
    ]

    lookup = build_grid(
        positive_rows,
        eta_values,
        gamma_values,
    )

    components = connected_components(
        lookup,
        eta_values,
        gamma_values,
    )

    chosen_component, chosen_stats, all_stats = (
        choose_component(
            components,
            lookup,
            eta_values,
            gamma_values,
        )
    )

    selected = component_center(
        chosen_component,
        lookup,
        eta_values,
        gamma_values,
    )

    selected_rows = [
        lookup[key]
        for key in chosen_component
    ]

    summary = {
        "input_file": str(INPUT),
        "discrepancy_band": [
            DISC_LO,
            DISC_HI,
        ],
        "activity_tolerance": ACTIVE_TOL,
        "selection_rule": (
            "center of largest connected "
            "noise-consistent support-stable "
            "positive-gamma region"
        ),
        "n_regions": len(components),
        "region_statistics": all_stats,
        "selected_region_statistics":
            chosen_stats,
        "selected": clean(selected),
        "oracle": clean(oracle),
        "selected_region": [
            clean(r)
            for r in selected_rows
        ],
    }

    with (
        OUT / "eta_gamma_reanalysis.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    plot_regions(
        lookup,
        eta_values,
        gamma_values,
        components,
        chosen_component,
        selected,
        oracle,
    )

    plot_discrepancy(
        lookup,
        eta_values,
        gamma_values,
        selected,
        oracle,
    )

    print()
    print("Eta-gamma region reanalysis")
    print("---------------------------")
    print(
        f"feasible regions:    "
        f"{len(components)}"
    )
    print(
        f"selected region size:"
        f" {chosen_stats['size']}"
    )
    print(
        f"eta range:           "
        f"{chosen_stats['eta_min']:.3f}"
        f" to {chosen_stats['eta_max']:.3f}"
    )
    print(
        f"gamma range:         "
        f"{chosen_stats['gamma_min']:.3f}"
        f" to {chosen_stats['gamma_max']:.3f}"
    )
    print()
    print(
        f"selected eta:        "
        f"{selected['eta']:.3f}"
    )
    print(
        f"selected gamma:      "
        f"{selected['gamma']:.3f}"
    )
    print(
        f"selected lambda_s:   "
        f"{selected['lambda_s']:.3f}"
    )
    print(
        f"selected lambda_t:   "
        f"{selected['lambda_t']:.4f}"
    )
    print(
        f"discrepancy ratio:   "
        f"{selected['discrepancy_ratio']:.4f}"
    )
    print(
        f"selected X error:    "
        f"{selected['coefficient_error']:.5f}"
    )
    print()
    print(
        f"oracle eta/gamma:    "
        f"{oracle['eta']:.3f}/"
        f"{oracle['gamma']:.3f}"
    )
    print(
        f"oracle lambdas:      "
        f"{oracle['lambda_s']:.3f}/"
        f"{oracle['lambda_t']:.4f}"
    )
    print(
        f"oracle X error:      "
        f"{oracle['coefficient_error']:.5f}"
    )


if __name__ == "__main__":
    run()
