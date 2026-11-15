import json
from collections import deque
from pathlib import Path

import numpy as np


INPUT = Path(
    "results/barycentric_joint_snr14/"
    "barycentric_joint_summary.json"
)

OUTPUT = Path(
    "results/barycentric_joint_snr14/"
    "barycentric_region_reanalysis.json"
)

DISC_LO = 0.98
DISC_HI = 1.02


def load_data():
    with INPUT.open("r", encoding="utf-8") as f:
        return json.load(f)


def key(x):
    return round(float(x), 10)


def trajectory_signature(row):
    tr = row["trajectory"]

    return (
        tuple(tr["physical_positions"]),
        tuple(tr["scale_indices"]),
    )


def build_lookup(rows):
    return {
        (
            key(r["lambda_s"]),
            key(r["lambda_t"]),
        ): r
        for r in rows
    }


def grid_neighbors(ls, lt, ls_grid, lt_grid):
    i = ls_grid.index(key(ls))
    j = lt_grid.index(key(lt))

    out = []

    for di, dj in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]:
        ii = i + di
        jj = j + dj

        if (
            0 <= ii < len(ls_grid)
            and 0 <= jj < len(lt_grid)
        ):
            out.append(
                (
                    ls_grid[ii],
                    lt_grid[jj],
                )
            )

    return out


def connected_regions(
    eligible,
    ls_grid,
    lt_grid,
):
    lookup = build_lookup(eligible)

    unvisited = set(lookup.keys())
    regions = []

    while unvisited:
        start = next(iter(unvisited))

        queue = deque([start])
        unvisited.remove(start)

        region = []

        while queue:
            current = queue.popleft()
            row = lookup[current]
            region.append(row)

            ls, lt = current
            sig = trajectory_signature(row)

            for nb in grid_neighbors(
                ls,
                lt,
                ls_grid,
                lt_grid,
            ):
                if nb not in unvisited:
                    continue

                candidate = lookup.get(nb)

                if candidate is None:
                    continue

                if (
                    trajectory_signature(candidate)
                    != sig
                ):
                    continue

                unvisited.remove(nb)
                queue.append(nb)

        regions.append(region)

    return regions


def normalized_coordinates(region):
    ls = np.array([
        r["lambda_s"]
        for r in region
    ])

    lt = np.array([
        r["lambda_t"]
        for r in region
    ])

    ls_span = ls.max() - ls.min()
    lt_span = lt.max() - lt.min()

    if ls_span == 0:
        x = np.zeros_like(ls)
    else:
        x = (
            ls - ls.min()
        ) / ls_span

    if lt_span == 0:
        y = np.zeros_like(lt)
    else:
        y = (
            lt - lt.min()
        ) / lt_span

    return np.column_stack((x, y))


def region_center(region):
    """
    Select actual grid point nearest the geometric centroid
    of the connected parameter region.

    Tie-breakers:
      1. smaller |log(D/Dnoise)|
      2. smaller distance to centroid
      3. smaller lambda_s
      4. smaller lambda_t
    """
    coords = normalized_coordinates(
        region
    )

    centroid = coords.mean(axis=0)

    distances = np.linalg.norm(
        coords - centroid,
        axis=1,
    )

    candidates = []

    for i, row in enumerate(region):
        disc_error = abs(
            np.log(
                row["discrepancy_ratio"]
            )
        )

        candidates.append(
            (
                distances[i],
                disc_error,
                row["lambda_s"],
                row["lambda_t"],
                row,
            )
        )

    candidates.sort(
        key=lambda z: (
            z[0],
            z[1],
            z[2],
            z[3],
        )
    )

    return candidates[0][-1]


def region_stats(region):
    return {
        "size": len(region),
        "lambda_s_min": min(
            r["lambda_s"]
            for r in region
        ),
        "lambda_s_max": max(
            r["lambda_s"]
            for r in region
        ),
        "lambda_t_min": min(
            r["lambda_t"]
            for r in region
        ),
        "lambda_t_max": max(
            r["lambda_t"]
            for r in region
        ),
        "discrepancy_min": min(
            r["discrepancy_ratio"]
            for r in region
        ),
        "discrepancy_max": max(
            r["discrepancy_ratio"]
            for r in region
        ),
        "trajectory":
            region[0]["trajectory"],
    }


def clean(row):
    return dict(row)


def main():
    data = load_data()

    rows = data["results"]

    ls_grid = [
        key(x)
        for x in data[
            "lambda_s_grid"
        ]
    ]

    lt_grid = [
        key(x)
        for x in data[
            "lambda_t_grid"
        ]
    ]

    converged = [
        r for r in rows
        if r["converged"]
    ]

    broad_feasible = [
        r for r in converged
        if (
            0.90
            <= r["discrepancy_ratio"]
            <= 1.10
        )
    ]

    if not broad_feasible:
        raise RuntimeError(
            "No broadly feasible grid points."
        )

    min_support = min(
        r["active"]
        for r in broad_feasible
    )

    eligible = [
        r for r in converged
        if (
            r["active"] == min_support
            and DISC_LO
            <= r["discrepancy_ratio"]
            <= DISC_HI
            and r["lambda_t"] > 0
        )
    ]

    if not eligible:
        raise RuntimeError(
            "No points satisfy the "
            "sparse discrepancy filter."
        )

    regions = connected_regions(
        eligible,
        ls_grid,
        lt_grid,
    )

    if not regions:
        raise RuntimeError(
            "No connected regions found."
        )

    # Prefer largest connected stable region.
    # Tie-break by mean discrepancy error.
    regions.sort(
        key=lambda region: (
            -len(region),
            np.mean([
                abs(np.log(
                    r["discrepancy_ratio"]
                ))
                for r in region
            ]),
        )
    )

    selected_region = regions[0]
    selected = region_center(
        selected_region
    )

    stats = region_stats(
        selected_region
    )

    # Synthetic diagnostics only after selection.
    oracle = data["oracle"]

    output = {
        "selection_rule": {
            "description":
                "center of largest connected "
                "minimum-support, noise-consistent, "
                "trajectory-stable positive-lambda_t region",
            "discrepancy_band": [
                DISC_LO,
                DISC_HI,
            ],
            "minimum_support":
                min_support,
            "uses_ground_truth":
                False,
        },
        "eligible_count":
            len(eligible),
        "region_count":
            len(regions),
        "selected_region":
            stats,
        "selected":
            clean(selected),
        "oracle_posthoc":
            clean(oracle),
        "regions": [
            {
                "statistics":
                    region_stats(region),
                "members": [
                    {
                        "lambda_s":
                            r["lambda_s"],
                        "lambda_t":
                            r["lambda_t"],
                        "active":
                            r["active"],
                        "discrepancy_ratio":
                            r[
                                "discrepancy_ratio"
                            ],
                        "trajectory":
                            r["trajectory"],
                    }
                    for r in region
                ],
            }
            for region in regions
        ],
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("Barycentric region reanalysis")
    print("-----------------------------")
    print(
        f"minimum feasible support: "
        f"{min_support}"
    )
    print(
        f"eligible points:          "
        f"{len(eligible)}"
    )
    print(
        f"connected regions:        "
        f"{len(regions)}"
    )

    print()
    print("Largest stable region")
    print("---------------------")
    print(
        f"size:          "
        f"{stats['size']}"
    )
    print(
        f"lambda_s:      "
        f"{stats['lambda_s_min']:.3f}"
        f" to "
        f"{stats['lambda_s_max']:.3f}"
    )
    print(
        f"lambda_t:      "
        f"{stats['lambda_t_min']:.4g}"
        f" to "
        f"{stats['lambda_t_max']:.4g}"
    )
    print(
        f"D/Dnoise:      "
        f"{stats['discrepancy_min']:.4f}"
        f" to "
        f"{stats['discrepancy_max']:.4f}"
    )

    print()
    print("Unsupervised selection")
    print("----------------------")
    print(
        f"lambda_s:      "
        f"{selected['lambda_s']:.4f}"
    )
    print(
        f"lambda_t:      "
        f"{selected['lambda_t']:.6f}"
    )
    print(
        f"support:       "
        f"{selected['active']}"
    )
    print(
        f"D/Dnoise:      "
        f"{selected['discrepancy_ratio']:.6f}"
    )
    print(
        "trajectory:    "
        f"{selected['trajectory']['physical_positions']} "
        f"{selected['trajectory']['scale_indices']}"
    )

    print()
    print("Post-hoc synthetic validation")
    print("-----------------------------")
    print(
        f"selected Xerr: "
        f"{selected['coefficient_error']:.6f}"
    )
    print(
        f"selected SNR:  "
        f"{selected['output_snr_db']:.2f} dB"
    )
    print(
        f"oracle:        "
        f"ls={oracle['lambda_s']:.4f}, "
        f"lt={oracle['lambda_t']:.6f}"
    )
    print(
        f"oracle support:"
        f" {oracle['active']}"
    )
    print(
        f"oracle Xerr:   "
        f"{oracle['coefficient_error']:.6f}"
    )


if __name__ == "__main__":
    main()
    