import time

import numpy as np

from local_graph import (
    build_local_transport_graph,
    state_index,
)
from local_sinkhorn_log import local_sinkhorn_log


SCALES = np.array([
    1.0, 0.8, 0.75, 2.0/3.0, 0.5
])

L = 136
T = 20
EPSILON = 0.1
TOL = 1e-6
MAXITER = 5000
CHECK_EVERY = 5


def make_sparse_frame(
    graph,
    center_n,
    center_j,
    amplitude=1.0,
):
    """
    Sparse local coefficient distribution.

    Seven active states:
      center
      +/-1 shift
      +/-2 shift
      +/-1 scale
    """
    J = graph.J
    x = np.zeros(graph.P, dtype=np.float64)

    entries = [
        (center_n, center_j, 1.00),
        (center_n-1, center_j, .25),
        (center_n+1, center_j, .25),
        (center_n-2, center_j, .08),
        (center_n+2, center_j, .08),
    ]

    if center_j > 0:
        entries.append(
            (center_n, center_j-1, .12)
        )

    if center_j < J-1:
        entries.append(
            (center_n, center_j+1, .12)
        )

    for n, j, a in entries:
        if 0 <= n < graph.L and 0 <= j < J:
            x[state_index(n, j, J)] += amplitude*a

    x /= x.sum()
    return x


def build_frame_sequence(graph):
    positions = [
        45, 45, 46, 46, 47,
        47, 48, 48, 49, 49,
        50, 50, 51, 51, 52,
        52, 53, 53, 54, 54,
    ]

    scale_indices = [
        1, 1, 1, 2, 2,
        2, 2, 3, 3, 3,
        3, 2, 2, 2, 1,
        1, 1, 1, 1, 1,
    ]

    assert len(positions) == T
    assert len(scale_indices) == T

    return [
        make_sparse_frame(
            graph,
            n,
            j,
        )
        for n, j in zip(
            positions,
            scale_indices,
        )
    ]


def solve_pair(
    p,
    q,
    graph,
    warm_start=None,
):
    start = time.perf_counter()

    result = local_sinkhorn_log(
        p,
        q,
        graph,
        epsilon=EPSILON,
        tol=TOL,
        maxiter=MAXITER,
        warm_start=warm_start,
        check_every=CHECK_EVERY,
        center_every=50,
    )

    elapsed = time.perf_counter() - start
    return result, elapsed


def run_cold(frames, graph):
    rows = []

    for t in range(T-1):
        r, elapsed = solve_pair(
            frames[t],
            frames[t+1],
            graph,
            warm_start=None,
        )

        rows.append({
            "t": t,
            "iterations": r["info"]["iterations"],
            "converged": r["info"]["converged"],
            "error": r["info"]["marginal_error"],
            "time": elapsed,
        })

    return rows


def adapt_warm_start(
    previous,
    p,
    q,
):
    """
    Keep previous finite potentials where support persists.
    Newly active states are initialized to zero by
    local_sinkhorn_log().
    """
    if previous is None:
        return None

    f = previous["dual_source"].copy()
    g = previous["dual_target"].copy()

    return f, g


def run_warm(frames, graph):
    rows = []
    previous = None

    for t in range(T-1):
        warm = adapt_warm_start(
            previous,
            frames[t],
            frames[t+1],
        )

        r, elapsed = solve_pair(
            frames[t],
            frames[t+1],
            graph,
            warm_start=warm,
        )

        rows.append({
            "t": t,
            "iterations": r["info"]["iterations"],
            "converged": r["info"]["converged"],
            "error": r["info"]["marginal_error"],
            "time": elapsed,
        })

        previous = r

    return rows


def summarize(name, rows):
    iterations = np.array(
        [r["iterations"] for r in rows]
    )

    runtime = np.array(
        [r["time"] for r in rows]
    )

    errors = np.array(
        [r["error"] for r in rows]
    )

    converged = sum(
        r["converged"]
        for r in rows
    )

    print()
    print(name)
    print("-" * len(name))
    print(
        f"converged:     "
        f"{converged}/{len(rows)}"
    )
    print(
        f"iter mean:     "
        f"{iterations.mean():.1f}"
    )
    print(
        f"iter median:   "
        f"{np.median(iterations):.1f}"
    )
    print(
        f"iter max:      "
        f"{iterations.max()}"
    )
    print(
        f"runtime total: "
        f"{runtime.sum():.3f} s"
    )
    print(
        f"runtime mean:  "
        f"{runtime.mean():.4f} s"
    )
    print(
        f"max error:     "
        f"{errors.max():.3e}"
    )

    return {
        "iterations": iterations,
        "runtime": runtime,
    }


def print_pairwise(cold, warm):
    print()
    print("Pairwise comparison")
    print("-------------------")
    print(
        "pair   cold_iter   warm_iter   "
        "cold_ms   warm_ms"
    )

    for c, w in zip(cold, warm):
        print(
            f"{c['t']:4d}  "
            f"{c['iterations']:9d}  "
            f"{w['iterations']:9d}  "
            f"{1000*c['time']:8.2f}  "
            f"{1000*w['time']:8.2f}"
        )


def main():
    graph = build_local_transport_graph(
        L=L,
        scales=SCALES,
        shift_radius=3,
        scale_radius=1,
        shift_velocity=1.0,
    )

    frames = build_frame_sequence(graph)

    print("Sparse nearby-frame benchmark")
    print("-----------------------------")
    print(f"P:           {graph.P}")
    print(f"E:           {graph.E}")
    print(f"frames:      {T}")
    print(f"pairs:       {T-1}")
    print(f"epsilon:     {EPSILON}")
    print(f"tolerance:   {TOL}")

    cold = run_cold(frames, graph)
    warm = run_warm(frames, graph)

    c = summarize(
        "Cold start",
        cold,
    )

    w = summarize(
        "Warm start",
        warm,
    )

    print_pairwise(cold, warm)

    speedup_iter = (
        c["iterations"].sum()
        / w["iterations"].sum()
    )

    speedup_time = (
        c["runtime"].sum()
        / w["runtime"].sum()
    )

    print()
    print("Warm-start gain")
    print("---------------")
    print(
        f"iteration speedup: "
        f"{speedup_iter:.2f}x"
    )
    print(
        f"runtime speedup:   "
        f"{speedup_time:.2f}x"
    )


if __name__ == "__main__":
    main()