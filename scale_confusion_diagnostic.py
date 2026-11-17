import numpy as np

from gaussian_experiment import (
    SCALES,
    add_noise_snr,
    make_dictionary,
)
from local_graph import physical_scale_values
from multiscale_convolution import MultiscaleConvolutionOperator


SNR_VALUES = [20, 17, 14, 11, 8]
SEED = 100

M, N, T = 41, 96, 12

ACTIVE_FRAMES = np.arange(2, 10)

TRUE_POSITIONS = np.array([
    28, 29, 30, 31,
    32, 33, 34, 35,
])

TRUE_SCALES = np.array([
    0, 1, 2, 3,
    3, 2, 1, 0,
])


def make_truth(A):
    X = np.zeros(
        (A.L, A.K, A.J, T),
        dtype=float,
    )

    for t, pos, j in zip(
        ACTIVE_FRAMES,
        TRUE_POSITIONS,
        TRUE_SCALES,
    ):
        n = int(pos + M - 1)
        X[n, 0, j, t] = 1.0

    return X


def atom_signal(A, n, j):
    """
    Signal produced by one unit coefficient at
    coefficient position n and scale j.
    """
    X = np.zeros(
        (A.L, A.K, A.J, 1),
        dtype=float,
    )

    X[n, 0, j, 0] = 1.0

    return A.forward(X)[:, 0]


def optimal_amplitude(y, atom):
    denom = float(
        np.dot(atom, atom)
    )

    if denom <= 0:
        raise ValueError(
            "Atom has zero norm"
        )

    return max(
        0.0,
        float(
            np.dot(atom, y)
            / denom
        ),
    )


def residual_energy(y, atom, amplitude):
    r = y - amplitude * atom

    return float(
        0.5 * np.dot(r, r)
    )


def frame_scale_confusion(
    A,
    y,
    physical_position,
    true_scale,
):
    n = int(
        physical_position
        + M - 1
    )

    rows = []

    for j in range(A.J):
        atom = atom_signal(
            A,
            n,
            j,
        )

        amp = optimal_amplitude(
            y,
            atom,
        )

        energy = residual_energy(
            y,
            atom,
            amp,
        )

        rows.append({
            "scale": j,
            "amplitude": amp,
            "energy": energy,
        })

    rows.sort(
        key=lambda r:
            r["energy"]
    )

    best = rows[0]
    second = rows[1]

    true_row = next(
        r for r in rows
        if r["scale"] == true_scale
    )

    gap = (
        second["energy"]
        - best["energy"]
    )

    relative_gap = (
        gap
        / max(
            abs(best["energy"]),
            1e-15,
        )
    )

    true_excess = (
        true_row["energy"]
        - best["energy"]
    )

    return {
        "rows": rows,
        "best_scale":
            best["scale"],
        "second_scale":
            second["scale"],
        "true_scale":
            int(true_scale),
        "best_amplitude":
            best["amplitude"],
        "true_amplitude":
            true_row["amplitude"],
        "best_energy":
            best["energy"],
        "true_energy":
            true_row["energy"],
        "gap":
            gap,
        "relative_gap":
            relative_gap,
        "true_excess":
            true_excess,
        "truth_is_best":
            best["scale"]
            == true_scale,
    }


def print_frame_result(
    snr,
    t,
    pos,
    result,
):
    print()
    print(
        f"SNR={snr:2d} dB  "
        f"t={t:2d}  "
        f"position={pos:2d}  "
        f"true j={result['true_scale']}  "
        f"best j={result['best_scale']}"
    )

    print(
        "j   amplitude      energy"
    )

    by_scale = sorted(
        result["rows"],
        key=lambda r:
            r["scale"],
    )

    for r in by_scale:
        marker = ""

        if (
            r["scale"]
            == result["true_scale"]
        ):
            marker += " true"

        if (
            r["scale"]
            == result["best_scale"]
        ):
            marker += " best"

        print(
            f"{r['scale']:1d} "
            f"{r['amplitude']:11.6f} "
            f"{r['energy']:12.6e}"
            f"{marker}"
        )

    print(
        f"best-second gap: "
        f"{result['gap']:.6e}"
    )

    print(
        f"relative gap:    "
        f"{result['relative_gap']:.6e}"
    )

    print(
        f"true excess:     "
        f"{result['true_excess']:.6e}"
    )


def main():
    D = make_dictionary(
        M,
        4.0,
    )

    A = MultiscaleConvolutionOperator(
        D=D,
        scales=SCALES,
        center=(M + 1) // 2,
        N=N,
        normalize=True,
    )

    Xtrue = make_truth(A)
    Sclean = A.forward(Xtrue)

    scales = physical_scale_values(
        SCALES
    )

    print()
    print("Scale-confusion diagnostic")
    print("--------------------------")
    print(
        "physical scales:",
        scales,
    )

    summaries = []

    for snr in SNR_VALUES:
        # Same convention as the multi-SNR experiment:
        # identical seeded noise direction, rescaled
        # to each requested SNR.
        rng = np.random.default_rng(
            SEED
        )

        Snoisy, _, achieved = (
            add_noise_snr(
                Sclean,
                snr,
                rng,
            )
        )

        correct = 0
        gaps = []
        true_excess = []

        print()
        print("=" * 70)
        print(
            f"SNR = {achieved:.2f} dB"
        )
        print("=" * 70)

        for t, pos, true_j in zip(
            ACTIVE_FRAMES,
            TRUE_POSITIONS,
            TRUE_SCALES,
        ):
            y = Snoisy[:, t]

            result = (
                frame_scale_confusion(
                    A,
                    y,
                    physical_position=
                        int(pos),
                    true_scale=
                        int(true_j),
                )
            )

            correct += int(
                result["truth_is_best"]
            )

            gaps.append(
                result["gap"]
            )

            true_excess.append(
                result["true_excess"]
            )

            print_frame_result(
                snr,
                int(t),
                int(pos),
                result,
            )

        summary = {
            "snr": snr,
            "correct":
                correct,
            "accuracy":
                correct
                / len(ACTIVE_FRAMES),
            "mean_gap":
                float(np.mean(gaps)),
            "min_gap":
                float(np.min(gaps)),
            "mean_true_excess":
                float(
                    np.mean(true_excess)
                ),
            "max_true_excess":
                float(
                    np.max(true_excess)
                ),
        }

        summaries.append(summary)

    print()
    print("Summary")
    print("-------")
    print(
        "SNR  frame-best "
        "mean-gap     min-gap      "
        "mean-true-excess max-true-excess"
    )

    for s in summaries:
        print(
            f"{s['snr']:3d} "
            f"{s['correct']:2d}/8 "
            f"({100*s['accuracy']:5.1f}%) "
            f"{s['mean_gap']:11.4e} "
            f"{s['min_gap']:11.4e} "
            f"{s['mean_true_excess']:16.4e} "
            f"{s['max_true_excess']:15.4e}"
        )


if __name__ == "__main__":
    main()