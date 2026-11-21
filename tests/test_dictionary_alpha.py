import numpy as np

from dictionary_alpha import compute_dictionary_alpha
from dictionary_projection import project_dictionary_nonnegative
from multiscale_operator import resample_atom


def test_dictionary_alpha_normalization():
    rng = np.random.default_rng(6)

    N, M, K = 32, 9, 3
    scales = [
        (1, 1),
        (2, 1),
        (1, 2),
    ]

    center = (M + 1) // 2

    A = rng.standard_normal((M, K))
    G = project_dictionary_nonnegative(A)

    alpha, norm_vals = compute_dictionary_alpha(
        G=G,
        scales=scales,
        center=center,
        N=N,
    )

    errors = []

    for k in range(K):
        for j, (c1, c2) in enumerate(scales):
            atom, _ = resample_atom(
                G[:, k],
                c1,
                c2,
                center,
                N,
            )

            scaled = alpha[k, j] * atom
            err = abs(np.linalg.norm(scaled) - 1.0)
            errors.append(err)

    maxerr = max(errors)

    print("NormVals:")
    print(norm_vals)
    print("alpha:")
    print(alpha)
    print(
        f"maximum normalized-atom error: "
        f"{maxerr:.3e}"
    )

    assert np.all(norm_vals > 0.0)
    assert np.allclose(
        alpha,
        1.0 / norm_vals,
    )
    assert maxerr < 1e-12