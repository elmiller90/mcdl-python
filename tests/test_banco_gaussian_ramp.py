import numpy as np

from banco_gaussian_ramp import (
    BANCO_SCALES,
    banco_trajectories,
    make_banco_coefficients,
    make_banco_dictionary,
)


def test_banco_geometry():
    M, N, T = 41, 80, 80
    L = N + M - 1

    assert BANCO_SCALES == [
        (1, 5), (1, 4), (1, 3), (2, 5), (1, 2),
        (3, 5), (2, 3), (3, 4), (4, 5), (1, 1),
    ]

    D = make_banco_dictionary(M)
    assert D.shape == (41, 2)
    assert np.all(D >= 0)
    assert np.allclose(np.linalg.norm(D, axis=0), 1.0)

    # Ramp occupies MATLAB 6:35 -> Python 5:34.
    assert np.flatnonzero(D[:, 1])[0] == 6
    assert np.flatnonzero(D[:, 1])[-1] == 34

    p1, p2, j1, j2 = banco_trajectories(T, M, N, 10)

    assert np.all((p1 >= 0) & (p1 < L))
    assert np.all((p2 >= 0) & (p2 < L))
    assert np.all((j1 >= 0) & (j1 < 10))
    assert np.all((j2 >= 0) & (j2 < 10))

    X = make_banco_coefficients(T, M, N)
    assert X.shape == (120, 2, 10, 80)
    assert np.count_nonzero(X) == 160
    assert np.allclose(np.sum(X, axis=(0, 1, 2)), 2.0)

    print("scales:", BANCO_SCALES)
    print("position ranges:", (p1.min(), p1.max()), (p2.min(), p2.max()))
    print("scale ranges:", (j1.min(), j1.max()), (j2.min(), j2.max()))