import numpy as np

from dictionary_projection import project_dictionary_nonnegative


def test_dictionary_projection():
    rng = np.random.default_rng(5)

    M, K = 9, 3
    A = rng.standard_normal((M, K))

    G = project_dictionary_nonnegative(A)

    norms = np.linalg.norm(G, axis=0)

    print("minimum value:", np.min(G))
    print("atom norms:", norms)

    assert G.shape == (M, K)
    assert np.all(G >= 0.0)
    assert np.allclose(norms, 1.0)


def test_dictionary_projection_zero_atom():
    A = -np.ones((9, 2))

    G = project_dictionary_nonnegative(A)

    print("zero projection norms:", np.linalg.norm(G, axis=0))

    assert np.all(np.isfinite(G))
    assert np.all(G == 0.0)