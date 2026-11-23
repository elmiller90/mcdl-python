import numpy as np

from dictionary_projection import project_dictionary_nonnegative
from dictionary_initialization import make_banco_boxes



def test_banco_box_initialization():
    M, K = 41, 2
    G = make_banco_boxes(M, K)

    widths = [
        np.count_nonzero(G[:, k])
        for k in range(K)
    ]

    norms = np.linalg.norm(G, axis=0)

    support0 = np.flatnonzero(G[:, 0])
    support1 = np.flatnonzero(G[:, 1])

    print("widths:", widths)
    print("norms:", norms)
    print("atom 0 support:", support0[[0, -1]])
    print("atom 1 support:", support1[[0, -1]])

    assert widths == [21, 41]

    assert support0[0] == 10
    assert support0[-1] == 30

    assert support1[0] == 0
    assert support1[-1] == 40

    assert np.allclose(
        np.linalg.norm(G, axis=0),
        1.0,
    )