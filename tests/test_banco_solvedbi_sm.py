import numpy as np

from banco_solvedbi_sm import solvedbi_sm_rank1


def test_rank1_solver_against_direct_complex_solve():
    rng = np.random.default_rng(80)

    Q, F, T = 7, 20, 3
    rho = 3.7

    a = (
        rng.standard_normal((Q, F, T))
        + 1j*rng.standard_normal((Q, F, T))
    )

    b = (
        rng.standard_normal((Q, F, T))
        + 1j*rng.standard_normal((Q, F, T))
    )

    x = solvedbi_sm_rank1(a, rho, b)

    for q in range(Q):
        for t in range(T):
            av = a[q, :, t]
            bv = b[q, :, t]

            H = (
                rho*np.eye(F)
                + np.outer(np.conj(av), av)
            )

            xd = np.linalg.solve(H, bv)

            assert np.allclose(
                x[q, :, t],
                xd,
                rtol=1e-11,
                atol=1e-12,
            )


def test_rank1_solver_residual():
    rng = np.random.default_rng(81)

    a = (
        rng.standard_normal((5, 4, 2))
        + 1j*rng.standard_normal((5, 4, 2))
    )
    b = (
        rng.standard_normal((5, 4, 2))
        + 1j*rng.standard_normal((5, 4, 2))
    )

    rho = 2.0
    x = solvedbi_sm_rank1(a, rho, b)

    Ax = np.sum(
        a*x,
        axis=-2,
        keepdims=True,
    )

    lhs = rho*x + np.conj(a)*Ax

    rel = np.linalg.norm(lhs-b)/np.linalg.norm(b)
    print(f"relative residual: {rel:.3e}")

    assert rel < 1e-12


def test_invalid_rho():
    a = np.ones((3, 2, 1), dtype=complex)
    b = np.ones_like(a)

    try:
        solvedbi_sm_rank1(a, 0.0, b)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")