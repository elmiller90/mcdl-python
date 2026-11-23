import numpy as np

from multiscale_convolution import MultiscaleConvolutionOperator
from sparse_admm_stateful_compat import solve_l1_admm_stateful


def make_problem():
    M, N = 21, 48
    x = np.arange(M, dtype=float)
    d = np.exp(-0.5*((x-(M-1)/2)/3.0)**2)
    d /= np.linalg.norm(d)

    A = MultiscaleConvolutionOperator(
        D=d[:, None],
        scales=[(1, 1)],
        center=(M+1)//2,
        N=N,
        normalize=True,
    )

    Xtrue = np.zeros((A.L, 1, 1, 1))
    Xtrue[M-1+18, 0, 0, 0] = 1.0
    S = A.forward(Xtrue)

    X0 = A.adjoint(S, structured=True)
    X0 /= np.max(np.abs(X0))

    return A, S, X0


def test_stateful_one_steps_match_multi_step():
    A, S, X0 = make_problem()

    _, Ym, _, _ = solve_l1_admm_stateful(
        X0, S, A, lambda1=0.03,
        rho=1.0, max_admm_iter=5,
        adaptive_rho=False,
    )

    state = None
    Xstart = X0.copy()

    for _ in range(5):
        _, Ys, _, info = solve_l1_admm_stateful(
            Xstart, S, A, lambda1=0.03,
            rho=1.0, max_admm_iter=1,
            adaptive_rho=False,
            state0=state,
        )
        state = info["state"]
        Xstart = Ys.copy()

    assert np.allclose(Ys, Ym, rtol=1e-10, atol=1e-12)
    