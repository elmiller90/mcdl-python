import numpy as np

from gaussian_experiment import (
    SCALES, add_noise_snr, make_dictionary, make_true_coefficients,
)
from log_admm import solve_log_admm as solve_original
from log_admm_stateful_compat import solve_log_admm_stateful
from multiscale_convolution import MultiscaleConvolutionOperator


def main():
    rng = np.random.default_rng(100)
    M, N, T = 41, 48, 7

    D = make_dictionary(M, 4.0)
    A = MultiscaleConvolutionOperator(
        D=D,
        scales=SCALES,
        center=(M + 1) // 2,
        N=N,
        normalize=True,
    )

    Xtrue, _ = make_true_coefficients(
        L=A.L, M=M, J=len(SCALES), T=T
    )
    Sclean = A.forward(Xtrue)
    Snoisy, _, _ = add_noise_snr(Sclean, 14.0, rng)

    X0 = np.maximum(A.adjoint(Snoisy, structured=True), 0.0)
    if X0.max() > 0:
        X0 /= X0.max()

    kwargs = dict(
        X0=X0,
        S=Snoisy,
        operator=A,
        lambda_s=.15,
        lambda2=0.0,
        tau=1e-2,
        rho=.1,
        max_admm_iter=120,
        x_maxiter=30,
        abs_tol=1e-4,
        rel_tol=1e-3,
        adaptive_rho=True,
        rho_mu=10,
        rho_tau=2,
        a_min=1e-4,
        continuation_iters=100,
        a_fraction=.95,
        verbose=False,
    )

    X1, Y1, U1, i1 = solve_original(**kwargs)
    X2, Y2, U2, i2 = solve_log_admm_stateful(**kwargs)

    def relerr(a, b):
        return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12)

    print("Cold-start compatibility")
    print("------------------------")
    print("X relerr:", relerr(X1, X2))
    print("Y relerr:", relerr(Y1, Y2))
    print("U relerr:", relerr(U1, U2))
    print("iterations:", i1["iterations"], i2["iterations"])
    print("rho:", i1["rho"], i2["rho"])
    print("a:", i1["a"], i2["a"])

    assert np.allclose(X1, X2, rtol=1e-11, atol=1e-12)
    assert np.allclose(Y1, Y2, rtol=1e-11, atol=1e-12)
    assert np.allclose(U1, U2, rtol=1e-11, atol=1e-12)
    assert i1["iterations"] == i2["iterations"]
    assert np.isclose(i1["rho"], i2["rho"])
    assert np.isclose(i1["a"], i2["a"])

    state = i2["state"]
    assert state["X"].shape == X0.shape
    assert state["Y"].shape == X0.shape
    assert state["U"].shape == X0.shape
    assert state["rho"] > 0
    assert state["a"] > 0

    print("state payload: PASS")
    print("COLD-START COMPATIBILITY: PASS")


if __name__ == "__main__":
    main()
