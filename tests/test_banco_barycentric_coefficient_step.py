import numpy as np

from barycentric_transition import (
    make_barycentric_transition_regularizer,
)
from multiscale_convolution import MultiscaleConvolutionOperator


def test_barycentric_smooth_objective_gradient():
    rng = np.random.default_rng(140)

    M, N, K, J, T = 11, 24, 1, 3, 4
    scales = [(1, 2), (3, 4), (1, 1)]

    d = rng.random((M, K))
    d /= np.linalg.norm(d, axis=0)

    A = MultiscaleConvolutionOperator(
        D=d,
        scales=scales,
        center=(M+1)//2,
        N=N,
        normalize=True,
    )

    shape = (A.L, K, J, T)

    # Interior positive point for finite differences.
    X = 0.05 + 0.1*rng.random(shape)
    Y = 0.05 + 0.1*rng.random(shape)
    U = 0.01*rng.standard_normal(shape)
    S = rng.standard_normal((N, T))

    shift_coordinates = np.arange(A.L, dtype=float)
    scale_coordinates = np.array(
        [c1/c2 for c1, c2 in scales],
        dtype=float,
    )

    temporal = make_barycentric_transition_regularizer(
        shift_coordinates=shift_coordinates,
        scale_coordinates=scale_coordinates,
        gate_mass=0.05,
        mass_floor=1e-10,
        shift_velocity=1.0,
        scale_velocity=None,
        spread_weight=0.10,
    )

    rho = 0.7
    lambda_t = 3e-5

    def value_grad(Z):
        R = A.forward(Z)-S
        Q = Z-Y+U
        rt, gt = temporal(Z)

        value = (
            0.5*np.sum(R*R)
            + 0.5*rho*np.sum(Q*Q)
            + lambda_t*rt
        )

        grad = (
            A.adjoint(R, structured=True)
            + rho*Q
            + lambda_t*gt
        )

        return float(value), grad

    value, grad = value_grad(X)

    D = rng.standard_normal(shape)
    D /= np.linalg.norm(D)

    analytical = float(np.sum(grad*D))

    print(f"value: {value:.8e}")
    print(f"analytical: {analytical:.8e}")

    best = np.inf

    for h in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
        fp, _ = value_grad(X+h*D)
        fm, _ = value_grad(X-h*D)

        fd = (fp-fm)/(2*h)

        rel = abs(fd-analytical) / max(
            abs(fd), abs(analytical), 1e-12
        )

        best = min(best, rel)

        print(
            f"h={h:.1e} "
            f"fd={fd:.8e} "
            f"relerr={rel:.3e}"
        )

    assert best < 1e-7