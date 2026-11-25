import numpy as np
from banco_barycentric_spg import solve_spg


def test_spg_quadratic():
    rng = np.random.default_rng(170)
    d = 0.2+2*rng.random(200)
    b = rng.standard_normal(200)

    def fg(x):
        return float(0.5*np.dot(d*x, x)-b@x), d*x-b

    truth = np.maximum(b/d, 0)
    r = solve_spg(
        fg, np.zeros(200),
        maxiter=1000, gtol=1e-8,
    )

    assert r.success
    assert r.projected_grad_inf < 1e-8
    assert np.allclose(r.x, truth, rtol=1e-6, atol=1e-7)