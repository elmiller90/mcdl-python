import numpy as np

from banco_sparse_penalty import (
    banco_log_shrink,
    banco_sparse_penalty,
    banco_sparse_update,
)


def test_first_20_iterations_use_l1():
    v = np.array([-1.0, 0.05, 0.2, 1.0])
    expected = np.array([0.0, 0.0, 0.1, 0.9])

    for iteration in (1, 10, 20):
        got = banco_sparse_update(
            v, lambda_s=0.1, rho=1.0,
            iteration=iteration,
        )
        assert np.allclose(got, expected)


def test_iteration_21_switches_to_log():
    v = np.array([0.05, 0.1, 0.2, 1.0])
    lambda_s, rho = 0.1, 1.0
    threshold = lambda_s/rho
    a = 0.95/lambda_s

    expected = banco_log_shrink(
        v, threshold, a,
        nonnegative=True,
    )
    got = banco_sparse_update(
        v, lambda_s, rho,
        iteration=21,
    )

    assert np.allclose(got, expected)
    assert got[0] == 0.0
    assert got[-1] > 0.0


def test_penalty_switch():
    y = np.array([0.0, 0.25, 1.0])
    lambda_s = 0.1

    p20 = banco_sparse_penalty(
        y, lambda_s, iteration=20
    )
    p21 = banco_sparse_penalty(
        y, lambda_s, iteration=21
    )

    a = 0.95/lambda_s

    assert np.isclose(p20, np.sum(np.abs(y)))
    assert np.isclose(
        p21,
        np.sum(np.log1p(a*np.abs(y))/a),
    )


def test_invalid_parameters():
    v = np.ones(3)

    for kwargs in (
        dict(lambda_s=0.0, rho=1.0, iteration=1),
        dict(lambda_s=0.1, rho=0.0, iteration=1),
        dict(lambda_s=0.1, rho=1.0, iteration=0),
    ):
        try:
            banco_sparse_update(v, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")