import numpy as np

from log_admm import (
    continuation_a,
    log_penalty,
    log_threshold_nonnegative,
)


def test_threshold_limit():
    x = np.array([-1.0, 0.0, 0.2, 0.5, 1.0, 3.0])
    threshold = 0.5

    y_log = log_threshold_nonnegative(x, threshold, a=1e-8)
    y_l1 = np.maximum(x - threshold, 0.0)

    error = np.linalg.norm(y_log - y_l1)

    print("L1-limit error:", error)
    assert error < 1e-6


def test_threshold():
    x = np.array([-1.0, 0.2, 0.5, 1.0, 3.0])
    y = log_threshold_nonnegative(x, threshold=0.5, a=0.8)

    print("input: ", x)
    print("output:", y)

    assert np.all(y >= 0)
    assert y[0] == 0
    assert y[1] == 0
    assert np.all(np.isfinite(y))


def test_continuation():
    lambda_s = 0.03
    values = [
        continuation_a(i, lambda_s)
        for i in [0, 25, 50, 75, 100, 150]
    ]

    print("a values:", values)

    assert values[0] == 1e-4
    assert values[-1] == values[-2]
    assert np.all(np.diff(values[:5]) > 0)


def test_penalty():
    x = np.array([0.0, 1.0, 2.0])
    value = log_penalty(x, lambda_s=0.03, a=0.5)

    print("penalty:", value)
    assert np.isfinite(value)
    assert value > 0


if __name__ == "__main__":
    test_threshold_limit()
    test_threshold()
    test_continuation()
    test_penalty()
