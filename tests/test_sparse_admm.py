import numpy as np

from sparse_admm import soft_threshold


def test_soft_threshold():
    x = np.array(
        [-3.0, -1.0, -0.2, 0.0, 0.2, 1.0, 3.0]
    )

    expected = np.array(
        [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]
    )

    result = soft_threshold(
        x,
        threshold=1.0,
    )

    error = np.linalg.norm(
        result - expected
    )

    print()
    print("Soft threshold test")
    print("-------------------")
    print("input:   ", x)
    print("result:  ", result)
    print("expected:", expected)
    print("error:", error)

    assert error == 0.0


def test_nonnegative_threshold():
    x = np.array(
        [-3.0, -1.0, 0.2, 1.0, 3.0]
    )

    expected = np.array(
        [0.0, 0.0, 0.0, 0.0, 2.0]
    )

    result = soft_threshold(
        x,
        threshold=1.0,
        nonnegative=True,
    )

    error = np.linalg.norm(
        result - expected
    )

    print()
    print("Nonnegative threshold test")
    print("--------------------------")
    print("result:  ", result)
    print("expected:", expected)
    print("error:", error)

    assert error == 0.0


if __name__ == "__main__":
    test_soft_threshold()
    test_nonnegative_threshold()