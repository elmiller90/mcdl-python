import numpy as np

from convolution_operator import ConvolutionOperator


def test_impulse():

    N = 64
    F = 1
    T = 1

    D = np.zeros((N, F))
    D[0, 0] = 1.0

    X = np.zeros((N, F, T))
    X[17, 0, 0] = 1.0

    A = ConvolutionOperator(D)

    S = A.forward(X)

    error = np.linalg.norm(
        S[:, 0] - X[:, 0, 0]
    )

    print()
    print("Impulse convolution test")
    print("------------------------")
    print("error:", error)


def test_shift():

    N = 64
    F = 1
    T = 1

    D = np.zeros((N, F))
    D[3, 0] = 1.0

    X = np.zeros((N, F, T))
    X[10, 0, 0] = 1.0

    A = ConvolutionOperator(D)

    S = A.forward(X)

    peak = int(np.argmax(S[:, 0]))

    print()
    print("Shift convolution test")
    print("----------------------")
    print("expected peak:", 13)
    print("actual peak:  ", peak)


def test_adjoint():

    rng = np.random.default_rng(20)

    N = 117
    F = 7
    T = 5

    D = rng.normal(
        size=(N, F)
    )

    X = rng.normal(
        size=(N, F, T)
    )

    R = rng.normal(
        size=(N, T)
    )

    A = ConvolutionOperator(D)

    AX = A.forward(X)
    AtR = A.adjoint(R)

    lhs = np.vdot(
        AX,
        R,
    ).real

    rhs = np.vdot(
        X,
        AtR,
    ).real

    error = abs(
        lhs - rhs
    ) / max(
        1.0,
        abs(lhs),
        abs(rhs),
    )

    print()
    print("Convolution adjoint test")
    print("------------------------")
    print(f"<AX,R>  = {lhs:.12e}")
    print(f"<X,A*R> = {rhs:.12e}")
    print(f"relative error = {error:.3e}")


if __name__ == "__main__":
    test_impulse()
    test_shift()
    test_adjoint()