import numpy as np

from zero_padded_convolution import (
    ZeroPaddedConvolutionOperator,
)


def test_shapes():
    rng = np.random.default_rng(60)

    N = 80
    M = 41
    F = 3
    T = 5

    L = N + M - 1

    D = rng.normal(
        size=(N, F)
    )

    X = rng.normal(
        size=(L, F, T)
    )

    A = ZeroPaddedConvolutionOperator(
        D=D,
        M=M,
    )

    Z = A.forward_padded(X)
    S = A.forward(X)

    R = rng.normal(
        size=(N, T)
    )

    AtR = A.adjoint(R)

    print()
    print("Zero-padded shape test")
    print("----------------------")
    print("N:", N)
    print("M:", M)
    print("L:", L)
    print("D shape:   ", D.shape)
    print("Dpad shape:", A.D_padded.shape)
    print("X shape:   ", X.shape)
    print("Z shape:   ", Z.shape)
    print("S shape:   ", S.shape)
    print("A*R shape: ", AtR.shape)


def test_crop():
    """
    Verify that forward() returns exactly the final N
    samples of forward_padded().
    """
    rng = np.random.default_rng(61)

    N = 20
    M = 7
    F = 2
    T = 3

    L = N + M - 1

    D = rng.normal(
        size=(N, F)
    )

    X = rng.normal(
        size=(L, F, T)
    )

    A = ZeroPaddedConvolutionOperator(
        D=D,
        M=M,
    )

    Z = A.forward_padded(X)
    S = A.forward(X)

    error = np.linalg.norm(
        S - Z[M - 1:, :]
    )

    print()
    print("Pre-crop test")
    print("-------------")
    print("error:", error)


def test_adjoint():
    rng = np.random.default_rng(62)

    N = 80
    M = 41
    F = 5
    T = 4

    L = N + M - 1

    D = rng.normal(
        size=(N, F)
    )

    X = rng.normal(
        size=(L, F, T)
    )

    R = rng.normal(
        size=(N, T)
    )

    A = ZeroPaddedConvolutionOperator(
        D=D,
        M=M,
    )

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

    relative_error = abs(
        lhs - rhs
    ) / max(
        1.0,
        abs(lhs),
        abs(rhs),
    )

    print()
    print("Zero-padded adjoint test")
    print("------------------------")
    print(f"<AX,R>  = {lhs:.12e}")
    print(f"<X,A*R> = {rhs:.12e}")
    print(
        f"relative error = "
        f"{relative_error:.3e}"
    )

    assert relative_error < 1e-12


def test_impulse():
    """
    Diagnostic impulse test for understanding the extended
    coefficient-coordinate convention.
    """
    N = 20
    M = 7
    F = 1
    T = 1

    L = N + M - 1

    D = np.zeros(
        (N, F)
    )

    # Dictionary impulse at physical index zero.
    D[0, 0] = 1.0

    X = np.zeros(
        (L, F, T)
    )

    coefficient_index = M - 1 + 5
    X[coefficient_index, 0, 0] = 1.0

    A = ZeroPaddedConvolutionOperator(
        D=D,
        M=M,
    )

    Z = A.forward_padded(X)
    S = A.forward(X)

    padded_peak = int(
        np.argmax(np.abs(Z[:, 0]))
    )

    physical_peak = int(
        np.argmax(np.abs(S[:, 0]))
    )

    print()
    print("Impulse-coordinate test")
    print("-----------------------")
    print(
        "coefficient index:",
        coefficient_index,
    )
    print(
        "padded peak:      ",
        padded_peak,
    )
    print(
        "physical peak:    ",
        physical_peak,
    )
    print(
        "expected physical:",
        5,
    )


if __name__ == "__main__":
    test_shapes()
    test_crop()
    test_adjoint()
    test_impulse()
