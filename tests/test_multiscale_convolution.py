import numpy as np

from multiscale_convolution import (
    MultiscaleConvolutionOperator,
)


def make_dictionary(M, K):
    x = np.arange(
        M,
        dtype=np.float64,
    )

    D = np.zeros(
        (M, K),
        dtype=np.float64,
    )

    centers = [22.0, 46.0]
    widths = [4.0, 7.0]

    for k in range(K):
        D[:, k] = np.exp(
            -0.5
            * (
                (x - centers[k])
                / widths[k]
            ) ** 2
        )

        D[:, k] /= np.linalg.norm(
            D[:, k]
        )

    return D


def test_dictionary():
    M = 71
    N = 201
    K = 2
    center = 36

    scales = [
        (1, 1),
        (1, 2),
        (2, 3),
        (3, 2),
        (2, 1),
    ]

    D = make_dictionary(
        M,
        K,
    )

    A = MultiscaleConvolutionOperator(
        D=D,
        scales=scales,
        center=center,
        N=N,
        normalize=True,
    )

    print()
    print("Scaled dictionary test")
    print("----------------------")
    print("M:", A.M)
    print("N:", A.N)
    print("L:", A.L)

    print(
        "base shape:  ",
        D.shape,
    )

    print(
        "scaled shape:",
        A.D_scaled.shape,
    )

    print(
        "padded shape:",
        A.conv.D_padded.shape,
    )

    for k in range(K):
        for j, scale in enumerate(
            scales
        ):
            f = A.channel(
                k,
                j,
            )

            norm = np.linalg.norm(
                A.D_scaled[:, f]
            )

            peak = int(
                np.argmax(
                    A.D_scaled[:, f]
                )
            )

            print(
                f"k={k}, "
                f"scale={scale[0]}/{scale[1]}, "
                f"norm={norm:.12f}, "
                f"shift={A.shifts[k,j]:d}, "
                f"peak={peak}"
            )


def test_shapes():
    rng = np.random.default_rng(70)

    M = 71
    N = 201
    K = 2
    T = 4
    center = 36

    scales = [
        (1, 1),
        (2, 3),
        (1, 2),
    ]

    J = len(scales)
    L = N + M - 1

    D = make_dictionary(
        M,
        K,
    )

    A = MultiscaleConvolutionOperator(
        D=D,
        scales=scales,
        center=center,
        N=N,
    )

    X = rng.normal(
        size=(L, K, J, T)
    )

    S = A.forward(X)

    Z = A.forward_padded(X)

    X_adj = A.adjoint(S)

    print()
    print("Shape test")
    print("----------")

    print(
        "X shape:       ",
        X.shape,
    )

    print(
        "padded S shape:",
        Z.shape,
    )

    print(
        "S shape:       ",
        S.shape,
    )

    print(
        "A*S shape:     ",
        X_adj.shape,
    )


def test_crop():
    rng = np.random.default_rng(71)

    M = 41
    N = 80
    K = 1
    T = 3
    center = 21

    scales = [
        (1, 2),
        (1, 1),
        (1, 2),
    ]

    J = len(scales)
    L = N + M - 1

    D = np.zeros(
        (M, K),
        dtype=np.float64,
    )

    D[M // 2, 0] = 1.0

    A = MultiscaleConvolutionOperator(
        D=D,
        scales=scales,
        center=center,
        N=N,
    )

    X = rng.normal(
        size=(L, K, J, T)
    )

    Z = A.forward_padded(X)
    S = A.forward(X)

    error = np.linalg.norm(
        S - Z[M - 1:, :]
    )

    print()
    print("Composite crop test")
    print("-------------------")
    print("error:", error)


def test_adjoint():
    rng = np.random.default_rng(72)

    M = 71
    N = 201
    K = 2
    T = 5
    center = 36

    scales = [
        (1, 1),
        (2, 3),
        (1, 2),
    ]

    J = len(scales)
    L = N + M - 1

    D = make_dictionary(
        M,
        K,
    )

    A = MultiscaleConvolutionOperator(
        D=D,
        scales=scales,
        center=center,
        N=N,
        normalize=True,
    )

    X = rng.normal(
        size=(L, K, J, T)
    )

    R = rng.normal(
        size=(N, T)
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

    error = abs(
        lhs - rhs
    ) / max(
        1.0,
        abs(lhs),
        abs(rhs),
    )

    print()
    print(
        "Zero-padded multiscale adjoint test"
    )
    print("-----------------------------------")

    print(
        f"<AX,R>  = {lhs:.12e}"
    )

    print(
        f"<X,A*R> = {rhs:.12e}"
    )

    print(
        f"relative error = {error:.3e}"
    )

    assert error < 1e-12


if __name__ == "__main__":
    test_dictionary()
    test_shapes()
    test_crop()
    test_adjoint()
