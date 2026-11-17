import numpy as np

from multiscale_operator import (
    lowpass_m,
    upsample_center,
    downsample_center,
    resample_atom,
    resample_atom_adjoint,
)

def test_lowpass():
    x = np.zeros(9)
    x[4] = 1.0

    y = lowpass_m(x, 3)

    print()
    print("lowpass m=3")
    print("-------------")
    print(y)


def test_identity():
    M = 71
    N = 201
    center = 36

    d = np.zeros(M)
    d[15:56] = np.concatenate([
        np.arange(1, 21),
        [21],
        np.arange(20, 0, -1),
    ])

    out, shift = resample_atom(
        d,
        1,
        1,
        center,
        N,
    )

    error = np.linalg.norm(
        out[:M] - d
    )

    print()
    print("Identity scale")
    print("--------------")
    print("shift:", shift)
    print("error:", error)


def test_scales():

    M = 71
    N = 201
    center = 36

    d = np.zeros(M)

    d[15:56] = np.concatenate([
        np.arange(1, 21),
        [21],
        np.arange(20, 0, -1),
    ])

    scales = [
        (1, 1),
        (1, 2),
        (2, 3),
        (3, 2),
        (2, 1),
    ]

    print()
    print("Scale tests")
    print("-----------")

    for c1, c2 in scales:

        out, shift = resample_atom(
            d,
            c1,
            c2,
            center,
            N,
        )

        print(
            f"{c1}/{c2}: "
            f"norm={np.linalg.norm(out):.8f}, "
            f"shift={shift}, "
            f"peak={np.argmax(out)}"
        )

def test_resample_adjoint():

    rng = np.random.default_rng(10)

    M = 71
    N = 201
    center = 36

    scales = [
        (1, 1),
        (1, 2),
        (2, 3),
        (3, 2),
        (2, 1),
    ]

    print()
    print("Resampling adjoint test")
    print("-----------------------")
    print(
        "scale      <Rd,z>          "
        "<d,R*z>         rel error"
    )

    for c1, c2 in scales:

        d = rng.normal(size=M)
        z = rng.normal(size=N)

        Rd, _ = resample_atom(
            d,
            c1,
            c2,
            center,
            N,
        )

        Rtz = resample_atom_adjoint(
            z,
            M,
            c1,
            c2,
            center,
            N,
        )

        lhs = np.dot(Rd, z)
        rhs = np.dot(d, Rtz)

        error = abs(lhs - rhs) / max(
            1.0,
            abs(lhs),
            abs(rhs),
        )

        print(
            f"{c1}/{c2}"
            f"    {lhs: .10e}"
            f"    {rhs: .10e}"
            f"    {error:.3e}"
        )


if __name__ == "__main__":
    test_lowpass()
    test_identity()
    test_scales()
    test_resample_adjoint()

