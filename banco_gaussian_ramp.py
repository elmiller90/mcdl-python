import numpy as np

BANCO_SCALES = [
    (1, 5), (1, 4), (1, 3), (2, 5), (1, 2),
    (3, 5), (2, 3), (3, 4), (4, 5), (1, 1),
]


def make_banco_dictionary(M=41):
    center = (M + 1) // 2       # MATLAB 1-based
    x = np.arange(1, M + 1)

    g = np.exp(-0.5*((x-center)/8.0)**2)
    g /= np.linalg.norm(g)

    r = np.zeros(M)
    width = 30
    i1 = center - width//2      # MATLAB index 6
    i2 = i1 + width - 1         # MATLAB index 35
    vals = np.linspace(0.0, 10.0, width)
    r[i1-1:i2] = vals/np.linalg.norm(vals)

    return np.column_stack((g, r))


def banco_trajectories(T=80, M=41, N=80, J=10):
    tt = np.linspace(-0.25*np.pi, 1.25*np.pi, T)

    # MATLAB round is half-away-from-zero.
    sig = np.floor(
        (J+1)/2 + (J-1)/2*np.cos(tt) + 0.5
    ).astype(int)

    width = np.floor(
        (J+1)/2 - (J-1)/2*np.cos(tt) + 0.5
    ).astype(int)

    tt = np.linspace(0.0, np.pi, T) - np.pi/4

    p1 = np.floor(
        1.5*(M-1) - 20*np.cos(tt) + 0.5
    ).astype(int)

    p2 = np.floor(
        1.5*(M-1) + 20*np.cos(tt) + 0.5
    ).astype(int)

    L = N + M - 1

    # MATLAB wrap_shift(x,L)=mod(x-1,L)+1,
    # converted to zero-based Python.
    p1 = (p1 - 1) % L
    p2 = (p2 - 1) % L

    # MATLAB scale indices 1..J -> Python 0..J-1.
    return p1, p2, sig-1, width-1


def make_banco_coefficients(T=80, M=41, N=80):
    J = len(BANCO_SCALES)
    L = N + M - 1

    p1, p2, j1, j2 = banco_trajectories(
        T=T, M=M, N=N, J=J
    )

    X = np.zeros((L, 2, J, T), dtype=float)

    for t in range(T):
        X[p1[t], 0, j1[t], t] = 1.0
        X[p2[t], 1, j2[t], t] = 1.0

    return X