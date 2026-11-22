import numpy as np

from dictionary_alpha import compute_dictionary_alpha
from dictionary_cg import solve_dictionary_cg
from dictionary_forward_operator import DictionaryForwardOperator
from dictionary_projection import project_dictionary_nonnegative


def cosine_similarity(a, b):
    return np.vdot(a, b).real / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def similarity_matrix(G, Gtrue):
    K = G.shape[1]
    C = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            C[i, j] = cosine_similarity(G[:, i], Gtrue[:, j])
    return C


def main():
    rng = np.random.default_rng(20)

    N, M, K, T = 64, 9, 2, 12
    scales = [(1, 1), (2, 1), (1, 2)]
    J = len(scales)
    L = N + M - 1
    center = (M + 1) // 2

    # Two distinct nonnegative base atoms.
    x = np.arange(M) - (M - 1) / 2

    g1 = np.exp(
        -0.5 * (x / 1.2) ** 2
    )

    g2 = (
        0.75 * np.exp(
            -0.5 * ((x + 1.7) / 0.8) ** 2
        )
        + 0.35 * np.exp(
            -0.5 * ((x - 2.0) / 0.9) ** 2
        )
    )

    Gtrue = np.column_stack((g1, g2))
    Gtrue = project_dictionary_nonnegative(Gtrue)

    alpha_true, _ = compute_dictionary_alpha(
        Gtrue,
        scales,
        center,
        N,
    )

    # Exact known sparse coefficients.
    X = np.zeros((L, K, J, T))

    for t in range(T):
        for k in range(K):
            for j in range(J):
                p = rng.integers(M - 1, L)
                X[p, k, j, t] = rng.uniform(0.5, 1.5)

    # Generate exact noiseless observations.
    true_op = DictionaryForwardOperator(
        X=X,
        alpha=alpha_true,
        scales=scales,
        center=center,
        N=N,
        M=M,
    )

    S = true_op.forward(Gtrue)

    # Independent feasible initialization.
    G = rng.uniform(
        0.0, 1.0, size=(M, K)
    )
    G = project_dictionary_nonnegative(G)

    H = np.zeros_like(G)

    rho = 1.0
    maxiter = 150
    tol = 1e-8

    print("Known-X two-atom recovery")
    print("-------------------------")

    for it in range(maxiter):
        alpha, _ = compute_dictionary_alpha(
            G,
            scales,
            center,
            N,
        )

        op = DictionaryForwardOperator(
            X=X,
            alpha=alpha,
            scales=scales,
            center=center,
            N=N,
            M=M,
        )

        D, info = solve_dictionary_cg(
            op=op,
            S=S,
            G=G,
            H=H,
            rho=rho,
            D0=G,
            rtol=1e-10,
            maxiter=500,
        )

        if info != 0:
            raise RuntimeError(
                f"CG failed at iteration {it}: info={info}"
            )

        # DRelaxParam = 1.
        Dr = D

        Gnew = project_dictionary_nonnegative(
            Dr + H
        )

        H = H + Dr - Gnew

        primal = np.linalg.norm(
            Dr - Gnew
        )

        change = np.linalg.norm(
            Gnew - G
        )

        G = Gnew

        # Recompute alpha from projected G.
        alpha_eval, _ = compute_dictionary_alpha(
            G,
            scales,
            center,
            N,
        )

        eval_op = DictionaryForwardOperator(
            X=X,
            alpha=alpha_eval,
            scales=scales,
            center=center,
            N=N,
            M=M,
        )

        residual = (
            np.linalg.norm(
                eval_op.forward(G) - S
            )
            / max(np.linalg.norm(S), 1e-15)
        )

        C = similarity_matrix(G, Gtrue)
        diag_min = np.min(np.diag(C))

        if it == 0 or (it + 1) % 10 == 0:
            print(
                f"{it + 1:3d} "
                f"res={residual:.3e} "
                f"pri={primal:.3e} "
                f"dG={change:.3e} "
                f"cosmin={diag_min:.8f}"
            )

        if (
            residual < tol
            and primal < tol
            and change < tol
        ):
            break

    C = similarity_matrix(G, Gtrue)

    alpha_final, _ = compute_dictionary_alpha(
        G,
        scales,
        center,
        N,
    )

    final_op = DictionaryForwardOperator(
        X=X,
        alpha=alpha_final,
        scales=scales,
        center=center,
        N=N,
        M=M,
    )

    final_residual = (
        np.linalg.norm(
            final_op.forward(G) - S
        )
        / np.linalg.norm(S)
    )

    print()
    print(f"iterations:        {it + 1}")
    print(f"final residual:    {final_residual:.3e}")
    print("similarity matrix:")
    print(C)
    print("true dictionary:")
    print(Gtrue)
    print("recovered dictionary:")
    print(G)

    assert final_residual < 1e-5
    assert np.min(np.diag(C)) > 0.99999


if __name__ == "__main__":
    main()