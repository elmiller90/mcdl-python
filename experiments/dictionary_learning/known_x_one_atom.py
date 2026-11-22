import numpy as np

from dictionary_alpha import compute_dictionary_alpha
from dictionary_cg import solve_dictionary_cg
from dictionary_forward_operator import DictionaryForwardOperator
from dictionary_projection import project_dictionary_nonnegative


def cosine_similarity(a, b):
    return np.vdot(a, b).real / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def main():
    rng = np.random.default_rng(10)

    # Small, controlled known-X problem.
    N, M, K, T = 64,  9, 1, 8
    scales = [(1, 1), (2, 1), (1, 2)]
    J = len(scales)
    L = N + M - 1
    center = (M + 1) // 2

    # True nonnegative unit-norm base atom.
    x = np.arange(M) - (M - 1) / 2
    Gtrue = np.exp(-0.5 * (x / 1.4) ** 2)[:, None]
    Gtrue = project_dictionary_nonnegative(Gtrue)

    alpha_true, _ = compute_dictionary_alpha(
        Gtrue, scales, center, N
    )

    # Fixed known coefficients: sparse impulses with varying signs/amplitudes.
    X = np.zeros((L, K, J, T))

    for t in range(T):
        for j in range(J):
            p = rng.integers(M - 1, L)
            X[p, 0, j, t] = rng.uniform(0.5, 1.5)

    # Generate exact noiseless data.
    true_op = DictionaryForwardOperator(
        X=X,
        alpha=alpha_true,
        scales=scales,
        center=center,
        N=N,
        M=M,
    )

    S = true_op.forward(Gtrue)

    # Deliberately different feasible initialization.
    G = rng.uniform(0.0, 1.0, size=(M, K))
    G = project_dictionary_nonnegative(G)

    H = np.zeros_like(G)
    rho = 1.0

    maxiter = 100
    tol = 1e-8

    print("Known-X one-atom recovery")
    print("-------------------------")

    for it in range(maxiter):
        alpha, _ = compute_dictionary_alpha(
            G, scales, center, N
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

        # DRelaxParam = 1 for this first experiment.
        Dr = D

        Gnew = project_dictionary_nonnegative(
            Dr + H
        )

        H = H + Dr - Gnew

        primal = np.linalg.norm(Dr - Gnew)
        change = np.linalg.norm(Gnew - G)
        similarity = cosine_similarity(
            Gnew[:, 0], Gtrue[:, 0]
        )

        G = Gnew

        # Alpha is defined by projected G.
        alpha_eval, _ = compute_dictionary_alpha(
            G, scales, center, N
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
            np.linalg.norm(eval_op.forward(G) - S)
            / max(np.linalg.norm(S), 1e-15)
        )

        if it == 0 or (it + 1) % 10 == 0:
            print(
                f"{it + 1:3d} "
                f"res={residual:.3e} "
                f"pri={primal:.3e} "
                f"dG={change:.3e} "
                f"cos={similarity:.8f}"
            )

        if (
            residual < tol
            and primal < tol
            and change < tol
        ):
            break

    final_similarity = cosine_similarity(
        G[:, 0], Gtrue[:, 0]
    )

    final_alpha, _ = compute_dictionary_alpha(
        G, scales, center, N
    )

    final_op = DictionaryForwardOperator(
        X=X,
        alpha=final_alpha,
        scales=scales,
        center=center,
        N=N,
        M=M,
    )

    final_residual = (
        np.linalg.norm(final_op.forward(G) - S)
        / np.linalg.norm(S)
    )

    print()
    print(f"iterations:        {it + 1}")
    print(f"final residual:    {final_residual:.3e}")
    print(f"atom similarity:   {final_similarity:.10f}")
    print("true atom:")
    print(Gtrue[:, 0])
    print("recovered atom:")
    print(G[:, 0])

    assert final_residual < 1e-5
    assert final_similarity > 0.999999


if __name__ == "__main__":
    main()