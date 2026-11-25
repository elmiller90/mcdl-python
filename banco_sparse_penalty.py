import numpy as np


def banco_log_shrink(v, threshold, a, nonnegative=True):
    v = np.asarray(v, dtype=float)

    y1 = np.abs(v)/2.0 - 1.0/(2.0*a)
    y2 = np.abs(v)/2.0 + 1.0/(2.0*a)
    disc = np.maximum(y2*y2 - threshold/a, 0.0)

    out = np.sign(v)*(y1 + np.sqrt(disc))
    out[np.abs(v) < threshold] = 0.0

    if nonnegative:
        out = np.maximum(out, 0.0)

    return out


def banco_sparse_update(
    v, lambda_s, rho, iteration,
    l1_iters=20, nonnegative=True,
):
    if lambda_s <= 0 or rho <= 0:
        raise ValueError("lambda_s and rho must be positive")
    if iteration <= 0:
        raise ValueError("iteration must be positive")

    threshold = lambda_s/rho

    if iteration <= l1_iters:
        if nonnegative:
            return np.maximum(v-threshold, 0.0)
        return np.sign(v)*np.maximum(np.abs(v)-threshold, 0.0)

    a = 0.95/lambda_s
    return banco_log_shrink(
        v, threshold, a,
        nonnegative=nonnegative,
    )


def banco_sparse_penalty(
    y, lambda_s, iteration, l1_iters=20,
):
    y = np.asarray(y, dtype=float)

    if iteration <= l1_iters:
        return float(np.sum(np.abs(y)))

    a = 0.95/lambda_s
    return float(np.sum(np.log1p(a*np.abs(y))/a))