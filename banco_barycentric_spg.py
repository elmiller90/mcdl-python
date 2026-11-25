from collections import deque
import numpy as np
from scipy.optimize import OptimizeResult


def projected_gradient(x, g):
    x = np.asarray(x)
    return x - np.maximum(x-g, 0.0)


def solve_spg(
    fg, x0, maxiter=1000, gtol=1e-4,
    memory=10, armijo=1e-4, backtrack=0.5,
    alpha0=1.0, alpha_min=1e-10, alpha_max=1e10,
):
    x = np.maximum(np.asarray(x0, float).ravel(), 0.0)
    f, g = fg(x)
    g = np.asarray(g).ravel()
    nfev, alpha = 1, alpha0
    fmem = deque([float(f)], maxlen=memory)

    for k in range(maxiter):
        pg = projected_gradient(x, g)
        pginf = np.linalg.norm(pg, np.inf)
        if pginf <= gtol:
            return OptimizeResult(
                x=x, fun=f, jac=g, nit=k, nfev=nfev,
                success=True, status=0,
                message="projected-gradient tolerance reached",
                projected_grad_inf=float(pginf),
            )

        d = np.maximum(x-alpha*g, 0.0)-x
        gtd = float(g @ d)

        if gtd >= 0:
            d = np.maximum(x-g, 0.0)-x
            gtd = float(g @ d)

        step, fref = 1.0, max(fmem)

        while True:
            xt = np.maximum(x+step*d, 0.0)
            ft, gt = fg(xt)
            gt = np.asarray(gt).ravel()
            nfev += 1

            if ft <= fref + armijo*step*gtd:
                break

            step *= backtrack
            if step < 1e-16:
                return OptimizeResult(
                    x=x, fun=f, jac=g, nit=k, nfev=nfev,
                    success=False, status=2,
                    message="SPG line search failed",
                    projected_grad_inf=float(pginf),
                )

        s, y = xt-x, gt-g
        sty, sts, yty = s@y, s@s, y@y

        if sty > 1e-16:
            alpha = sts/sty if k % 2 == 0 else sty/max(yty, 1e-16)
            alpha = np.clip(alpha, alpha_min, alpha_max)
        else:
            alpha = alpha_max

        x, f, g = xt, float(ft), gt
        fmem.append(f)

    pginf = np.linalg.norm(projected_gradient(x, g), np.inf)
    return OptimizeResult(
        x=x, fun=f, jac=g, nit=maxiter, nfev=nfev,
        success=pginf <= gtol, status=1,
        message="maximum SPG iterations reached",
        projected_grad_inf=float(pginf),
    )