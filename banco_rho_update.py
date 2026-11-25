def banco_rho_update(
    rho, primal, dual, scaled_dual,
    ratio=10.0, factor=2.0,
    dual_floor_boost=False,
):
    old_rho = float(rho)
    rho = old_rho

    if primal > ratio*dual:
        rho *= factor
    elif dual > ratio*primal:
        rho /= factor

    if dual_floor_boost and dual < 1e-8:
        rho = 10.0*old_rho

    if rho != old_rho:
        scaled_dual *= old_rho/rho

    return rho, scaled_dual