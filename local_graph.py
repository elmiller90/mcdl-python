from dataclasses import dataclass

import numpy as np


@dataclass
class LocalTransportGraph:
    L: int
    J: int
    scales: np.ndarray
    shift_radius: int
    scale_radius: int
    shift_velocity: float
    scale_velocity: float
    src: np.ndarray
    dst: np.ndarray
    cost: np.ndarray

    @property
    def P(self):
        return self.L * self.J

    @property
    def E(self):
        return self.src.size


def state_index(n, j, J):
    return n * J + j


def state_coordinates(index, J):
    return index // J, index % J


def median_log_scale_spacing(scales):
    scales = np.asarray(scales, dtype=np.float64)

    if np.any(scales <= 0):
        raise ValueError("All scales must be positive.")

    if scales.size < 2:
        return 1.0

    spacing = np.abs(
        np.diff(np.log(scales))
    )

    positive = spacing[spacing > 0]

    if positive.size == 0:
        return 1.0

    return float(np.median(positive))


def build_local_transport_graph(
    L,
    scales,
    shift_radius=3,
    scale_radius=1,
    shift_velocity=1.0,
    scale_velocity=None,
):
    scales = np.asarray(scales, dtype=np.float64)
    J = scales.size

    if L <= 0:
        raise ValueError("L must be positive.")

    if J <= 0:
        raise ValueError("At least one scale is required.")

    if np.any(scales <= 0):
        raise ValueError("All scales must be positive.")

    if shift_radius < 0 or scale_radius < 0:
        raise ValueError("Radii must be nonnegative.")

    if shift_velocity <= 0:
        raise ValueError("shift_velocity must be positive.")

    if scale_velocity is None:
        scale_velocity = median_log_scale_spacing(scales)

    if scale_velocity <= 0:
        raise ValueError("scale_velocity must be positive.")

    src = []
    dst = []
    cost = []

    log_scales = np.log(scales)

    for n in range(L):
        for j in range(J):
            source = state_index(n, j, J)

            n0 = max(0, n - shift_radius)
            n1 = min(L - 1, n + shift_radius)

            j0 = max(0, j - scale_radius)
            j1 = min(J - 1, j + scale_radius)

            for nn in range(n0, n1 + 1):
                for jj in range(j0, j1 + 1):
                    target = state_index(nn, jj, J)

                    dn = (nn - n) / shift_velocity

                    ds = (
                        log_scales[jj]
                        - log_scales[j]
                    ) / scale_velocity

                    c = dn * dn + ds * ds

                    src.append(source)
                    dst.append(target)
                    cost.append(c)

    return LocalTransportGraph(
        L=int(L),
        J=int(J),
        scales=scales.copy(),
        shift_radius=int(shift_radius),
        scale_radius=int(scale_radius),
        shift_velocity=float(shift_velocity),
        scale_velocity=float(scale_velocity),
        src=np.asarray(src, dtype=np.int64),
        dst=np.asarray(dst, dtype=np.int64),
        cost=np.asarray(cost, dtype=np.float64),
    )


def transpose_edges(graph):
    return (
        graph.dst.copy(),
        graph.src.copy(),
        graph.cost.copy(),
    )


def outgoing_edge_counts(graph):
    return np.bincount(
        graph.src,
        minlength=graph.P,
    )


def incoming_edge_counts(graph):
    return np.bincount(
        graph.dst,
        minlength=graph.P,
    )