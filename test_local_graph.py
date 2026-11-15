import numpy as np

from local_graph import (
    build_local_transport_graph,
    incoming_edge_counts,
    median_log_scale_spacing,
    outgoing_edge_counts,
    state_coordinates,
    state_index,
    transpose_edges,
)


SCALES = np.array([
    1.0,
    0.8,
    0.75,
    2.0 / 3.0,
    0.5,
])


def test_indexing():
    L = 136
    J = len(SCALES)

    for n in [0, 1, 35, L - 1]:
        for j in range(J):
            p = state_index(n, j, J)
            nn, jj = state_coordinates(p, J)

            assert nn == n
            assert jj == j

    print("indexing: PASS")


def test_graph():
    L = 136

    graph = build_local_transport_graph(
        L=L,
        scales=SCALES,
        shift_radius=3,
        scale_radius=1,
        shift_velocity=1.0,
    )

    print()
    print("Local transport graph")
    print("---------------------")
    print(f"L:              {graph.L}")
    print(f"J:              {graph.J}")
    print(f"P:              {graph.P}")
    print(f"E:              {graph.E}")
    print(f"shift radius:   {graph.shift_radius}")
    print(f"scale radius:   {graph.scale_radius}")
    print(f"shift velocity: {graph.shift_velocity:.6f}")
    print(f"scale velocity: {graph.scale_velocity:.6f}")

    assert graph.P == L * len(SCALES)

    assert np.all(graph.src >= 0)
    assert np.all(graph.src < graph.P)
    assert np.all(graph.dst >= 0)
    assert np.all(graph.dst < graph.P)
    assert np.all(np.isfinite(graph.cost))
    assert np.all(graph.cost >= 0)

    print("bounds/costs: PASS")


def test_self_edges():
    graph = build_local_transport_graph(
        136,
        SCALES,
        shift_radius=3,
        scale_radius=1,
    )

    pairs = set(
        zip(graph.src.tolist(), graph.dst.tolist())
    )

    for p in range(graph.P):
        assert (p, p) in pairs

    print("self edges: PASS")


def test_costs():
    L = 136
    J = len(SCALES)

    vs = median_log_scale_spacing(SCALES)

    graph = build_local_transport_graph(
        L=L,
        scales=SCALES,
        shift_radius=3,
        scale_radius=1,
        shift_velocity=1.0,
        scale_velocity=vs,
    )

    edge_cost = {
        (s, d): c
        for s, d, c in zip(
            graph.src,
            graph.dst,
            graph.cost,
        )
    }

    n = 50
    j = 2

    p = state_index(n, j, J)

    shift = state_index(n + 1, j, J)
    shift_cost = edge_cost[(p, shift)]

    expected_shift = 1.0

    assert np.isclose(
        shift_cost,
        expected_shift,
    )

    scale = state_index(n, j + 1, J)

    expected_scale = (
        (
            np.log(SCALES[j + 1])
            - np.log(SCALES[j])
        ) / vs
    ) ** 2

    assert np.isclose(
        edge_cost[(p, scale)],
        expected_scale,
    )

    diagonal = state_index(
        n + 1,
        j + 1,
        J,
    )

    expected_diagonal = (
        expected_shift
        + expected_scale
    )

    assert np.isclose(
        edge_cost[(p, diagonal)],
        expected_diagonal,
    )

    print()
    print("Cost test")
    print("---------")
    print(f"one shift:      {shift_cost:.6f}")
    print(f"one scale:      {expected_scale:.6f}")
    print(f"shift + scale:  {expected_diagonal:.6f}")
    print("cost geometry: PASS")


def test_degrees():
    graph = build_local_transport_graph(
        136,
        SCALES,
        shift_radius=3,
        scale_radius=1,
    )

    out_count = outgoing_edge_counts(graph)
    in_count = incoming_edge_counts(graph)

    assert np.all(out_count > 0)
    assert np.all(in_count > 0)

    interior = state_index(
        50,
        2,
        graph.J,
    )

    # Full interior neighborhood:
    # 7 shifts x 3 scale levels = 21 edges.
    assert out_count[interior] == 21
    assert in_count[interior] == 21

    print()
    print("Degree test")
    print("-----------")
    print(f"out degree min/max: {out_count.min()}/{out_count.max()}")
    print(f"in degree min/max:  {in_count.min()}/{in_count.max()}")
    print("degrees: PASS")


def test_transpose():
    graph = build_local_transport_graph(
        136,
        SCALES,
        shift_radius=3,
        scale_radius=1,
    )

    src_t, dst_t, cost_t = transpose_edges(
        graph
    )

    assert np.array_equal(
        src_t,
        graph.dst,
    )

    assert np.array_equal(
        dst_t,
        graph.src,
    )

    assert np.allclose(
        cost_t,
        graph.cost,
    )

    print("transpose: PASS")


if __name__ == "__main__":
    test_indexing()
    test_graph()
    test_self_edges()
    test_costs()
    test_degrees()
    test_transpose()

    print()
    print("ALL LOCAL GRAPH TESTS PASSED")