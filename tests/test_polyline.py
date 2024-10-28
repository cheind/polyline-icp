import numpy as np
from polyicp.polylines import polyline


def test_polylines_straight_line():

    lpts = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    x = np.array([[0.0, 0.0], [1.5, -0.1], [1.5, 0.1], [3.0, 0.0], [5.0, 2.0]])

    pl = polyline(lpts)
    q, extra = pl.project(x, with_extra=True)

    assert np.allclose(q, [[1.0, 0.0], [1.5, 0.0], [1.5, 0.0], [3.0, 0.0], [3.0, 0.0]])
    assert np.allclose(extra["dist2_qx"], [1.0, 0.01, 0.01, 0.0, 8.0])
    assert np.allclose(extra["seg_idx"], [0, 0, 0, 1, 1])
    assert np.allclose(extra["t_seg"], [0, 0.5, 0.5, 1, 1])
    assert np.allclose(extra["pathlen_q"], [0, 0.5, 0.5, 2, 2])


def test_polylines_multiple_straight_lines():

    lpts = np.array(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],  # 1->2->3
            [[3.0, 1.0], [2.0, 1.0], [1.0, 1.0]],  # 3->2->1
        ]
    )

    x = np.array(
        [
            [0.0, 0.0],
            [1.5, -0.1],
            [1.5, 0.1],
            [3.0, 0.0],
            [5.0, 2.0],
        ]
    )

    pl = polyline(lpts)
    q, extra = pl.project(x, with_extra=True)

    assert q.shape == (2, 5, 2)

    assert np.allclose(
        q[0], [[1.0, 0.0], [1.5, 0.0], [1.5, 0.0], [3.0, 0.0], [3.0, 0.0]]
    )
    assert np.allclose(
        q[1], [[1.0, 1.0], [1.5, 1.0], [1.5, 1.0], [3.0, 1.0], [3.0, 1.0]]
    )

    assert np.allclose(
        extra["dist2_qx"],
        [
            [1.0, 0.01, 0.01, 0.0, 8.0],
            [2.0, 1.1**2, 0.9**2, 1, 5],
        ],
    )
    assert np.allclose(
        extra["seg_idx"],
        [
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
        ],
    )
    assert np.allclose(
        extra["t_seg"],
        [
            [0, 0.5, 0.5, 1, 1],
            [1, 0.5, 0.5, 0.0, 0.0],
        ],
    )
    assert np.allclose(
        extra["pathlen_q"],
        [
            [0, 0.5, 0.5, 2, 2],
            [2, 1.5, 1.5, 0, 0],
        ],
    )


def test_polylines_closed():

    lpts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    x = np.array(
        [
            [-0.5, 0.5],
        ]
    )

    pl = polyline(lpts, ptype="closed")
    q, extra = pl.project(x, with_extra=True)

    assert np.allclose(q, [[0.0, 0.5]])
    assert np.allclose(extra["dist2_qx"], [0.25])
    assert np.allclose(extra["seg_idx"], [3])
    assert np.allclose(extra["t_seg"], [0.5])
    assert np.allclose(extra["pathlen_q"], [3.5])


def test_polylines_ambiguous():

    lpts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    x = np.array(
        [
            [0.5, 0.5],
        ]
    )
    # argmin: In case of multiple occurrences of the minimum values,
    # the indices corresponding to the first occurrence are returned.

    pl = polyline(lpts, ptype="closed")
    q, extra = pl.project(x, with_extra=True)

    assert np.allclose(q, [[0.5, 0.0]])
    assert np.allclose(extra["dist2_qx"], [0.25])
    assert np.allclose(extra["seg_idx"], [0])
    assert np.allclose(extra["t_seg"], [0.5])
    assert np.allclose(extra["pathlen_q"], [0.5])
