import numpy as np


class polyline:
    def __init__(self, line: np.ndarray, ptype: str = "open"):
        """Initialize one or more n-dimensional polylines.

        Note:
            When the polyline contains duplicate points are is
            self-crossing, the results are undefined.

        Params:
            line: (L,N,D) or (N,D) line points. L poly lines
                having N points in D dimensions each.
            ptype: Interpret as 'open' or 'closed' polyline.
                When 'closed' the internal representation of
                line changes to (L,N+1,D).
        """
        self.ptype = ptype
        self.squeeze_line_dim = line.ndim == 2
        if self.squeeze_line_dim:
            line = line[None,]

        if ptype == "closed":
            line = np.concatenate((line, line[:, :1]), 1)

        self.line = line
        # Segment related

        self.b = line[:, 1:] - line[:, :-1]  # start->end (L,N-1,3)
        self.bb = np.einsum(
            "lnk,lnk->ln", self.b, self.b
        )  # squared seg lengths (L,N-1)

        # (L,N-1), len_cumsum[i] = sum of squared lengths before segment i
        self.seg_lens = np.sqrt(self.bb)
        self.seg_lens_cumsum = np.cumulative_sum(
            self.seg_lens, axis=1, include_initial=True
        )[:, :-1]

    def project(self, x: np.ndarray, with_extra: bool = False):
        """Computes the closest point on each line $l_i$ for each query point $x_j$.

        In the case of ambiguous closest points, this method returns the
        first occurrence along the line.

        Params:
            x: (M,D) M query points in D dimensions
            with_extra: when true, computes additional statistics

        Returns:
            q: (L,M,D) closest polyline point fo each input point/line
            extra: (optional) dictionary containing extra information
                dist2_qx: (L,M) squared distances of q to x for each point/line
                seg_idx: (L,M) line segment index containing q
                t: (L,M) parametric distance t of q in segment [0..1]
                len2_pq: (L,M) squared path length  to q

        """
        squeeze_point_dim = x.ndim == 1
        if squeeze_point_dim:
            x = x[None,]

        L = self.line.shape[0]
        M = x.shape[0]
        N = self.line.shape[1]

        # vectors start->query (L,N-1,M,D)
        a = x[None, None, :] - self.line[:, :-1, None]

        # part of projection of a onto b
        ab = np.einsum("lnmk,lnmk->lnm", a, self.b[:, :, None])  # (L,N-1,M)

        # note, bb can be zero, when segment length is zero.
        # this happens when (start,end) same point in polyline.
        t = np.zeros((L, N - 1, M), dtype=self.bb.dtype)  # (L,N-1,M)
        mask = self.bb > 0.0  # (L,N-1)
        bb = self.bb.reshape(L, N - 1, 1)  # (L,N-1,1)
        t[mask] = np.clip(ab[mask] / bb[mask], 0, 1.0)

        # projection of a onto b (line segments) (L,N-1,M,D)
        proj_ba = self.line[:, :-1][:, :, None] + t[..., None] * self.b[:, :, None, :]
        dist2seg = np.square(x[None, None] - proj_ba).sum(-1)  # (L,N-1,M)
        seg_idx = np.argmin(dist2seg, 1)  # (L,M)

        # closest point on p to x (L,M,D)
        q = np.take_along_axis(proj_ba, seg_idx[:, None, :, None], axis=1).squeeze(1)

        extra = {}
        if with_extra:
            # t relative to the line segment that contains q (L,M)
            t_seg = np.take_along_axis(t, seg_idx[:, None, :], axis=1).squeeze(1)
            # distance squared from q to x (L,M)
            dist2_qx = np.take_along_axis(
                dist2seg, seg_idx[:, None, :], axis=1
            ).squeeze(1)
            # path length from start of polyline to begin of
            # segment that contains q (L,M)
            pathlen_upto_seg = np.take_along_axis(self.seg_lens_cumsum, seg_idx, axis=1)
            # add length of path of q in segment
            seg_len = np.take_along_axis(self.seg_lens, seg_idx, axis=1)
            pathlen_q = pathlen_upto_seg + t_seg * seg_len

            extra = {
                "dist2_qx": dist2_qx,
                "seg_idx": seg_idx,
                "t_seg": t_seg,
                "pathlen_q": pathlen_q,
            }

        if self.squeeze_line_dim:
            q = q.squeeze(0)
            extra = {k: v.squeeze(0) for k, v in extra.items()}

        if squeeze_point_dim:
            q = q.squeeze(-2)
            extra = {k: v.squeeze(-1) for k, v in extra.items()}

        if with_extra:
            return q, extra
        else:
            return q


def test_very_simple():
    import matplotlib.pyplot as plt

    line = np.array([[0.0, 0.0], [2.0, 0.0]])

    # pts = np.random.rand(10, 2)
    # pts[:, 1] = -0.1

    pts = np.array([[0.5, -0.1]])
    print(pts.shape)

    # line = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    # pts = np.random.rand(10, 2)
    # pts[:, 1] = -0.1

    fig, ax = plt.subplots(1, 1)

    pl = polyline(line)
    c, dist, seg, t, pathlen = pl.project(pts)

    print(c)
    print(dist)
    print(t)
    print(seg)
    print(pathlen)

    ax.plot(line[:, 0], line[:, 1], c="k")
    ax.scatter(line[:, 0], line[:, 1], c="k")
    ax.scatter(pts[:, 0], pts[:, 1], c="orange")
    ax.scatter(c[:, 0], c[:, 1], c="green")
    for a, b in zip(pts, c):
        l = np.stack((a, b), 0)
        ax.plot(l[:, 0], l[:, 1], c="g")
    ax.set_aspect("equal")
    plt.show()


def test_simple():
    import matplotlib.pyplot as plt

    line = np.load(r"etc/data/traj.npy")[::10]

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})

    idx = np.random.choice(len(line), 1)
    pts = line[idx] + np.random.randn(1, 3) * 1e-1

    pl = polyline(line)
    c, dist, seg, t = pl.project(pts)

    print(c)
    print(t)
    print(seg)

    ax.plot(line[:, 0], line[:, 1], line[:, 2], c="k")
    ax.scatter(line[:, 0], line[:, 1], line[:, 2], c="k")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="orange")
    ax.scatter(c[:, 0], c[:, 1], c[:, 2], c="green")
    for a, b in zip(pts, c):
        l = np.stack((a, b), 0)
        ax.plot(l[:, 0], l[:, 1], l[:, 2], c="g")

    ax.set_box_aspect(
        (np.ptp(line[..., 0]), np.ptp(line[..., 1]), np.ptp(line[..., 2]))
    )
    plt.show()


def test_pairwise_closest():
    import matplotlib.pyplot as plt

    line = np.load(r"etc/data/traj.npy")

    line0 = line[::10]

    line1 = line0 + (0.1, 0.0, 0.1)
    line1 += np.random.randn(*line1.shape) * 1e-2

    lines = np.stack((line0, line1), 0)
    x = lines
    M = x.shape[1]
    L = lines.shape[0]
    print("here", x.shape)

    pl = polyline(lines)

    p, d, s, t, pathlen = pl.project(x.reshape(-1, 3))

    x = x.reshape(2, line0.shape[0], 3)
    p = p.reshape(2, 2, line0.shape[0], 3)

    print(p.shape)

    i = 0
    j = (i + 1) % 2

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})

    ax.plot(lines[0, :, 0], lines[0, :, 1], lines[0, :, 2], c="k")
    ax.plot(lines[1, :, 0], lines[1, :, 1], lines[1, :, 2], c="k")
    ax.scatter(lines[0, :, 0], lines[0, :, 1], lines[0, :, 2], c="k")
    ax.scatter(lines[1, :, 0], lines[1, :, 1], lines[1, :, 2], c="k")
    ax.scatter(p[i, j, :, 0], p[i, j, :, 1], p[i, j, :, 2])
    ax.scatter(x[j, :, 0], x[j, :, 1], x[j, :, 2])
    for pp, xx in zip(p[i, j], x[j]):
        l = np.stack((pp, xx), 0)
        ax.plot(l[:, 0], l[:, 1], l[:, 2], c="g")
    ax.set_box_aspect(
        (np.ptp(lines[..., 0]), np.ptp(lines[..., 1]), np.ptp(lines[..., 2]))
    )
    plt.show()


if __name__ == "__main__":
    test_pairwise_closest()
