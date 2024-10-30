import matplotlib.pyplot as plt
import numpy as np
import io
import imageio.v3 as iio
import scipy.interpolate as interpolate

import matplotlib.animation as animation
from matplotlib import collections as mc

from polyicp.polylines import polyline
from polyicp.icp import icp


class ImageIOWriter(animation.AbstractMovieWriter):
    @classmethod
    def isAvailable(cls):
        return True

    def setup(self, fig, outfile, dpi=None):
        super().setup(fig, outfile, dpi=dpi)
        self._frames = []

    def grab_frame(self, **savefig_kwargs):
        buf = io.BytesIO()
        self.fig.savefig(buf, **{**savefig_kwargs, "format": "rgba", "dpi": self.dpi})
        self._frames.append(
            np.frombuffer(buf.getbuffer(), dtype=np.uint8).reshape(
                (self.frame_size[1], self.frame_size[0], 4)
            )
        )

    def finish(self):
        iio.imwrite(self.outfile, self._frames, fps=self.fps, loop=1)


def plot_closest_on_polyline():

    x = np.array([0.0, 1.2, 1.9, 3.2, 4.0, 6.5])
    y = np.array([0.0, 2.3, 3.0, 4.3, 2.9, 3.1])
    t, c, k = interpolate.splrep(x, y, s=0, k=4)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)
    xx = np.linspace(x.min(), x.max(), 100)
    xy = np.stack((xx, spline(xx)), -1)
    pc = polyline(xy, ptype="open")

    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.grid()
    ax.set_xlim(-0.3, 6.8)
    ax.set_ylim(-2.4, 4.6)
    ax.autoscale(False)
    ax.plot(xy[:, 0], xy[:, 1], c="k", label="polyline")
    # ax.scatter(xy[:, 0], xy[:, 1], c="k", s=6, label="polyline-points")

    # second spline
    xmin2, xmax2 = x.min() + 0.5, x.max() - 1.8
    shiftx, shifty = 1.0, -0.7
    xx = np.linspace(xmin2, xmax2, 60)
    xy = np.stack((xx + shiftx, spline(xx) + shifty), -1)
    ax.plot(xy[:, 0], xy[:, 1], c="magenta", linewidth=1, linestyle="dashed")

    anim_p = ax.scatter([], [], c="magenta", s=20, label="query", zorder=10)
    anim_q = ax.scatter([], [], c="k", s=20, label="closest", zorder=10)
    anim_pq = ax.plot([], [], c="k", zorder=5, linewidth=1)[0]
    anim_l = mc.LineCollection([], cmap="jet", zorder=8)
    ax.add_collection(anim_l)

    nframes = 150

    ax.legend(loc="lower left", ncols=2)

    ps = []  # (N,2)
    qs = []  # (N,2)

    def init():
        ps.clear()
        qs.clear()
        anim_pq.set_data([], [])
        anim_p.set_offsets(np.empty((0, 2)))
        anim_p.set_offsets(np.empty((0, 2)))
        anim_l.set_segments([])
        return (
            anim_p,
            anim_pq,
            anim_pq,
            anim_l,
        )

    def animate(i):
        t = i / nframes
        xp = xmin2 + (xmax2 - xmin2) * t
        p = np.array([xp + shiftx, spline(xp) + shifty])
        q = pc.project(p)
        anim_p.set_offsets(p)
        anim_q.set_offsets(q)

        ret = [anim_p, anim_q]
        if (i + 1) % 5 == 0:
            ps.append(p)
            qs.append(q)
            qp = np.stack((qs, ps), 1)
            anim_l.set_segments(qp)
            anim_l.set_array(np.arange(len(qp)))
            anim_l.set_clim(0, nframes / 5 - 1)
            ret.append(anim_l)
            ret.append(anim_pq)
            anim_pq.set_data([[], []])
        else:
            anim_pq.set_data([p[0], q[0]], [p[1], q[1]])
            ret.append(anim_pq)

        return ret

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, repeat=False, frames=nframes, interval=30
    )
    ani.save(
        "closest_on_polyline.gif",
        dpi=150,
        writer=ImageIOWriter(fps=30),
    )

    plt.show()


def plot_icp():
    np.random.seed(456)
    x = np.array([0.0, 1.2, 1.9, 3.2, 4.0, 6.5])
    y = np.array([0.0, 2.3, 3.0, 4.3, 2.9, 3.1])
    t, c, k = interpolate.splrep(x, y, s=0, k=4)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)
    xx = np.linspace(x.min(), x.max(), 60)
    y = np.stack((xx, spline(xx)), -1)  # Target
    idx = np.random.choice(len(y), 30, replace=False)
    idx = np.sort(idx)
    y = y[idx]

    py = polyline(y, ptype="open")

    xmin2, xmax2 = x.min() + 0.5, x.max() - 1.8
    shiftx, shifty = 1.0, -0.5
    xx = np.linspace(xmin2, xmax2, 16)
    x = np.stack((xx + shiftx, spline(xx) + shifty), -1)  # Source

    r_point = icp(
        x,
        y,
        max_iter=100,
        pairing_fn="point",
        weighting_fn="constant",
    )
    r_line = icp(
        x,
        y,
        max_iter=100,
        pairing_fn="polyline",
        weighting_fn="constant",
    )

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    def setup_ax(ax):
        ax.set_axisbelow(True)
        ax.grid()
        ax.set_xlim(-0.3, 6.8)
        ax.set_ylim(-2.4, 4.6)
        ax.autoscale(False)
        ax.plot(y[:, 0], y[:, 1], c="k", label="reference")
        ax.scatter(y[:, 0], y[:, 1], c="k", s=6)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

    setup_ax(axs[0])
    setup_ax(axs[1])
    axs[0].set_title("point-to-point")
    axs[1].set_title("point-to-polyline")

    # ax.scatter(xy[:, 0], xy[:, 1], c="k", s=6, label="polyline-points")

    anim_x0 = axs[0].plot(
        x[:, 0],
        x[:, 1],
        c="magenta",
        linewidth=1.5,
        linestyle="dashed",
        zorder=10,
        label="source",
    )[0]
    anim_x1 = axs[1].plot(
        x[:, 0],
        x[:, 1],
        c="magenta",
        linewidth=1.5,
        linestyle="dashed",
        zorder=10,
        label="source",
    )[0]
    anim_l0 = mc.LineCollection([], color="yellow", zorder=8, label="matches")
    anim_l1 = mc.LineCollection([], color="yellow", zorder=8, label="matches")
    axs[0].add_collection(anim_l0)
    axs[1].add_collection(anim_l1)

    axs[0].legend(loc="lower left", ncols=2)
    axs[1].legend(loc="lower left", ncols=2)

    nframes = max(len(r_point.history), len(r_line.history))

    def init():
        anim_x0.set_data(x[:, 0], x[:, 1])
        anim_x1.set_data(x[:, 0], x[:, 1])
        return (anim_x0, anim_x1)

    def animate(i):
        updates = []
        if i < len(r_point.history):
            s, R, t = r_point.history[i]
            xhat = s * (x @ R.T) + t
            anim_x0.set_data(xhat[:, 0], xhat[:, 1])

            d2 = np.square(xhat[:, None] - y[None, :]).sum(-1)
            amin = np.argmin(d2, axis=1)
            qp = np.stack((xhat[::5], y[amin][::5]), 1)
            anim_l0.set_segments(qp)

            updates.append(anim_x0)
            updates.append(anim_l0)
        if i < len(r_line.history):
            s, R, t = r_line.history[i]
            xhat = s * (x @ R.T) + t
            anim_x1.set_data(xhat[:, 0], xhat[:, 1])

            qp = np.stack((xhat[::5], py.project(xhat)[::5]), 1)
            anim_l1.set_segments(qp)

            updates.append(anim_x1)
            updates.append(anim_l1)

        return updates

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, repeat=False, frames=nframes, interval=150
    )
    ani.save(
        "icp.gif",
        dpi=150,
        writer=ImageIOWriter(fps=30),
    )

    plt.show()


if __name__ == "__main__":
    plot_closest_on_polyline()
    plot_icp()
