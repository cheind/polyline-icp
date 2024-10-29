import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate

import matplotlib.animation as animation
from matplotlib import collections as mc

from polyicp.polylines import polyline


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
        fig, animate, repeat=False, frames=nframes, interval=30
    )

    plt.show()


if __name__ == "__main__":
    plot_closest_on_polyline()
