import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import motion, polylines

_logger = logging.getLogger("polyicp")

PairingFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
RejectFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class icp_result:
    converged: bool  # did we converge?
    rmse: float  # last RMSE of all points
    x_hat: np.ndarray  # final point coordinates
    steps: int  # number of steps
    history: list[
        tuple[float, np.ndarray, np.ndarray]
    ]  # sequence of transformation parameters


def _outlier_threshold(x, iqr_factor):
    """Compute the threshold for moderate outliers."""
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    th = q75 + iqr_factor * iqr
    th += 1e-8  # account for situation that all are equal
    return th


def _reject_outlier(dist2, iqr_factor: float = 3.0):
    th = _outlier_threshold(dist2, iqr_factor)
    return dist2 < th


def _point_distance(y):
    """For each $x_i$ compute closest point in $y$."""

    def compute(x, y):
        dists = ((y[:, None] - x[None, :]) ** 2).sum(-1)
        argmin_c = np.argmin(dists, axis=0)
        return y[argmin_c]

    return compute


def _index_pairing(_):
    def compute(x, y):
        n = min(len(x), len(y))
        return y[:n]

    return compute


def _polyline_distance(y):
    """For each point $x_i$ compute closest point on polyline $y$."""
    # Memorize y to utilize precomputed values in multiple iterations
    pl = polylines.polyline(y)

    def compute(x, _):
        y_hat = pl.project(x, with_extra=False)
        return y_hat

    return compute


_pair_name_map = {
    "point": _point_distance,
    "polyline": _polyline_distance,
    "index": _index_pairing,
}

_reject_name_map = {
    "outlier": _reject_outlier,
    "none": lambda d: np.ones_like(d, dtype=bool),
}


def icp(
    x: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    with_scale: bool,
    pairing_fn: str | PairingFn = "point",
    reject_fn: str | RejectFn = "none",
    err_th: float = 1e-5,
    err_diff_th: float = 1e-4,
    use_tqdm: bool = None,
) -> icp_result:
    """Iteratively computes the transformation that aligns $x$ with $y$.

    The algorithm iteratively determines the 'closest' points of $x$ to
    $y$. From these pairings a similarity/rigid transformation is estimated
    and applied to $x$ for the next iteration.

    Depending on the pairing function, $y$ is interpreted as a pointset
    or a polyline.

    Errors are measured as RMSE and have hence the units of $x$.

    Params:
        x: point-set to be registered to $y$
        y: target point-set/polyline
        max_iter: maximum number of iterations
        with_scale: when true computes scale, rotation and translation.
            when false, computes rotation and translation components.
    """

    # Setup pairing function
    if pairing_fn is None:
        pairing_fn = _pair_name_map["point"](y)
        _logger.debug("Assuming two pointclouds. Specify pairing_fn=...")
    elif isinstance(pairing_fn, str):
        pairing_fn = _pair_name_map[pairing_fn](y)

    # Setup rejection function
    if reject_fn is None:
        reject_fn = _reject_name_map["none"]
        _logger.debug("Assuming no rejection. Specify reject_fn=...")
    elif isinstance(reject_fn, str):
        reject_fn = _reject_name_map[reject_fn]

    # Use tqdm if None and available
    if use_tqdm is None:
        try:
            from tqdm import tqdm

            use_tqdm = True
        except ImportError:
            use_tqdm = False

    gen = range(max_iter)
    if use_tqdm:
        gen: tqdm = tqdm(gen)

    prev_rmse = None
    failed = False

    # kick-off
    closest_y = pairing_fn(x, y)
    r = np.square(closest_y - x).sum(-1)

    history = []
    for step in gen:

        mask = reject_fn(r)
        if ~np.any(mask):
            # No single inliner
            _logger.debug("Failed to converge: all-rejected")
            failed = True
            break

        scale, R, t = motion.compute_motion(
            x[mask], closest_y[mask], with_scale=with_scale
        )
        history.append((scale, R, t))
        x = scale * (x @ R.T) + t.reshape(1, 3)

        closest_y = pairing_fn(x, y)
        r = np.square(closest_y - x).sum(-1)
        rmse = np.sqrt(r.mean())

        if use_tqdm and (step % 5 == 0):
            gen.set_postfix({"rmse": rmse})

        if (rmse < err_th) or ((step > 0) and (prev_rmse - rmse < err_diff_th)):
            _logger.debug(f"Converged RMSE: {rmse}/{prev_rmse}")
            break

        prev_rmse = rmse

    result = icp_result(
        converged=step < max_iter and not failed,
        rmse=rmse,
        x_hat=x,
        steps=step + 1,
        history=history,
    )
    return result


def test_a():
    logging.basicConfig(level=logging.DEBUG)

    ref = np.load(r"etc/data/ref.npy")[0]
    test = np.load(r"etc/data/test.npy")[2]
    print(test.shape)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})

    # coarse align
    r = icp(test, ref, with_scale=False, max_iter=1, pairing_fn="index")

    r1 = icp(r.x_hat, ref, with_scale=False, max_iter=20, pairing_fn="point")
    r2 = icp(r.x_hat, ref, with_scale=False, max_iter=20, pairing_fn="polyline")

    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], c="k")
    ax.plot(r1.x_hat[:, 0], r1.x_hat[:, 1], r1.x_hat[:, 2], c="g")
    ax.plot(r2.x_hat[:, 0], r2.x_hat[:, 1], r2.x_hat[:, 2], c="magenta")
    ax.scatter(ref[:1, 0], ref[:1, 1], ref[:1, 2], c="k")
    ax.scatter(r1.x_hat[:1, 0], r1.x_hat[:1, 1], r1.x_hat[:1, 2], c="g")

    ax.set_box_aspect((np.ptp(ref[..., 0]), np.ptp(ref[..., 1]), np.ptp(ref[..., 2])))
    plt.show()


def test_b():
    np.random.seed(71189)
    ref = np.load(r"etc/data/ref.npy")[0]

    test = ref + np.random.randn(3)
    ref = ref[20::40]  # this flips!
    # ref = ref[::20]

    import matplotlib.pyplot as plt

    # coarse align
    # r = icp(test, ref, with_scale=False, max_iter=1, pairing_fn="index")

    r1 = icp(
        test,
        ref,
        with_scale=False,
        max_iter=100,
        pairing_fn="point",
        reject_fn="outlier",
    )
    r2 = icp(
        test,
        ref,
        with_scale=False,
        max_iter=100,
        pairing_fn="polyline",
        reject_fn="outlier",
    )

    print(np.linalg.det(r1.history[-1][1]))
    print(np.linalg.det(r2.history[-1][1]))

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], c="k")
    ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], s=4, c="k")
    ax.plot(r1.x_hat[:, 0], r1.x_hat[:, 1], r1.x_hat[:, 2], c="g")
    ax.plot(r2.x_hat[:, 0], r2.x_hat[:, 1], r2.x_hat[:, 2], c="magenta")
    ax.scatter(ref[:1, 0], ref[:1, 1], ref[:1, 2], c="k")
    ax.scatter(r1.x_hat[:1, 0], r1.x_hat[:1, 1], r1.x_hat[:1, 2], c="g")
    ax.scatter(r2.x_hat[:1, 0], r2.x_hat[:1, 1], r2.x_hat[:1, 2], c="magenta")
    ax.set_box_aspect((np.ptp(ref[..., 0]), np.ptp(ref[..., 1]), np.ptp(ref[..., 2])))

    # matches
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], c="k")
    ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], s=4, c="k")
    ax.plot(r2.x_hat[:, 0], r2.x_hat[:, 1], r2.x_hat[:, 2], c="magenta")
    ax.scatter(ref[:1, 0], ref[:1, 1], ref[:1, 2], c="k")
    ax.scatter(r2.x_hat[:1, 0], r2.x_hat[:1, 1], r2.x_hat[:1, 2], c="magenta")

    pl = polylines.polyline(ref)
    c = pl.project(r2.x_hat)
    for a, b in zip(r2.x_hat, c):
        l = np.stack((a, b), 0)
        ax.plot(l[:, 0], l[:, 1], l[:, 2], c="g")

    ax.set_box_aspect((np.ptp(ref[..., 0]), np.ptp(ref[..., 1]), np.ptp(ref[..., 2])))

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], c="k")
    ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], s=4, c="k")
    ax.plot(r1.x_hat[:, 0], r1.x_hat[:, 1], r1.x_hat[:, 2], c="magenta")
    ax.scatter(ref[:1, 0], ref[:1, 1], ref[:1, 2], c="k")
    ax.scatter(r1.x_hat[:1, 0], r1.x_hat[:1, 1], r1.x_hat[:1, 2], c="magenta")

    c = _point_distance(ref)(r1.x_hat, ref)
    for a, b in zip(r1.x_hat, c):
        l = np.stack((a, b), 0)
        ax.plot(l[:, 0], l[:, 1], l[:, 2], c="g")

    ax.set_box_aspect((np.ptp(ref[..., 0]), np.ptp(ref[..., 1]), np.ptp(ref[..., 2])))

    plt.show()


if __name__ == "__main__":

    test_b()
