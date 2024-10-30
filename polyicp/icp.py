import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from . import motion, polylines, utils

_logger = logging.getLogger("polyicp")


class PairingFn(Protocol):
    def pair(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Determine the corresponding points in y for x.

        Params:
            x: (N,D) array of x at iteration t.
            y: (N,D) array of reference points.

        Returns:
            x_pair: (M,D) partner points in x, such that x_i and y_i
                match up.
            y_pair: (M,D) partner points in y such that x_i and y_i
                match up.
            dist2: (M,) Squared distance
        """
        ...


class WeightingFn(Protocol):
    def weight(
        self, x_pair: np.ndarray, y_pair: np.ndarray, dist2: np.ndarray
    ) -> np.ndarray:
        """Determine the weights for each pair of points

        Params:
            x_pair: (M,D) array of partner points in x
            y_pair: (M,D) array of partner points in y
            dist2: (M,) Squared distance

        Returns:
            w: (M,) array of weights for each pair.
        """


@dataclass
class icp_result:
    converged: bool  # did we converge?
    rmse: float  # last RMSE of all points
    x_hat: np.ndarray  # final point coordinates
    theta_hat: tuple[float, np.ndarray, np.ndarray]  # final transform
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


class point_to_point(PairingFn):
    """For each $x_i$ compute closest point in $y$."""

    def __init__(self, y: np.ndarray):
        pass

    def pair(self, x, y):
        dist2 = ((y[:, None] - x[None, :]) ** 2).sum(-1)
        amin = np.argmin(dist2, axis=0)
        return x, y[amin], dist2[amin, np.arange(x.shape[0])]


class point_to_polyline(PairingFn):
    """For each point $x_i$ compute closest point on polyline $y$."""

    def __init__(self, y: np.ndarray, ptype: str = "open"):
        self.pl = polylines.polyline(y, ptype=ptype)

    def pair(self, x, y):
        y_hat, extra = self.pl.project(x, with_extra=True)
        return x, y_hat, extra["dist2_qx"]


class outlier_rejection(WeightingFn):
    """Hard rejects pairs with distances greater than a IQR threshold."""

    def __init__(self, y: np.ndarray, iqr_factor: float = 3.0, mode="soft"):
        self.iqr_factor = iqr_factor
        self.mode = mode

    def weight(self, x_pair, y_pair, dist2):
        th = _outlier_threshold(dist2, self.iqr_factor)
        if self.mode == "soft":
            w = np.minimum(np.exp(-(dist2 - th)), 1.0)
        else:
            w = np.ones_like(dist2)
            w[dist2 > th] = 0.0
        return w


class constant_weighting(WeightingFn):
    def __init__(self, y: np.ndarray):
        pass

    def weight(self, x_pair, y_pair, dist2):
        return np.ones_like(dist2)


_pair_map = {
    "point": point_to_point,
    "polyline": point_to_polyline,
    # "index": _index_pairing,
}

_weight_map = {
    "outlier": outlier_rejection,
    "constant": constant_weighting,
}


def icp(
    x: np.ndarray,
    y: np.ndarray,
    max_iter: int = 100,
    pairing_fn: str | PairingFn = "point",
    weighting_fn: str | WeightingFn = "constant",
    with_scale: bool = False,
    err_th: float = 1e-16,
    err_diff_th: float = 1e-12,
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
        x: point-set to be registered to y
        y: target point-set/polyline
        max_iter: maximum number of iterations
        pairing_fn: Pairing function to determine matches
        weighting_fn: Weighting function to compute individual match weights
        with_scale: when true computes scale, rotation and translation
            when false, computes rotation and translation components
        err_th: Terminates when error is less than this threshold
        err_diff_th: Terminates when the error progress falls below this
            threshold
        use_tqdm: When true and tqdm is found, progress is shown
    """

    # Setup pairing function
    if pairing_fn is None:
        pairing_fn = _pair_map["point"](y)
        _logger.debug("Assuming two pointclouds. Specify pairing_fn=...")
    elif isinstance(pairing_fn, str):
        pairing_fn = _pair_map[pairing_fn](y)

    # Setup rejection function
    if weighting_fn is None:
        weighting_fn = _weight_map["constant"](y)
        _logger.debug("Assuming constant weighting. Specify weighting_fn=...")
    elif isinstance(weighting_fn, str):
        weighting_fn = _weight_map[weighting_fn](y)

    # Use tqdm if None and available
    if use_tqdm is None:
        try:
            from tqdm import tqdm

            use_tqdm = True
        except ImportError:
            use_tqdm = False

    gen = range(max_iter)
    if use_tqdm:
        from tqdm import tqdm

        gen: tqdm = tqdm(gen)

    prev_rmse = None
    failed = False
    history = []

    # kick-off
    x_pair, y_pair, xy_dist2 = pairing_fn.pair(x, y)

    for step in gen:

        weights = weighting_fn.weight(x_pair, y_pair, xy_dist2)
        if weights.sum() == 0:
            _logger.debug("Failed to converge: all-rejected")
            failed = True
            break

        incr_motion = motion.compute_motion(
            x_pair, y_pair, with_scale=with_scale, w=weights
        )

        history.append(incr_motion)
        scale, R, t = incr_motion
        x = scale * (x @ R.T) + t

        x_pair, y_pair, xy_dist2 = pairing_fn.pair(x, y)

        # actual the weighted rmse, required for convergence properties
        rmse = np.sqrt((xy_dist2 * weights).sum() / weights.sum())

        if use_tqdm:
            gen.set_postfix({"rmse": rmse})

        if (rmse < err_th) or ((step > 0) and (prev_rmse - rmse < err_diff_th)):
            _logger.debug(f"Converged RMSE: {rmse}/{prev_rmse}")
            break

        prev_rmse = rmse

    history = utils.incremental_to_cummulative_motions(history)

    result = icp_result(
        converged=step < max_iter and not failed,
        rmse=rmse,
        x_hat=x,
        theta_hat=history[-1],
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

    test = ref + np.random.randn(3) * 1e-1
    # ref = ref[::20]  # this flips!
    # ref = ref[::20]

    import matplotlib.pyplot as plt

    # coarse align
    # r = icp(test, ref, with_scale=False, max_iter=1, pairing_fn="index")

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], c="k")
    ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], s=4, c="k")
    ax.plot(test[:, 0], test[:, 1], test[:, 2], c="g")
    ax.scatter(test[:1, 0], test[:1, 1], test[:1, 2], c="g")
    ax.set_box_aspect((np.ptp(ref[..., 0]), np.ptp(ref[..., 1]), np.ptp(ref[..., 2])))

    r1 = icp(
        test,
        ref,
        with_scale=False,
        max_iter=100,
        pairing_fn="point",
        weighting_fn="outlier",
    )
    r2 = icp(
        test,
        ref,
        with_scale=False,
        max_iter=100,
        pairing_fn="polyline",
        weighting_fn="outlier",
    )

    print(r1.converged, r2.converged)

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

    x_hat, y_hat, _ = point_to_point(ref).pair(r1.x_hat, ref)
    for a, b in zip(x_hat, y_hat):
        l = np.stack((a, b), 0)
        ax.plot(l[:, 0], l[:, 1], l[:, 2], c="g")

    ax.set_box_aspect((np.ptp(ref[..., 0]), np.ptp(ref[..., 1]), np.ptp(ref[..., 2])))

    plt.show()


if __name__ == "__main__":

    test_b()
