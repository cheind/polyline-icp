import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import motion, polyline

_logger = logging.getLogger("polyicp")

PairingFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
RejectFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class ICPResult:
    converged: bool  # did we converge?
    rmse: float  # last RMSE of all points
    x_hat: np.ndarray  # final point coordinates
    steps: int


def _outlier_threshold(x, iqr_factor):
    """Compute the threshold for moderate outliers."""
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    th = q75 + 1.5 * iqr
    th += 1e-8  # account for situation that all are equal
    return th


def _reject_outlier(dist2, iqr_factor: float = 1.5):
    th = _outlier_threshold(dist2, iqr_factor)
    return dist2 < th


def _closest_point_exhaustive(x, y):
    """For each $x_i$ compute closest point in $y$."""
    dists = ((y[:, None] - x[None, :]) ** 2).sum(-1)
    argmin_c = np.argmin(dists, axis=0)
    return y[argmin_c]


def _polyline_distance(x, y):
    """For each point $x_i$ compute closest point on polyline $y$."""
    y_hat, *_ = polyline.linear_reference_polyline(y, x)
    return y_hat


_pair_name_map = {
    "point": _closest_point_exhaustive,
    "polyline": _polyline_distance,
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
    use_tqdm: bool = None,
    pairing_fn: str | PairingFn = "point",
    reject_fn: str | RejectFn = "none",
    err_th: float = 1e-5,
    err_diff_th: float = 1e-4,
) -> ICPResult:

    # Setup pairing function
    if pairing_fn is None:
        pairing_fn = _closest_point_exhaustive
        _logger.debug("Assuming two pointclouds. Specify pairing_fn=...")
    elif isinstance(str, pairing_fn):
        pairing_fn = _pair_name_map[pairing_fn]

    # Setup rejection function
    if pairing_fn is None:
        reject_fn = _reject_name_map["none"]
        _logger.debug("Assuming no rejection. Specify reject_fn=...")
    elif isinstance(str, pairing_fn):
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

    for step in gen:

        mask = reject_fn(r)
        if ~np.any(mask):
            # No single inliner
            _logger.debug("Failed to converge: all-rejected")
            failed = True
            break

        scale, R, t = motion.compute_motion(
            closest_y[mask], x[mask], with_scale=with_scale
        )
        x = scale * (x @ R.T) + t.reshape(1, 3)

        closest_y = pairing_fn(x, y)
        r = np.square(closest_y - x).sum(-1)
        rmse = np.sqrt(r.mean())

        if use_tqdm and (step % 5 == 0):
            gen.set_postfix({"rmse": rmse})

        if (rmse < err_th) or ((step > 0) and (prev_rmse - rmse < err_diff_th)):
            _logger.debug(f"Converged RMSE: {rmse}")
            break

        prev_rmse = rmse

    result = ICPResult(
        converged=step < max_iter and not failed,
        rmse=rmse,
        x_hat=x,
        steps=step + 1,
    )
    return result
