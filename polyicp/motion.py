import numpy as np
import logging

_logger = logging.getLogger("polyicp")


def compute_motion(
    x: np.ndarray, y: np.ndarray, with_scale: bool, w: np.ndarray = None
) -> tuple[float, np.ndarray, np.ndarray]:
    """Computes the similarity/rigid transform that aligns source with reference
    points in a least square sense.

    Computed components are $sRx + t$, where R is a rotation matrix, t is a
    translational shift and s is a uniform scaling constant. The transformation
    minimizes

        s*,R*,t*= argmin_{s,R,t} \sum_i w_i |y_i - sRx_i +t|^2

    Alternatively, one can fix s=1 to give the rigid motion in a
    least squares sense.

    This paper combines [1] and [2] and adds weighted scale support.

    Params:
        x: (N,D) array of source points
        y: (N,D) array of target points
        w: (N,) array of weights

    Returns:
        scale: uniform scale
        R: (D,D) rotation matrix
        t: (1,D) translation vector

    References:

    [1] Umeyama, Shinji, Least-squares estimation of transformation parameters
    between two point patterns, IEEE PAMI, 1991
    https://web.stanford.edu/class/cs273/refs/umeyama.pdf

    [2] Sorkine-Hornung, Olga, and Michael Rabinovich. "Least-squares rigid motion using svd." Computing 1.1 (2017): 1-5.

    """

    if w is None:
        w = np.ones(x.shape[0], dtype=x.dtype)

    assert x.shape == y.shape and x.shape[0] == w.shape[0]
    dims = x.shape[-1]

    w_sum = w.sum()
    if w_sum < 1e-7:
        _logger.warning("Sum of weights less than threshold")
        # raise RuntimeError("Sum of weights less than threshold")

    # Weighted centroid
    x_mu = (x * w[:, None]).sum(0, keepdims=True) / w_sum
    y_mu = (y * w[:, None]).sum(0, keepdims=True) / w_sum

    # Remove mean
    xn = x - x_mu
    yn = y - y_mu

    # Covariance of x and y.
    cov = (xn.T @ np.diag(w) @ yn) / w_sum

    # Variance of n-times the variance of x all weighted.
    # Todo cheind: this needs to be derived from first principles.
    # Here it is just an extension of [1] and [2]
    var_x = (np.square(xn) * w[:, None]).sum() / w_sum

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of COV.
    U, D, Vh = np.linalg.svd(cov)
    V = Vh.T
    if (D > np.finfo(D.dtype).eps).sum() < dims:
        _logger.warning("Degenerate rank of covariance matrix.")
        # raise RuntimeError("Degenerate rank of covariance matrix")

    # Construct S that fixes the orientation of R to get det(R)=1.
    S = np.eye(U.shape[0])
    S[-1, -1] *= np.sign(np.linalg.det(U @ V.T))

    # Construct R
    R = V @ S @ U.T

    # Recover scale
    # Note, equiv. to np.trace(R @ cov) / var _x
    # svd(cov)=UDV
    # R=V'SU'
    # R(UDV)=V'SU'UDV=V'SDV
    # then by cyclic property: trace(V'SDV)=trace(VV'SD)=trace(SD)
    scale = 1.0

    if with_scale:
        scale = (1 / var_x) * np.trace(np.diag(D) @ S)

    # Recover translation
    t = y_mu - scale * (x_mu @ R.T)

    # Apply as s*Rot(x) + t, i.e T @ S @ R @ x
    return scale, R, t


if __name__ == "__main__":

    x = np.random.randn(10, 3)
    y = x.copy() * 0.5
    w = np.full((10,), 0.5)
    w[5:] = 0

    scale, R, t = compute_motion(y, x, with_scale=True)
    print(scale, R, t)

    np.random.seed(123)
    x = np.random.randn(10, 2)
    t = np.random.randn(1, 2) * 1e-2
    R = np.array([[np.cos(0.1), -np.sin(0.1)], [np.sin(0.1), np.cos(0.1)]])
    s = 0.5
    print(s, R, t)
    y = s * (x @ R.T) + t
    scale, R, t = compute_motion(y, x, with_scale=True)
    print(scale, R, t)
    xh = s * (x @ R.T) + t
    errs = np.linalg.norm(xh - y, axis=-1)
    print(errs)
    print("-" * 20)

    # noisy x
    x[-1] += 0.1

    scale, R, t = compute_motion(y, x, with_scale=True)
    print(scale, R, t)
    xh = s * (x @ R.T) + t
    errs = np.linalg.norm(xh - y, axis=-1)
    print(errs)

    print("-" * 20)

    # weight x[0] more
    w = np.ones((10,))
    w[-1] = 0
    scale, R, t = compute_motion(y, x, with_scale=True, w=w)
    print(scale, R, t)
    xh = s * (x @ R.T) + t
    errs = np.linalg.norm(xh - y, axis=-1)
    print(errs)

    print("-" * 20)

    scale, R, t = compute_motion(y[:-1], x[:-1], with_scale=True)
    print(scale, R, t)
    xh = s * (x @ R.T) + t
    errs = np.linalg.norm(xh - y, axis=-1)
    print(errs)
