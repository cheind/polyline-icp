import numpy as np


def compute_motion(
    y: np.ndarray,
    x: np.ndarray,
    with_scale: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Computes the similarity/rigid transform that aligns source with reference
    points in a least square sense.

    Computed components are $sRx + t$, where R is a rotation matrix, $t$ is a
    translational shift and $s$ is a uniform scaling constant. The transformation
    minimizes $\sum_i |y_i - sRx_i +t|^2.

    You might fix $s=1$ to retrieve a rigid motion.

    Based on
    Umeyama, Shinji, Least-squares estimation of transformation parameters
    between two point patterns, IEEE PAMI, 1991
    https://web.stanford.edu/class/cs273/refs/umeyama.pdf
    https://igl.ethz.ch/projects/ARAP/svd_rot.pdf

    Params:
        y: (N,D) array of target points
        x: (N,D) array of source points

    Returns:
        scale: uniform scale
        R: (D,D) rotation matrix
        t: (D,1) translation vector
    """

    assert x.shape == y.shape and x.shape[-1] == 3

    x = x.T
    y = y.T

    # 1. Remove mean.
    mu1 = x.mean(axis=1, keepdims=True)
    mu2 = y.mean(axis=1, keepdims=True)
    X1 = x - mu1
    X2 = y - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, d, Vh = np.linalg.svd(K)
    # if np.count_nonzero(d > np.finfo(d.dtype).eps) < 3:
    #    raise ValueError("Degenerate covariance rank")
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    # Ensure right-handiness
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))

    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1
    scale = scale if with_scale else 1.0

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # Apply as s*Rot(x) + t, i.e T@S@R@x
    return scale, R, t
