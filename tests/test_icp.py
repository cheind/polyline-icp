import pytest
import numpy as np
from polyicp.icp import icp

from .helpers import assert_transform_equal
from .helpers import random_problem2d


# import matplotlib.pyplot as plt
# plt.scatter(x[pidx, 0], x[pidx, 1])
# plt.scatter(y[:, 0], y[:, 1], c="k", s=4, zorder=10)
# plt.scatter(r.x_hat[:, 0], r.x_hat[:, 1], c="g")
# plt.show()
# print(r.rmse)


@pytest.mark.parametrize("seed", range(10))
def test_icp_2d_noisefree_point2point(seed):
    np.random.seed(seed)
    x, y, theta = random_problem2d(
        n=10, scale_std=0.0, noise_std=0.0, t_std=1e-1, angle_max=np.radians(20)
    )
    pidx = np.random.permutation(np.arange(10))
    r = icp(x[pidx], y, with_scale=False, pairing_fn="point")
    assert r.converged
    assert_transform_equal(r.theta_hat, theta)


@pytest.mark.parametrize("seed", range(10))
def test_icp_2d_noisefree_point2polyline(seed):
    np.random.seed(seed)
    x, y, theta = random_problem2d(
        n=10, scale_std=0.0, noise_std=0.0, t_std=1e-1, angle_max=np.radians(20)
    )
    pidx = np.random.permutation(np.arange(10))
    r = icp(x[pidx], y, with_scale=False, pairing_fn="polyline")
    assert r.converged
    assert_transform_equal(r.theta_hat, theta)


@pytest.mark.parametrize("seed", range(10))
def test_icp_2d_outlier_point2point(seed):
    np.random.seed(seed)
    x, y, theta = random_problem2d(
        n=10, scale_std=0.0, noise_std=0.0, t_std=1e-1, angle_max=np.radians(20)
    )
    x = np.concatenate((x, [[100, 100]]), 0)  # add outlier
    pidx = np.random.permutation(np.arange(11))
    r = icp(
        x[pidx],
        y,
        max_iter=20,
        with_scale=False,
        pairing_fn="point",
        weighting_fn="outlier",
    )

    assert r.converged
    assert_transform_equal(r.theta_hat, theta)


@pytest.mark.parametrize("seed", range(10))
def test_icp_2d_lines(seed):
    np.random.seed(seed)

    x = np.array([[0, 0], [1, 0], [2, 0.0]])
    y = x.copy()
    x -= (0.1, 0)

    r = icp(x, y, max_iter=20, with_scale=False, pairing_fn="point")

    assert r.converged
    assert_transform_equal(r.theta_hat, (1.0, np.eye(2), np.array([[0.1, 0]])))

    # resampled
    x = np.array([[0, 0], [1, 0], [2, 0.0]])
    x -= (0.1, 0)
    y = np.array([[0, 0], [0.3, 0], [1.2, 0], [2, 0.0]])

    # test with point2point distance fails due to resampling
    r = icp(x, y, max_iter=100, with_scale=False, pairing_fn="point")
    with pytest.raises(Exception):
        assert_transform_equal(r.theta_hat, (1.0, np.eye(2), np.array([[0.1, 0]])))

    # test with point2polyline succeeds
    r = icp(x, y, max_iter=100, with_scale=False, pairing_fn="polyline")
    assert_transform_equal(r.theta_hat, (1.0, np.eye(2), np.array([[0.1, 0]])))
