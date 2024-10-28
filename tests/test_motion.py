import pytest
import numpy as np
from polyicp.motion import compute_motion

from .helpers import assert_transform_equal
from .helpers import random_problem2d


def test_identity():
    x = np.random.randn(10, 3)
    y = x.copy()
    s, R, t = compute_motion(x, y, with_scale=True)
    assert_transform_equal((s, R, t), (1.0, np.eye(3), np.zeros((1, 3))))

    x = np.random.randn(10, 2)
    y = x.copy()
    s, R, t = compute_motion(x, y, with_scale=True)
    assert_transform_equal((s, R, t), (1.0, np.eye(2), np.zeros((1, 2))))

    x = np.random.randn(10, 4)
    y = x.copy()
    s, R, t = compute_motion(x, y, with_scale=True)
    assert_transform_equal((s, R, t), (1.0, np.eye(4), np.zeros((1, 4))))


@pytest.mark.parametrize("seed", range(20))
def test_rigid(seed):
    np.random.seed(seed)

    x, y, theta = random_problem2d(n=10, scale_std=0.0)
    s, R, t = compute_motion(x, y, with_scale=False)

    xh = s * (x @ R.T) + t
    errs = np.linalg.norm(xh - y, axis=-1)

    assert_transform_equal((s, R, t), theta)
    assert np.allclose(errs, 0, atol=1e-6)


@pytest.mark.parametrize("seed", range(20))
def test_similarity(seed):
    np.random.seed(seed)

    x, y, theta = random_problem2d(n=10, scale_std=0.1)
    s, R, t = compute_motion(x, y, with_scale=True)

    xh = s * (x @ R.T) + t
    errs = np.linalg.norm(xh - y, axis=-1)

    assert_transform_equal((s, R, t), theta)
    assert np.allclose(errs, 0, atol=1e-6)


@pytest.mark.parametrize("seed", range(20))
def test_noisy(seed):
    np.random.seed(seed)

    x, y, theta = random_problem2d(n=10, scale_std=0.1, noise_std=1e-2)
    s, R, t = compute_motion(x, y, with_scale=True)

    xh = s * (x @ R.T) + t
    errs = np.linalg.norm(xh - y, axis=-1)

    assert errs.mean() < 1e-1 and errs.mean() > 1e-3


@pytest.mark.parametrize("seed", range(20))
def test_weighted(seed):
    np.random.seed(seed)

    x, y, theta = random_problem2d(n=10, scale_std=0.1, noise_std=0.0)

    # simulate signifant outlier at last position
    x[-1] += 1.0

    # 1. solve without weights -> error distributes across all
    s, R, t = compute_motion(
        x,
        y,
        with_scale=True,
    )

    xh = s * (x @ R.T) + t
    errs = np.linalg.norm(xh - y, axis=-1)
    # expect least squares error
    assert errs.mean() > 0.1

    # 2. solve with last weight = 0, last one gets all the error
    w = np.ones(x.shape[0])
    w[-1] = 0.0
    s, R, t = compute_motion(x, y, with_scale=True, w=w)

    xh = s * (x @ R.T) + t
    errs = np.linalg.norm(xh - y, axis=-1)
    np.testing.assert_allclose(errs[:-1], 0, atol=1e-6)
    np.testing.assert_allclose(errs[-1], np.sqrt(2), atol=0.5)

    # 3. Increasing the last weight, distributions the error again
    # Not really a good test...


def test_weighted2():

    y = np.zeros((5, 2))
    y[:, 0] = np.arange(5)
    y[0, 1] = 0.5
    x = y.copy()
    x += (0.0, 1.0)
    y[-1] = (20, 0.0)

    ts = []
    for last_w in np.logspace(-5, 5, 10):
        w = np.ones(x.shape[0])
        w[-1] = last_w
        s, R, t = compute_motion(x, y, with_scale=False, w=w)
        # print(last_w, t)
        ts.append(t)

    assert abs(ts[0][0, 0]) < 1e-4
    assert abs(ts[-1][0, 0]) > 15


def test_subspace_degenerate():
    x, y, theta = random_problem2d(n=10, scale_std=0.1, noise_std=0.0)
    # lift to 3d

    x3d = np.concatenate((x, np.zeros((10, 1))), -1)
    y3d = np.concatenate((y, np.zeros((10, 1))), -1)
    s, R, t = compute_motion(x3d, y3d, with_scale=True)

    assert_transform_equal((s, R, t), theta, checkdims=[0, 1])
