import numpy as np


def assert_transform_equal(actual, expected, checkdims: list[int] = None, atol=1e-6):
    scale, R, t = actual
    scale_hat, R_hat, t_hat = expected

    D = R.shape[0]

    if checkdims is None:
        checkdims = list(range(D))

    remaindims = [i for i in range(D) if i not in checkdims]

    # scale
    assert np.allclose(
        scale,
        scale_hat,
        atol=atol,
    )

    # rotation up to dim check_dims
    assert np.allclose(
        R[checkdims, checkdims],
        R_hat[checkdims, checkdims],
        atol=atol,
    )
    # rotation after checkdims: should be identity
    assert np.allclose(
        R[remaindims, remaindims],
        np.eye(D - len(checkdims)),
        atol=atol,
    )

    # translation
    assert np.allclose(
        t[0, checkdims],
        t_hat[0, checkdims],
        atol=atol,
    )

    # remainder after checkdims should be zero
    assert np.allclose(
        t[0, remaindims],
        0.0,
        atol=atol,
    )


def random_problem2d(
    n: int = 10,
    scale_std: float = 1e-1,
    t_std: float = 1e-1,
    noise_std: float = 0.0,
    angle_max: float = (2 * np.pi),
):
    x = np.random.randn(n, 2)

    t = np.random.randn(1, 2) * t_std
    a = np.random.rand() * angle_max
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    s = 1.0 + np.random.randn() * scale_std

    y = s * (x @ R.T) + t

    if noise_std > 0:
        x += np.random.randn(*x.shape) * noise_std

    return x, y, (s, R, t)
