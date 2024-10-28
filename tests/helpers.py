import numpy as np


def assert_same_transform(actual, expected, checkdims: list[int] = None, atol=1e-6):
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
