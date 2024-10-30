import numpy as np


def compose_theta(theta):
    s, r, t = theta
    d = r.shape[0]
    T = np.eye(d + 1)
    S = np.diag([s] * d + [1.0])
    R = np.eye(d + 1)
    T[:-1, d:] = t.T
    R[:-1, :-1] = r

    return T @ S @ R


def decompose_matrix(m):
    d = m.shape[0] - 1
    t = m[:-1, d:].T
    r = m[:-1, :-1]
    s = np.linalg.norm(r[:, 0]).item()
    r /= s
    return s, r, t


def incremental_to_cummulative_motions(
    ts: list[tuple[float, np.ndarray, np.ndarray]],
):
    assert len(ts) > 0
    d = ts[0][1].shape[0]
    t = np.eye(d + 1)
    cum = []
    for theta in ts:
        m = compose_theta(theta)
        t = m @ t
        cum.append(decompose_matrix(t))
    return cum
