
import numpy as np

def ls_fit(Phi: np.ndarray, target: np.ndarray):
    Phi = np.asarray(Phi)
    target = np.asarray(target).reshape(-1, 1)
    theta, *_ = np.linalg.lstsq(Phi, target, rcond=None)
    y_hat = Phi @ theta
    return theta.reshape(-1), y_hat.reshape(-1)
