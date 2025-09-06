
import numpy as np

def mse_loss(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))

def mdl_score(J: float, N: int, params: int):
    if J <= 0: J = 1e-12
    return float(N * np.log(J) + 2 * params * np.log(N))

def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    num = np.sqrt(np.mean((y_true - y_pred) ** 2))
    den = (y_true.max() - y_true.min()) or 1.0
    return float(1 - num / den)
