
import numpy as np

def autocorr(x, max_lag=50, normalize=True):
    x = np.asarray(x).reshape(-1)
    x = x - np.mean(x)
    N = len(x)
    ac = np.array([np.sum(x[:N-k] * x[k:]) for k in range(max_lag+1)], dtype=float)
    if normalize and ac[0] != 0: ac = ac / ac[0]
    return ac

def crosscorr(x, y, max_lag=50):
    """Cross-correlation r_xy(k) for k in [-max_lag..max_lag], normalized."""
    x = np.asarray(x).reshape(-1) - np.mean(x)
    y = np.asarray(y).reshape(-1) - np.mean(y)
    N = min(len(x), len(y))
    x, y = x[:N], y[:N]
    denom = (np.std(x) * np.std(y)) or 1.0
    cc = []
    lags = range(-max_lag, max_lag+1)
    for k in lags:
        if k >= 0:
            cc.append(np.sum(x[k:] * y[:N-k]) / (N * denom))
        else:
            kk = -k
            cc.append(np.sum(x[:N-kk] * y[kk:]) / (N * denom))
    return np.array(list(lags)), np.array(cc)

def train_valid_split(u, y, frac=0.8):
    N = min(len(u), len(y))
    n_train = int(N * frac)
    return (u[:n_train], y[:n_train], u[n_train:], y[n_train:])
