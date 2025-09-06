
import numpy as np

def hankel_fir(u: np.ndarray, n: int) -> np.ndarray:
    u = np.asarray(u).reshape(-1)
    N = len(u)
    if n <= 0 or n >= N:
        raise ValueError("Order n must be in [1, N-1)")
    Phi = np.column_stack([u[n-k-1: N-k-1] for k in range(n)])
    return Phi

def hankel_arx(u: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    u = np.asarray(u).reshape(-1)
    y = np.asarray(y).reshape(-1)
    N = min(len(u), len(y))
    if n <= 0 or n >= N:
        raise ValueError("Order n must be in [1, N-1)")
    U = np.column_stack([u[n-k-1: N-k-1] for k in range(n)])
    Y = np.column_stack([y[n-k-1: N-k-1] for k in range(n)])
    return np.hstack([U, Y])
