
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def nll(X, y, theta):
    # Negative log-likelihood (binary)
    z = X @ theta
    p = sigmoid(z)
    eps = 1e-12
    return -np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))

def fit_binary_newton(X, y, theta0=None, max_iter=50, tol=1e-6, lr=1.0):
    m, n = X.shape
    if theta0 is None:
        theta = np.zeros((n, 1))
    else:
        theta = theta0.reshape(n, 1).astype(float)
    history = []
    for it in range(max_iter):
        z = X @ theta
        p = sigmoid(z)
        W = (p * (1 - p)).reshape(-1, 1)
        grad = X.T @ (p - y.reshape(-1,1)) / m
        H = (X.T * W.reshape(-1)).dot(X) / m  # X^T W X / m
        # Solve H * step = grad
        step = np.linalg.solve(H + 1e-8*np.eye(n), grad)
        theta_new = theta - lr * step
        history.append(nll(X, y, theta_new))
        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new
    return theta.reshape(-1,1), history

def fit_ovr_newton(X, y, classes=None, max_iter=50, lr=1.0):
    """One-vs-rest logistic regression with Newton's method.
    X: (m, n) with bias term already included (first column ones)
    y: (m,) integer labels in {0..C-1}
    Returns theta: (n, C), histories: list of per-class loss history
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    m, n = X.shape
    if classes is None:
        classes = np.unique(y)
    C = len(classes)
    theta_all = np.zeros((n, C))
    histories = []
    for ci, cls in enumerate(classes):
        yy = (y == cls).astype(float)
        theta, hist = fit_binary_newton(X, yy, max_iter=max_iter, lr=lr)
        theta_all[:, ci] = theta.ravel()
        histories.append(hist)
    return theta_all, histories, classes

def predict_proba_ovr(X, theta):
    # X: (m,n) with bias; theta:(n,C)
    z = X @ theta
    return sigmoid(z)  # per-class probs in OVR sense
