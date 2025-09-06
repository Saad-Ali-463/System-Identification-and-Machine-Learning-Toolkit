import os
import sys
# Add src directory to sys.path for module imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from classify.newton_logreg import fit_ovr_newton, predict_proba_ovr

def load_data(path):
    df = pd.read_csv(path)
    X = df[['x1','x2']].values
    y = df['y'].values.astype(int)
    return X, y

def split(X, y, frac=0.8):
    N = len(y); n = int(N*frac)
    return X[:n], y[:n], X[n:], y[n:]

def add_bias(X):  # phi(t) = [1, x1, x2]^T
    X = np.asarray(X)
    return np.hstack([np.ones((X.shape[0],1)), X])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/sample_classification.csv')
    ap.add_argument('--max-iter', type=int, default=40)
    ap.add_argument('--lr', type=float, default=1.0)
    ap.add_argument('--show-plots', action='store_true')
    ap.add_argument('--outdir', default='outputs')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    X, y = load_data(args.csv)
    Xtr, ytr, Xte, yte = split(X, y, 0.8)

    Phi_tr = add_bias(Xtr)
    Phi_te = add_bias(Xte)

    theta, histories, classes = fit_ovr_newton(Phi_tr, ytr, max_iter=args.max_iter, lr=args.lr)

    # Plot loss history for class with highest index (often class 3)
    plt.figure()
    last_hist = histories[-1]
    plt.plot(range(1, len(last_hist)+1), last_hist, '-o')
    plt.title("Loss function J(theta) during Newton updates (one class)")
    plt.xlabel("iteration")
    plt.ylabel("loss J(theta)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "newton_loss.png"))

    # Predictions & accuracy
    P_tr = predict_proba_ovr(Phi_tr, theta)
    P_te = predict_proba_ovr(Phi_te, theta)
    yhat_tr = P_tr.argmax(axis=1)
    yhat_te = P_te.argmax(axis=1)
    acc_tr = (yhat_tr == ytr).mean()
    acc_te = (yhat_te == yte).mean()

    # Scatter predictions for test set
    plt.figure()
    plt.scatter(Xte[:,0], Xte[:,1], c=yhat_te, marker='o')
    plt.title("Predicted classes (test)")
    plt.xlabel("x1"); plt.ylabel("x2"); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "classification_pred_test.png"))

    # Mark misclassified
    mis = yhat_te != yte
    if mis.any():
        plt.figure()
        plt.scatter(Xte[~mis,0], Xte[~mis,1], marker='o')
        plt.scatter(Xte[mis,0], Xte[mis,1], marker='x')
        plt.title("Test predictions: 'o' correct, 'x' misclassified")
        plt.xlabel("x1"); plt.ylabel("x2"); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "classification_correct_vs_missed.png"))

    print(f"Training accuracy: {acc_tr*100:.2f}%")
    print(f"Test accuracy: {acc_te*100:.2f}%")
    print(f"Saved figures -> {args.outdir}/")

    if args.show_plots: plt.show()

if __name__ == '__main__':
    main()
