import os
import sys
# Add src directory to sys.path for module imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from sysid.hankel import hankel_fir, hankel_arx
from sysid.ls import ls_fit
from sysid.metrics import mse_loss, mdl_score, nrmse
from sysid.utils import autocorr, train_valid_split

def load_data(path):
    df = pd.read_csv(path)
    return df["u"].values, df["y"].values

def choose_order_fir(u_train, y_train, max_order):
    N = len(y_train)
    losses, mdls = [], []
    for n in range(1, max_order+1):
        Phi = hankel_fir(u_train, n)
        target = y_train[n:]
        theta, yhat = ls_fit(Phi, target)
        J = mse_loss(target, yhat)
        losses.append(J)
        mdls.append(mdl_score(J, N - n, n))
    best = int(np.argmin(mdls) + 1)
    return best, np.array(losses), np.array(mdls)

def choose_order_arx(u_train, y_train, max_order):
    N = len(y_train)
    losses, mdls = [], []
    for n in range(1, max_order+1):
        Phi = hankel_arx(u_train, y_train, n)
        target = y_train[n:]
        theta, yhat = ls_fit(Phi, target)
        J = mse_loss(target, yhat)
        mdls.append(mdl_score(J, N - n, 2*n))
        losses.append(J)
    best = int(np.argmin(mdls) + 1)
    return best, np.array(losses), np.array(mdls)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/sample_timeseries.csv")
    p.add_argument("--max-order", type=int, default=20)
    p.add_argument("--show-plots", action="store_true")
    p.add_argument("--outdir", default="outputs")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    u, y = load_data(args.csv)
    u_tr, y_tr, u_va, y_va = train_valid_split(u, y, 0.8)

    ac_y = autocorr(y_tr, max_lag=30)
    plt.figure()
    plt.stem(range(len(ac_y)), ac_y, use_line_collection=True)
    plt.title("AutoCorrelation of y (train)"); plt.xlabel("Lag"); plt.ylabel("rho"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "autocorr_y.png"))

    best_fir, loss_fir, mdl_fir = choose_order_fir(u_tr, y_tr, args.max_order)
    Phi_fir = hankel_fir(u_tr, best_fir)
    theta_fir, yhat_tr_fir = ls_fit(Phi_fir, y_tr[best_fir:])
    Phi_fir_v = hankel_fir(u_va, best_fir); yhat_va_fir = Phi_fir_v @ theta_fir
    nrmse_fir = nrmse(y_va[best_fir:], yhat_va_fir)

    best_arx, loss_arx, mdl_arx = choose_order_arx(u_tr, y_tr, args.max_order)
    Phi_arx = hankel_arx(u_tr, y_tr, best_arx)
    theta_arx, yhat_tr_arx = ls_fit(Phi_arx, y_tr[best_arx:])
    Phi_arx_v = hankel_arx(u_va, y_va, best_arx); yhat_va_arx = Phi_arx_v @ theta_arx
    nrmse_arx = nrmse(y_va[best_arx:], yhat_va_arx)

    plt.figure()
    plt.plot(range(1, args.max_order+1), mdl_fir, "-o", label="FIR MDL")
    # mark min
    min_idx_fir = int(np.argmin(mdl_fir)+1)
    min_val_fir = float(np.min(mdl_fir))
    plt.scatter([min_idx_fir],[min_val_fir])
    plt.text(min_idx_fir, min_val_fir, f" min MDL={min_idx_fir}")
    plt.plot(range(1, args.max_order+1), mdl_arx, "-o", label="ARX MDL")
    min_idx_arx = int(np.argmin(mdl_arx)+1)
    min_val_arx = float(np.min(mdl_arx))
    plt.scatter([min_idx_arx],[min_val_arx])
    plt.text(min_idx_arx, min_val_arx, f" min MDL={min_idx_arx}")
    plt.axvline(best_fir, color="k", linestyle="--", alpha=0.5, label=f"FIR* {best_fir}")
    plt.axvline(best_arx, color="gray", linestyle=":", alpha=0.8, label=f"ARX* {best_arx}")
    plt.title("Order Estimation by MDL"); plt.xlabel("order n"); plt.ylabel("MDL(n)"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "mdl_vs_n.png"))

    k = range(len(y_va))
    plt.figure()
    plt.plot(list(k)[best_fir:], y_va[best_fir:], label="y_valid")
    plt.plot(list(k)[best_fir:], yhat_va_fir, label=f"FIR(n={best_fir})")
    plt.plot(list(k)[best_arx:], yhat_va_arx, label=f"ARX(n={best_arx})")
    plt.title("Validation Fit"); plt.xlabel("sample"); plt.ylabel("amplitude"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "validation_fit.png"))

    plt.plot(list(k)[best_fir:], y_va[best_fir:], label="y_valid")
    plt.plot(list(k)[best_fir:], yhat_va_fir, label=f"FIR(n={best_fir})")
    plt.plot(list(k)[best_arx:], yhat_va_arx, label=f"ARX(n={best_arx})")
    plt.title("Validation Fit"); plt.xlabel("sample"); plt.ylabel("amplitude"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "validation_fit.png"))

        