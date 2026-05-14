"""Step 2 — Inspect data, write backprop, train, save weights.

Run:  python 02_train.py --data data_v1.npz --tag v1

It loads the dataset from `--data`, saves diagnostic figures, runs the
gradient check on YOUR `my_backward()`, trains for 300 epochs (Adam,
batch 64, lr 1e-3, 90/10 train/val), and saves nav_<tag>.npz.

The function `my_backward` near the top is yours to fill in. The script
asserts that your gradients agree with numerical_gradient before it lets
training start. If the assertion fires, fix the bug.

This script is the baseline. Once you've passed the gradient check and got
your first benchmark, the iteration loop is yours: change the architecture
in `drive2win/nn.py`, change the data, change the training
schedule, retrain, rebenchmark, commit, repeat.
"""
from __future__ import annotations

import argparse
import numpy as np

from drive2win import nn as nn_mod
from drive2win import viz
from drive2win.normalize import (
    normalize_states, FEATURE_NAMES, N_FEATURES, N_ACTIONS,
)


# =========================================================================
# BACKWARD (FIXED)
# =========================================================================
def my_backward(X, y, w, cache  ):
    W1, W2, W3 =w["W1"],w ["W2"],w ["W3"]

    z1 = cache["z1"]
    a1 = cache["a1"]
    z2 = cache["z2"]
    a2 = cache["a2"]
    z3 = cache["z3"]
    y_pred = cache["y"]

    N = X.shape[0]
    dy = 2.0 * (y_pred - y) / (N * y_pred.shape[1])

    dz3 = dy * (1 - np.tanh(z3) ** 2)
    dW3 = a2.T @ dz3
    db3 = dz3.sum(axis=0)

    da2 = dz3 @ W3.T
    dz2 = da2 * (z2 > 0)
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
        "W3": dW3,
        "b3": db3,
    }


# =========================================================================
# GRADIENT CHECK
# =========================================================================
def gradient_check():
    rng = np.random.default_rng(0)

    w64 = {k: v.astype(np.float64) for k, v in nn_mod.init_weights(seed=0).items()}

    x = rng.normal(size=(8, N_FEATURES)).astype(np.float64)
    y = rng.uniform(-1, 1, size=(8, N_ACTIONS)).astype(np.float64)

    cache = nn_mod.forward_all(x, w64)
    grads = my_backward(x, y, w64, cache)

    print("\ngradient check (max relative error per parameter):")

    for key in w64:
        max_err = 0.0
        flat = w64[key].size

        for _ in range(5):
            idx = np.unravel_index(rng.integers(0, flat), w64[key].shape)

            num = nn_mod.numerical_gradient(x, y, w64, key, idx)
            ana = grads[key][idx]

            denom = max(1e-12, abs(num) + abs(ana))
            err = abs(num - ana) / denom
            max_err = max(max_err, err)

        print(f"  {key}: {max_err:.2e}")

        assert max_err < 1e-4, f"backward() bug in {key}"


# =========================================================================
# DATA INSPECTION
# =========================================================================
def inspect_dataset(states_raw, actions, tag: str):
    print("\nfeature ranges (raw):")
    for i, name in enumerate(FEATURE_NAMES):
        col = states_raw[:, i]
        print(f"  {name:>20s}: [{col.min():+7.2f}, {col.max():+7.2f}]   "
              f"mean={col.mean():+.2f}  std={col.std():.2f}")

    viz.plot_action_histograms(actions, out=f"fig_actions_{tag}.png")
    viz.plot_heading_vs_steering(states_raw, actions, out=f"fig_heading_{tag}.png")


# =========================================================================
# TRAIN
# =========================================================================
def train(X, Y, epochs=300, lr=1e-3, batch_size=64, val_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    N = len(X)

    perm = rng.permutation(N)
    n_val = max(1, int(N * val_frac))

    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xva, Yva = X[val_idx], Y[val_idx]

    w = nn_mod.init_weights(seed=seed)
    state = nn_mod.init_adam(w)

    best = {k: v.copy() for k, v in w.items()}
    best_val = float("inf")

    for epoch in range(epochs):
        idx = rng.permutation(len(Xtr))
        Xs, Ys = Xtr[idx], Ytr[idx]

        for i in range(0, len(Xs), batch_size):
            xb, yb = Xs[i:i+batch_size], Ys[i:i+batch_size]

            cache = nn_mod.forward_all(xb, w)
            grads = my_backward(xb, yb, w, cache)

            nn_mod.adam_step(w, grads, state, lr=lr)

        val = nn_mod.mse_loss(nn_mod.forward(Xva, w), Yva)

        if val < best_val:
            best_val = val
            best = {k: v.copy() for k, v in w.items()}

        if epoch % 25 == 0:
            print(f"epoch {epoch} val={val:.5f} best={best_val:.5f}")

    return best


# =========================================================================
# MAIN
# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v1.npz")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    states_raw, actions = d["states"], d["actions"]

    print(f"raw states  : {states_raw.shape}")
    print(f"raw actions : {actions.shape}")

    inspect_dataset(states_raw, actions, tag=args.tag)

    X = normalize_states(states_raw)
    Y = actions.astype(np.float64)   # FIX KRYESOR

    print(f"\nX range : [{X.min():+.2f}, {X.max():+.2f}]")
    print(f"Y range : [{Y.min():+.2f}, {Y.max():+.2f}]")

    gradient_check()

    best = train(X, Y, epochs=args.epochs, lr=args.lr, batch_size=args.batch)

    nn_mod.save(best, f"nav_{args.tag}.npz")
    print(f"Saved nav_{args.tag}.npz")


if __name__ == "__main__":
    main()
