"""AlexNet1D: per device, 1 full feature set + 9 random column subsets (>=3 columns).
Each subset: 20 independent runs; per run the best epoch (Rank-1, cosine) is kept;
mean/std are computed over the 20 runs. Cross-day: train = day 1, test = day 2.

Run from the folder containing data.py and alexnet_cnn.py:
    python run_alexnet.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from load import load_all, FEATURE_NAMES
from alexnet_cnn import AlexNet1D

try:
    from tqdm import tqdm
except ImportError:  # fallback: no progress bar, plain prints
    class tqdm:
        def __init__(self, total=None, **kw): pass
        def update(self, n=1): pass
        def set_description(self, s): pass
        def set_postfix_str(self, s): pass
        def close(self): pass
        @staticmethod
        def write(s): print(s, flush=True)

# ---------------- config ----------------
N_SUBSETS = 5
MIN_COLS = 3
N_RUNS = 10
EPOCHS = 20
EVAL_EVERY = 2         # compute Rank-1 every N epochs (and at the last one); higher = faster
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SUBSET_SEED = 123
OUT_DIR = Path("results_alexnet")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------


class AlexNetWrapper(AlexNet1D):
    """Adds (N,T,F)->(N,F,T) permutation and embed() (output of the penultimate FC layer, 1024-d)."""

    def forward(self, x):
        return super().forward(x.permute(0, 2, 1))

    def embed(self, x):
        return self.classifier[:-1](self.features(x.permute(0, 2, 1)))


def make_subsets(n_features, n_subsets, min_cols, seed):
    rng = np.random.default_rng(seed)
    subsets, seen = [], set()
    while len(subsets) < n_subsets:
        k = int(rng.integers(min_cols, n_features + 1))
        idx = tuple(sorted(rng.choice(n_features, size=k, replace=False).tolist()))
        if idx in seen or len(idx) == n_features:
            continue
        seen.add(idx)
        subsets.append(list(idx))
    return subsets


@torch.no_grad()
def embeddings(model, X, bs=256):
    model.eval()
    return np.concatenate([model.embed(torch.from_numpy(X[i:i + bs]).to(DEVICE)).cpu().numpy()
                           for i in range(0, len(X), bs)])


def rank1_cosine(E_tr, y_tr, E_te, y_te):
    norm = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    users = np.unique(y_tr)
    gallery = norm(np.stack([E_tr[y_tr == u].mean(0) for u in users]))
    return float((users[(norm(E_te) @ gallery.T).argmax(1)] == y_te).mean())


def one_run(X_tr, y_tr, X_te, y_te, num_classes, seed, on_epoch=None):
    """Train once; return best-epoch Rank-1, its epoch, and train/test embeddings at that epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AlexNetWrapper(X_tr.shape[2], num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr)

    best = dict(rank1=-1.0, epoch=-1, E_tr=None, E_te=None)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            if len(b) < 2:          # BatchNorm needs batch size > 1
                continue
            opt.zero_grad()
            loss_fn(model(Xt[b].to(DEVICE)), yt[b].to(DEVICE)).backward()
            opt.step()
        if (ep + 1) % EVAL_EVERY == 0 or ep + 1 == EPOCHS:
            E_tr, E_te = embeddings(model, X_tr), embeddings(model, X_te)
            r1 = rank1_cosine(E_tr, y_tr, E_te, y_te)
            if r1 > best["rank1"]:
                best = dict(rank1=r1, epoch=ep + 1, E_tr=E_tr, E_te=E_te)
        else:
            r1 = max(best["rank1"], 0.0)
        if on_epoch:
            on_epoch(ep + 1, r1, best["rank1"])
    return best


def settings():
    return dict(epochs=EPOCHS, eval_every=EVAL_EVERY, batch_size=BATCH_SIZE, lr=LR,
                weight_decay=WEIGHT_DECAY, subset_seed=SUBSET_SEED, n_subsets=N_SUBSETS,
                min_cols=MIN_COLS)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    data = load_all()
    n_feat = len(FEATURE_NAMES)

    configs = {"full": list(range(n_feat))}
    for i, idx in enumerate(make_subsets(n_feat, N_SUBSETS, MIN_COLS, SUBSET_SEED), 1):
        configs[f"subset{i}"] = idx
    cfg_json = {k: [FEATURE_NAMES[j] for j in v] for k, v in configs.items()}

    # --- safety check: refuse to resume if settings / subsets changed since the saved results ---
    meta = dict(settings=settings(), configs=cfg_json)
    meta_path, runs_csv = OUT_DIR / "meta.json", OUT_DIR / "runs.csv"
    if meta_path.exists() and runs_csv.exists() and json.loads(meta_path.read_text()) != meta:
        raise SystemExit(f"Settings or subsets differ from the ones in {OUT_DIR}/meta.json. "
                         f"Restore the old settings, or delete/rename {OUT_DIR}/ to start fresh.")
    meta_path.write_text(json.dumps(meta, indent=2))
    (OUT_DIR / "configs.json").write_text(json.dumps(cfg_json, indent=2))

    # --- resume: load already computed runs from runs.csv ---
    rows = []
    if runs_csv.exists():
        old = pd.read_csv(runs_csv).drop_duplicates(["device", "config", "run"])
        rows = [r for r in old.to_dict("records")
                if r["run"] < N_RUNS and (OUT_DIR / r["device"] / r["config"] / "test_embedding.npz").exists()]
        if rows:
            print(f"Resuming: {len(rows)} runs already in {runs_csv}")

    print(f"Device: {DEVICE} | {len(data)} devices x {len(configs)} subsets x {N_RUNS} runs x {EPOCHS} epochs")
    pbar = tqdm(total=len(data) * len(configs) * N_RUNS, initial=len(rows), unit="run", dynamic_ncols=True)
    for dev, d in data.items():
        num_classes = len(np.unique(d.y_train))
        for cfg, idx in configs.items():
            prev = [r for r in rows if r["device"] == dev and r["config"] == cfg]
            done_runs = {int(r["run"]) for r in prev}
            if len(done_runs) >= N_RUNS:
                continue

            X_tr = np.ascontiguousarray(d.X_train[..., idx])
            X_te = np.ascontiguousarray(d.X_test[..., idx])
            out = OUT_DIR / dev / cfg
            out.mkdir(parents=True, exist_ok=True)

            run_scores = [float(r["best_rank1"]) for r in prev]
            top_r1 = max(run_scores) if run_scores else -1.0
            npz_path = out / "test_embedding.npz"
            if prev and npz_path.exists():
                z = np.load(npz_path)
                if "rank1" in z.files:
                    top_r1 = max(top_r1, float(z["rank1"]))
            tag = f" resume {len(done_runs)}/{N_RUNS}" if done_runs else ""
            pbar.set_description(f"{dev} | {cfg} ({len(idx)} cols){tag}")

            for run in range(N_RUNS):
                if run in done_runs:
                    continue

                def on_epoch(ep, r1, best_r1, run=run):
                    avg = f"{np.mean(run_scores):.3f}" if run_scores else "-"
                    pbar.set_postfix_str(f"run {run + 1}/{N_RUNS} ep {ep}/{EPOCHS} "
                                         f"r1={r1:.3f} best={best_r1:.3f} | mean so far={avg}")

                best = one_run(X_tr, d.y_train, X_te, d.y_test, num_classes, seed=run, on_epoch=on_epoch)
                run_scores.append(best["rank1"])

                if best["rank1"] > top_r1:  # keep embeddings of the best run so far on disk
                    top_r1 = best["rank1"]
                    np.savez(out / "train_embedding.npz", embeddings=best["E_tr"], labels=d.y_train,
                             rank1=best["rank1"])
                    np.savez(npz_path, embeddings=best["E_te"], labels=d.y_test, rank1=best["rank1"])

                rows.append(dict(device=dev, config=cfg, run=run, n_cols=len(idx),
                                 best_rank1=best["rank1"], best_epoch=best["epoch"]))
                pd.DataFrame(rows).to_csv(runs_csv, index=False)
                pbar.update(1)

            tqdm.write(f"[done] {dev:7s} {cfg:9s} cols={len(idx):2d} | rank1 "
                       f"mean={np.mean(run_scores):.3f} std={np.std(run_scores, ddof=1):.3f} "
                       f"min={np.min(run_scores):.3f} max={np.max(run_scores):.3f}")

    pbar.close()
    df = pd.DataFrame(rows)
    summary = (df.groupby(["device", "config"])["best_rank1"]
                 .agg(mean="mean", std="std", max="max", min="min").reset_index())
    summary["n_cols"] = summary["config"].map({k: len(v) for k, v in configs.items()})
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print("\n", summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
