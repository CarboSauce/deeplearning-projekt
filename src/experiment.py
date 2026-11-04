import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from data import DEVICES, load_device
from metrics import (confusion_matrix, cosine_similarity_matrix, per_user_rank1,
                     predict, rank1, save_confusion)
from models import MODEL_NAMES, build_model

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = {"MultiBranchCNN": 1e-3, "AlexNetCNN": 1e-3, "Transformer": 3e-4}
WEIGHT_DECAY = 1e-4

N_RUNS = 5                  # trainings per ablation condition
VAL_THROWS_PER_USER = 2     # carved out of day 1, used only for choosing the best epoch
SEED = 42

# Ablation conditions: which features go into the model (all 41 users in every condition)
ABLATIONS = {
    "positions": lambda name: name.endswith(("_px", "_py", "_pz")),   # controllers + headset positions
    "angles":    lambda name: name.endswith(("_ax", "_ay", "_az")),   # controllers + headset Euler angles
    "all":       lambda name: True,                                   # everything, incl. right_trigger
}
CONDITIONS = list(ABLATIONS)
MODELS = list(MODEL_NAMES)   # MultiBranchCNN, AlexNetCNN, Transformer

EPOCH_LOG_EVERY = 10        # print a heartbeat line every N epochs (0 = off)
PROGRESS = None             # set by main.py to a progress.Progress instance

OUT_DIR = Path("results")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# --------------------------------------------------
# Data helpers
# --------------------------------------------------
def select_features(d, condition):
    """Slice the (already standardized) features for one ablation condition."""
    keep = [i for i, n in enumerate(d.feature_names) if ABLATIONS[condition](n)]
    names = [d.feature_names[i] for i in keep]
    return d.X_train[..., keep], d.X_test[..., keep], names


def split_val(y, rng):
    """Per-user split of day-1 throws: VAL_THROWS_PER_USER (after shuffling) go to validation."""
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        val_idx.extend(idx[:VAL_THROWS_PER_USER])
        train_idx.extend(idx[VAL_THROWS_PER_USER:])
    return np.array(train_idx), np.array(val_idx)


def to_tensor(X):
    """(N, T, F) numpy -> (N, F, T) tensor on DEVICE (Conv1d layout)."""
    return torch.tensor(X, dtype=torch.float32).permute(0, 2, 1).contiguous().to(DEVICE)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------
@torch.no_grad()
def embed_all(model, X):
    model.eval()
    return model.embed(X).cpu().numpy()


@torch.no_grad()
def evaluate_loss_acc(model, X, y, criterion):
    model.eval()
    logits = model(X)
    return criterion(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


# --------------------------------------------------
# One training run
# --------------------------------------------------
def log(message):
    if PROGRESS is not None:
        PROGRESS.log(message)
    else:
        print(message, flush=True)


def train_run(model_name, Xtr, ytr, Xva, yva, num_classes, feature_names, seed):
    torch.manual_seed(seed)
    model = build_model(model_name, feature_names, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE[model_name],
                                 weight_decay=WEIGHT_DECAY)

    best_val_loss, best_state, best_epoch = float("inf"), None, -1

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr), device=DEVICE)
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2:                       # BatchNorm needs more than one sample
                continue
            optimizer.zero_grad()
            loss = criterion(model(Xtr[idx]), ytr[idx])
            loss.backward()
            optimizer.step()

        val_loss, _ = evaluate_loss_acc(model, Xva, yva, criterion)
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())

        if EPOCH_LOG_EVERY and (epoch + 1) % EPOCH_LOG_EVERY == 0:
            log(f"      epoch {epoch + 1:>3}/{EPOCHS} | val loss {val_loss:.4f} | "
                f"best {best_val_loss:.4f} (epoch {best_epoch + 1})")

    model.load_state_dict(best_state)              # best epoch by validation loss (day 1 only)
    return model, best_val_loss, best_epoch


# --------------------------------------------------
# Ablation experiment for one model on one device
# --------------------------------------------------
def run_device(model_name, device_name):
    d = load_device(device_name)
    y_day1, y_te = d.y_train, d.y_test
    num_classes = len(np.unique(y_day1))
    y_te_t = torch.tensor(y_te, device=DEVICE)

    rows, per_user_rows = [], []
    for condition in CONDITIONS:
        X_day1, X_test, feature_names = select_features(d, condition)
        X_day1_t, X_te_t = to_tensor(X_day1), to_tensor(X_test)

        best = {"val_loss": float("inf")}          # best run by validation loss, for the .npz files
        cond_dir = OUT_DIR / model_name / device_name / condition
        (cond_dir / "runs").mkdir(parents=True, exist_ok=True)
        cm_sum = np.zeros((num_classes, num_classes), dtype=np.int64)

        for run in range(N_RUNS):
            run_start = time.time()
            seed = SEED + run
            tr_idx, va_idx = split_val(y_day1, np.random.default_rng(seed))

            Xtr, ytr = X_day1_t[tr_idx], torch.tensor(y_day1[tr_idx], device=DEVICE)
            Xva, yva = X_day1_t[va_idx], torch.tensor(y_day1[va_idx], device=DEVICE)

            model, val_loss, best_epoch = train_run(model_name, Xtr, ytr, Xva, yva, num_classes,
                                                    feature_names, seed)

            _, test_acc = evaluate_loss_acc(model, X_te_t, y_te_t, nn.CrossEntropyLoss())
            gallery_emb = embed_all(model, X_day1_t)          # all day-1 throws
            test_emb = embed_all(model, X_te_t)

            sim, classes = cosine_similarity_matrix(gallery_emb, y_day1, test_emb)
            pred = predict(sim, classes)
            test_rank1 = rank1(pred, y_te)
            cm_sum += confusion_matrix(pred, y_te, classes)

            for user, user_rank1 in zip(classes, per_user_rank1(pred, y_te, classes)):
                per_user_rows.append(dict(model=model_name, device=device_name, condition=condition,
                                          run=run, user=int(user), rank1=float(user_rank1)))

            # raw per-run results: similarity matrix (probes x users), predictions, true labels
            np.savez_compressed(cond_dir / "runs" / f"run_{run}.npz",
                                similarity=sim.astype(np.float32), predictions=pred,
                                labels=y_te, classes=classes)

            rows.append(dict(
                model=model_name, device=device_name, condition=condition, n_features=len(feature_names), run=run,
                best_epoch=best_epoch, val_loss=val_loss,
                test_acc_softmax=test_acc, test_rank1_cosine=test_rank1,
            ))

            if val_loss < best["val_loss"]:
                best = dict(val_loss=val_loss, train_emb=gallery_emb, test_emb=test_emb)

            if PROGRESS is not None:
                PROGRESS.step(f"{model_name} | {device_name} | {condition} | run {run + 1}/{N_RUNS}",
                              time.time() - run_start,
                              f"Rank-1 {test_rank1:.3f} | best epoch {best_epoch + 1}")

        r = [x["test_rank1_cosine"] for x in rows if x["condition"] == condition]
        log(f"[{model_name} | {device_name}] {condition:>9} ({len(feature_names):>2} features) "
              f"Rank-1: {np.mean(r):.3f} +- {np.std(r):.3f}")

        # Embeddings for RSA: one pair of files per model x device x condition, same sample order everywhere
        out = cond_dir
        np.savez(out / "train_embedding.npz", embeddings=best["train_emb"], labels=y_day1)
        np.savez(out / "test_embedding.npz", embeddings=best["test_emb"], labels=y_te)

        # Confusion matrix summed over all runs of this condition (counts CSV + normalized PNG)
        save_confusion(cm_sum, out / "confusion_matrix",
                       f"{model_name} | {device_name} | {condition} (sum of {N_RUNS} runs)")

    return rows, per_user_rows