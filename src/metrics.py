"""Torch-free evaluation helpers: similarity matrix, Rank-1, per-user results, confusion matrix."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")          # only save figures, never open a window
import matplotlib.pyplot as plt


def cosine_similarity_matrix(gallery_emb, gallery_y, probe_emb):
    """Gallery = mean embedding per user (day 1). Returns (similarity, classes) where
    similarity[i, j] = cosine(probe i, template of classes[j]), shape (n_probes, n_users)."""
    classes = np.unique(gallery_y)
    templates = np.stack([gallery_emb[gallery_y == c].mean(axis=0) for c in classes])
    templates = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + 1e-12)
    probes = probe_emb / (np.linalg.norm(probe_emb, axis=1, keepdims=True) + 1e-12)
    return probes @ templates.T, classes


def predict(similarity, classes):
    """Most similar template = predicted user (Rank-1 decision)."""
    return classes[similarity.argmax(axis=1)]


def rank1(predictions, probe_y):
    return float((predictions == probe_y).mean())


def confusion_matrix(predictions, probe_y, classes):
    """Counts, rows = true user, columns = predicted user (ordered like `classes`)."""
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(cm, (np.searchsorted(classes, probe_y), np.searchsorted(classes, predictions)), 1)
    return cm


def per_user_rank1(predictions, probe_y, classes):
    """Fraction of each user's probes that were recognized correctly."""
    return np.array([(predictions[probe_y == c] == c).mean() for c in classes])


def save_confusion(cm, out_base, title):
    """Save counts as CSV and a row-normalized heatmap as PNG (out_base without extension)."""
    pd.DataFrame(cm).to_csv(f"{out_base}.csv")
    normalized = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set_xlabel("Predicted user")
    ax.set_ylabel("True user")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Fraction of the user's probes")
    fig.savefig(f"{out_base}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)