"""Representational Similarity Analysis (RSA) on the saved test embeddings (best run per condition).

Reads  results/{model}/{device}/{condition}/test_embedding.npz  (written by experiment.py)
Writes results/rsa/*.csv and *.png

Two levels:
  * throw-level RDMs  (410 x 410) -> compare models / conditions WITHIN one device
                                     (same physical throws, same sample order)
  * user-level RDMs   (41 x 41)   -> compare everything, including ACROSS devices
                                     (users are shared between devices, throws are not)
Representations are compared with Spearman correlation of the RDM upper triangles.
No torch needed, runs in seconds.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

MODELS = ["MultiBranchCNN", "AlexNetCNN", "Transformer"]
DEVICES = ["Quest", "Vive", "Cosmos"]
CONDITIONS = ["positions", "angles", "all"]

SHORT_MODEL = {"MultiBranchCNN": "MB", "AlexNetCNN": "AN", "Transformer": "TR"}
SHORT_COND = {"positions": "pos", "angles": "ang", "all": "all"}


# --------------------------------------------------
# Loading
# --------------------------------------------------
def find_results_dir():
    here = Path(__file__).resolve().parent
    for candidate in (here / "results", here.parent / "results"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find a 'results' folder next to this file or one level up.")


def load_representations(results_dir):
    """{(device, model, condition): (embeddings (N, D), labels (N,))}, ordered device > model > condition."""
    reps = {}
    for device in DEVICES:
        for model in MODELS:
            for condition in CONDITIONS:
                path = results_dir / model / device / condition / "test_embedding.npz"
                if not path.exists():
                    print(f"  missing: {path}")
                    continue
                data = np.load(path)
                reps[(device, model, condition)] = (data["embeddings"], data["labels"])
    return reps


# --------------------------------------------------
# RDMs and comparison
# --------------------------------------------------
def rdm(embeddings):
    """Cosine-distance RDM (1 - cosine similarity), the same notion of similarity as Rank-1."""
    e = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    return 1.0 - e @ e.T


def upper_triangle(matrix):
    return matrix[np.triu_indices_from(matrix, k=1)]


def spearman_matrix(vectors):
    """Pairwise Spearman correlation between vectors (rank them, then Pearson)."""
    ranks = np.vstack([rankdata(v) for v in vectors])
    return np.corrcoef(ranks)


def user_templates(embeddings, labels):
    """Mean embedding per user -> (n_users, D), ordered by user id."""
    users = np.unique(labels)
    return np.stack([embeddings[labels == u].mean(axis=0) for u in users])


def label_of(key):
    device, model, condition = key
    return f"{device}|{SHORT_MODEL[model]}|{SHORT_COND[condition]}"


def plot_matrix(matrix, labels, title, path):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, 0.32 * n + 2),) * 2)
    im = ax.imshow(matrix, cmap="viridis", vmin=min(0.0, float(matrix.min())), vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8 if n > 12 else 9)
    ax.set_yticklabels(labels, fontsize=8 if n > 12 else 9)
    if n <= 12:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="white" if matrix[i, j] < 0.6 else "black", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Spearman correlation of RDMs", fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------
# Analyses
# --------------------------------------------------
def throw_level_per_device(reps, out_dir):
    """Models x conditions within each device, plus correlation with the 'identity' RDM
    (0 = same user, 1 = different user), i.e. how strongly each representation is organized by user."""
    identity_rows = []
    for device in DEVICES:
        keys = [k for k in reps if k[0] == device]
        if len(keys) < 2:
            continue

        labels = reps[keys[0]][1]
        for k in keys:
            if not np.array_equal(reps[k][1], labels):
                raise ValueError(f"Sample order differs between representations on {device}.")

        vectors = [upper_triangle(rdm(reps[k][0])) for k in keys]
        identity = upper_triangle((labels[:, None] != labels[None, :]).astype(float))

        names = [f"{SHORT_MODEL[m]}|{SHORT_COND[c]}" for _, m, c in keys]
        matrix = spearman_matrix(vectors)
        pd.DataFrame(matrix, index=names, columns=names).to_csv(out_dir / f"rsa_throw_level_{device}.csv")
        plot_matrix(matrix, names, f"RSA, throw level: {device}", out_dir / f"rsa_throw_level_{device}.png")

        for k, v in zip(keys, vectors):
            identity_rows.append(dict(device=k[0], model=k[1], condition=k[2],
                                      rsa_with_identity=float(spearman_matrix([v, identity])[0, 1])))

    identity_table = pd.DataFrame(identity_rows)
    identity_table.to_csv(out_dir / "rsa_identity.csv", index=False)
    return identity_table


def user_level_all(reps, out_dir):
    """All representations (all devices) compared through 41 x 41 user-template RDMs."""
    keys = list(reps)
    vectors = [upper_triangle(rdm(user_templates(*reps[k]))) for k in keys]
    matrix = spearman_matrix(vectors)
    names = [label_of(k) for k in keys]
    pd.DataFrame(matrix, index=names, columns=names).to_csv(out_dir / "rsa_user_level_all.csv")
    plot_matrix(matrix, names, "RSA, user level: all representations", out_dir / "rsa_user_level_all.png")

    # Average similarity for three kinds of pairs, to turn the big matrix into conclusions
    groups = {"between models (same device, same condition)": [],
              "between conditions (same device, same model)": [],
              "between devices (same model, same condition)": []}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (d1, m1, c1), (d2, m2, c2) = keys[i], keys[j]
            if d1 == d2 and c1 == c2 and m1 != m2:
                groups["between models (same device, same condition)"].append(matrix[i, j])
            elif d1 == d2 and m1 == m2 and c1 != c2:
                groups["between conditions (same device, same model)"].append(matrix[i, j])
            elif m1 == m2 and c1 == c2 and d1 != d2:
                groups["between devices (same model, same condition)"].append(matrix[i, j])

    summary = pd.DataFrame([dict(comparison=name, n_pairs=len(v), mean_spearman=float(np.mean(v)),
                                 std_spearman=float(np.std(v)))
                            for name, v in groups.items() if v])
    summary.to_csv(out_dir / "rsa_summary.csv", index=False)
    return summary


def main(results_dir=None):
    results_dir = Path(results_dir) if results_dir else find_results_dir()
    out_dir = results_dir / "rsa"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading embeddings from: {results_dir}")
    reps = load_representations(results_dir)
    print(f"Loaded {len(reps)} representations\n")
    if len(reps) < 2:
        raise SystemExit("Need at least two representations - run main.py first.")

    identity_table = throw_level_per_device(reps, out_dir)
    print("RSA with the identity RDM (higher = more organized by user):")
    print(identity_table.round(3).to_string(index=False))

    summary = user_level_all(reps, out_dir)
    print("\nAverage RDM similarity (user level):")
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()