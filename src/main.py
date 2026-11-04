import sys
from pathlib import Path

# Works whether main.py sits in the project root or inside src/
HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE / "src"):
    if (candidate / "experiment.py").exists():
        sys.path.insert(0, str(candidate))
        break

import pandas as pd

import experiment
from data import DEVICES
from models import MODEL_NAMES
from progress import Progress

# --------------------------------------------------
# Settings - edit here, then just press Run
# --------------------------------------------------
RUN_MODELS = MODEL_NAMES                 # e.g. ["Transformer"] to run just one model
RUN_DEVICES = DEVICES                    # e.g. ["Vive"] for a single device
CONDITIONS = list(experiment.ABLATIONS)  # positions, angles, all
RUNS_PER_CONDITION = 5
EPOCHS = 50
EPOCH_LOG_EVERY = 10                     # heartbeat line every N epochs inside a run (0 = off)
OUT_DIR = HERE / "results"

QUICK_TEST = False                       # True = tiny run just to check that everything works
if QUICK_TEST:
    RUNS_PER_CONDITION, EPOCHS = 1, 3


def build_tables(runs):
    """Per-run table (all models) -> (summary with mean/std, pretty 'mean ± std' table in %)."""
    summary = (
        runs.groupby(["model", "device", "condition"], sort=False)
        .agg(n_features=("n_features", "first"),
             rank1_mean=("test_rank1_cosine", "mean"), rank1_std=("test_rank1_cosine", "std"),
             acc_mean=("test_acc_softmax", "mean"), acc_std=("test_acc_softmax", "std"),
             best_epoch_mean=("best_epoch", "mean"))
        .reset_index()
    )

    pretty = summary.copy()
    pretty["rank1"] = [f"{m * 100:.2f} ± {s * 100:.2f}"
                       for m, s in zip(pretty.rank1_mean, pretty.rank1_std.fillna(0))]
    pretty = pretty.pivot(index=["model", "condition"], columns="device", values="rank1")

    all_conditions = list(experiment.ABLATIONS)
    row_order = [(m, c) for m in MODEL_NAMES for c in all_conditions if (m, c) in pretty.index]
    pretty = pretty.loc[row_order, [d for d in DEVICES if d in pretty.columns]]
    return summary, pretty


def load_all_runs():
    """Collect the per-run tables of every model that has results on disk."""
    frames = [pd.read_csv(OUT_DIR / m / "ablation_runs.csv")
              for m in MODEL_NAMES if (OUT_DIR / m / "ablation_runs.csv").exists()]
    return pd.concat(frames, ignore_index=True)


def save_per_user_summary():
    """Mean/std over runs of each user's Rank-1, for every model x device x condition."""
    frames = [pd.read_csv(OUT_DIR / m / "per_user.csv")
              for m in MODEL_NAMES if (OUT_DIR / m / "per_user.csv").exists()]
    if not frames:
        return
    per_user = pd.concat(frames, ignore_index=True)
    summary = (
        per_user.groupby(["model", "device", "condition", "user"], sort=False)
        .agg(rank1_mean=("rank1", "mean"), rank1_std=("rank1", "std"))
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "per_user_summary.csv", index=False)


def save_combined_tables():
    save_per_user_summary()
    runs = load_all_runs()
    summary, pretty = build_tables(runs)
    summary.to_csv(OUT_DIR / "ablation_summary.csv", index=False)
    pretty.to_csv(OUT_DIR / "ablation_table.csv")
    (OUT_DIR / "ablation_table.txt").write_text(
        "Rank-1 [%] (mean ± std)\n\n" + pretty.to_string(), encoding="utf-8")
    return pretty


def main():
    # push the settings above into experiment.py
    experiment.N_RUNS = RUNS_PER_CONDITION
    experiment.EPOCHS = EPOCHS
    experiment.CONDITIONS = CONDITIONS
    experiment.OUT_DIR = OUT_DIR
    experiment.EPOCH_LOG_EVERY = EPOCH_LOG_EVERY
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(RUN_MODELS) * len(RUN_DEVICES) * len(CONDITIONS) * RUNS_PER_CONDITION
    progress = Progress(total_runs, OUT_DIR / "progress.log")
    experiment.PROGRESS = progress

    print("Using device:", experiment.DEVICE)
    print(f"Models: {RUN_MODELS} | devices: {RUN_DEVICES} | conditions: {CONDITIONS} | "
          f"runs/condition: {RUNS_PER_CONDITION} | epochs: {EPOCHS}\n")

    for model_name in RUN_MODELS:
        model_dir = OUT_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        rows, per_user_rows = [], []
        for device_name in RUN_DEVICES:
            device_rows, device_per_user = experiment.run_device(model_name, device_name)
            rows.extend(device_rows)
            per_user_rows.extend(device_per_user)
            # save after every device so a crash doesn't lose finished work
            pd.DataFrame(rows).to_csv(model_dir / "ablation_runs.csv", index=False)
            pd.DataFrame(per_user_rows).to_csv(model_dir / "per_user.csv", index=False)
            save_combined_tables()

    pretty = save_combined_tables()
    progress.finish()
    print("\nRank-1 [%] (mean ± std)\n" + pretty.to_string())
    print(f"\nSaved tables to: {OUT_DIR}")


if __name__ == "__main__":
    main()