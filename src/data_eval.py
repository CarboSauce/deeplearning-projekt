from typing import Any

import numpy as np
import pandas as pd
import sweetviz as sv
from pathlib import Path

import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FILES = ["Quest1", "Quest2", "Vive1", "Vive2", "Cosmos1", "Cosmos2"]
datasets = {}
parts = ["px", "py", "pz", "ax", "ay", "az", "trigger"]            # position, Euler angles, trigger
columns = [f"{dev}_{p}" for dev in ["right", "head", "left"] for p in parts]   # same order as in the files

def load_data() -> dict[Any, Any]:
    datasets = {}
    for name in FILES:
        arr = np.load(DATA_DIR / f"{name}.npy")
        users, throws, steps, _ = arr.shape

        df = pd.DataFrame(arr.reshape(-1, arr.shape[-1]), columns=columns)
        df.insert(0, "user",  np.repeat(np.arange(users), throws * steps))
        df.insert(1, "throw", np.tile(np.repeat(np.arange(throws), steps), users))
        df.insert(2, "t",     np.tile(np.arange(steps), users * throws))
        datasets[name] = df
    df_full = pd.concat(datasets.values(), ignore_index=True)
    datasets["Full"] = df_full
    return datasets.copy()

def clean_data():
    datasets = load_data()
    for name, df in datasets.items():
        df.drop(columns=["head_trigger", "left_trigger"], inplace=True)

    return datasets

out = Path("sweetviz_reports")
out.mkdir(exist_ok=True)
for name, df in datasets.items():
    data = df.drop(columns=["user", "throw", "t"])  # skip the index-like columns
    report = sv.analyze([data, name])
    report.show_html(str(out / f"{name}.html"))
    # per-column stats







import matplotlib.pyplot as plt

def plot_throw(datasets, name, user, throw, trigger_threshold=0.5, save_path=None):
    """3D trajectory of right controller, headset and left controller for one throw.
    name: dataset key, e.g. "Vive1" (system + day).
    Blue dot = right trigger pressed, red dot = left trigger pressed."""
    df = datasets[name]
    d = df[(df.user == user) & (df.throw == throw)].sort_values("t")
    if d.empty:
        raise ValueError(f"No data for {name}, user {user}, throw {throw}")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # (column prefix, label, line colour, trigger dot colour)
    devices = [("right", "Right controller", "tab:orange", "blue"),
               ("head",  "Headset",          "tab:green",  None),
               ("left",  "Left controller",  "tab:purple", "red")]

    for prefix, label, line_color, dot_color in devices:
        # plot (x, z, y) so that the data's height axis (y) is vertical in the figure
        x, z, y = d[f"{prefix}_px"].values, d[f"{prefix}_pz"].values, d[f"{prefix}_py"].values
        ax.plot(x, z, y, color=line_color, label=label, linewidth=1.8)
        ax.scatter(x[0], z[0], y[0], color=line_color, marker="^", s=40, edgecolor="black")  # start

        if dot_color is not None:
            pressed = d[f"{prefix}_trigger"].values > trigger_threshold
            starts = np.flatnonzero(pressed & ~np.r_[False, pressed[:-1]])   # moments the trigger goes down
            if len(starts):
                ax.scatter(x[starts], z[starts], y[starts], color=dot_color, s=90,
                           edgecolor="black", zorder=5, label=f"{label.split()[0]} trigger press")

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y (height)")
    ax.set_title(f"{name} | user {user} | throw {throw}   (triangle = start)")
    ax.legend(loc="upper left")
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    else:
        plt.show()

#plot_throw(datasets, "Vive1", user=5, throw=4)

