#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent / "data"

DEVICES = ["Quest", "Vive", "Cosmos"]
DAY_TRAIN, DAY_TEST = 1, 2  # file suffix: Quest1 = day 1, Quest2 = day 2

PARTS = ["px", "py", "pz", "ax", "ay", "az", "trigger"]  # position, Euler angles, trigger
COLUMNS = [f"{dev}_{p}" for dev in ["right", "head", "left"] for p in PARTS]  # same order as in the files

# Same cleaning as before (inspectdata.clean_data): drop head_trigger and left_trigger
DROP_COLUMNS = ["head_trigger", "left_trigger"]
FEATURE_NAMES = [c for c in COLUMNS if c not in DROP_COLUMNS]
KEEP_IDX = [COLUMNS.index(c) for c in FEATURE_NAMES]


@dataclass
class DeviceData:
    """Cross-day data for one device. X shape: (samples, time steps, features)."""
    X_train: np.ndarray   # day 1, normalized
    y_train: np.ndarray
    X_test: np.ndarray    # day 2, normalized with train statistics
    y_test: np.ndarray
    feature_names: list
    scaler: StandardScaler


def _load_file(path):
    """Load one .npy file (users, throws, steps, 21) -> cleaned X (N, T, F), labels y (N,)."""
    arr = np.load(path)
    users, throws, steps, _ = arr.shape

    arr = arr[..., KEEP_IDX]                          # cleaning: drop head/left trigger
    X = arr.reshape(users * throws, steps, len(KEEP_IDX)).astype(np.float32)
    y = np.repeat(np.arange(users), throws).astype(np.int64)  # user id = 0..users-1
    return X, y


def load_device(device):
    """Load, clean and normalize one device: day 1 -> train, day 2 -> test."""
    X_train, y_train = _load_file(DATA_DIR / f"{device}{DAY_TRAIN}.npy")
    X_test, y_test = _load_file(DATA_DIR / f"{device}{DAY_TEST}.npy")

    # Per-feature standardization; statistics come from the training day only
    n_features = X_train.shape[2]
    scaler = StandardScaler().fit(X_train.reshape(-1, n_features))
    X_train = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    return DeviceData(
        X_train=X_train.astype(np.float32),
        y_train=y_train,
        X_test=X_test.astype(np.float32),
        y_test=y_test,
        feature_names=FEATURE_NAMES,
        scaler=scaler,
    )


def load_all():
    """Returns {"Quest": DeviceData, "Vive": DeviceData, "Cosmos": DeviceData}."""
    return {device: load_device(device) for device in DEVICES}


if __name__ == "__main__":
    for device, d in load_all().items():
        print(f"{device}: train {d.X_train.shape}, test {d.X_test.shape}, "
              f"users {len(np.unique(d.y_train))}")
    print("Features:", FEATURE_NAMES)
