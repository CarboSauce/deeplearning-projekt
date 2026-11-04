import math

import torch
from torch import nn

EMBEDDING_DIM = 64   # same embedding size for every model


def branches_from_features(feature_names):
    """Group feature indices by source device: right controller, headset, left controller.
    Works for any feature subset (e.g. position-only), empty groups are skipped."""
    groups = []
    for prefix in ("right_", "head_", "left_"):
        idx = [i for i, n in enumerate(feature_names) if n.startswith(prefix)]
        if idx:
            groups.append(idx)
    return groups


# ==================================================
# 1. Multi-Input Branch CNN
# ==================================================
class Branch(nn.Module):
    """1D CNN over one input stream. (N, C, T) -> (N, 64)"""

    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class MultiBranchCNN(nn.Module):
    """One CNN branch per input stream (right / head / left), concatenated into
    a shared embedding layer, followed by a linear classifier.

    forward(x): x has shape (N, features, time steps) -> logits
    embed(x):   same input -> embedding (N, embedding_dim), used for cosine Rank-1 / RSA
    """

    def __init__(self, branch_indices, num_classes, embedding_dim=EMBEDDING_DIM, dropout=0.3):
        super().__init__()
        self.branch_indices = branch_indices
        for k, idx in enumerate(branch_indices):
            self.register_buffer(f"idx_{k}", torch.tensor(idx, dtype=torch.long))

        self.branches = nn.ModuleList(Branch(len(idx)) for idx in branch_indices)

        self.embedding = nn.Sequential(
            nn.Linear(64 * len(branch_indices), embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def embed(self, x):
        feats = [
            branch(x.index_select(1, getattr(self, f"idx_{k}")))
            for k, branch in enumerate(self.branches)
        ]
        return self.embedding(torch.cat(feats, dim=1))

    def forward(self, x):
        return self.classifier(self.embed(x))


# ==================================================
# 2. AlexNet-type CNN (1D)
# ==================================================
class AlexNetCNN(nn.Module):
    """AlexNet adapted to 1D time series: 5 conv layers (large first kernel, overlapping
    max-pooling after conv 1, 2 and 5) followed by 2 fully connected layers with dropout.
    Channel widths are halved versus the original and BatchNorm is added, because the
    dataset is tiny compared to ImageNet. All input channels go into one shared stack.

    forward(x): (N, features, time steps) -> logits
    embed(x):   (N, embedding_dim)
    """

    def __init__(self, in_features, num_classes, embedding_dim=EMBEDDING_DIM, dropout=0.5):
        super().__init__()

        def block(cin, cout, k, pool):
            layers = [nn.Conv1d(cin, cout, kernel_size=k, padding=k // 2),
                      nn.BatchNorm1d(cout), nn.ReLU()]
            if pool:
                layers.append(nn.MaxPool1d(kernel_size=3, stride=2))
            return layers

        self.features = nn.Sequential(
            *block(in_features, 32, 11, pool=True),
            *block(32, 96, 5, pool=True),
            *block(96, 192, 3, pool=False),
            *block(192, 128, 3, pool=False),
            *block(128, 128, 3, pool=True),
            nn.AdaptiveAvgPool1d(4),              # fixed length, independent of T
            nn.Flatten(),
        )
        self.embedding = nn.Sequential(           # the two FC layers of AlexNet
            nn.Dropout(dropout),
            nn.Linear(128 * 4, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim), nn.ReLU(),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def embed(self, x):
        return self.embedding(self.features(x))

    def forward(self, x):
        return self.classifier(self.embed(x))


# ==================================================
# 3. Transformer encoder
# ==================================================
class TransformerClassifier(nn.Module):
    """Linear projection of each time step -> sinusoidal positional encoding ->
    Transformer encoder -> mean pooling over time -> embedding -> classifier.

    forward(x): (N, features, time steps) -> logits
    embed(x):   (N, embedding_dim)
    """

    def __init__(self, in_features, num_classes, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1, embedding_dim=EMBEDDING_DIM, max_len=1000):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))            # (1, max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.embedding = nn.Sequential(nn.Linear(d_model, embedding_dim), nn.ReLU())
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embedding_dim, num_classes))

    def embed(self, x):
        x = x.permute(0, 2, 1)                                  # (N, T, F)
        x = self.input_proj(x) + self.pe[:, : x.size(1)]
        x = self.encoder(x).mean(dim=1)                         # average over time
        return self.embedding(x)

    def forward(self, x):
        return self.classifier(self.embed(x))


# ==================================================
# Factory
# ==================================================
MODEL_NAMES = ["MultiBranchCNN", "AlexNetCNN", "Transformer"]


def build_model(name, feature_names, num_classes):
    n_features = len(feature_names)
    if name == "MultiBranchCNN":
        return MultiBranchCNN(branches_from_features(feature_names), num_classes)
    if name == "AlexNetCNN":
        return AlexNetCNN(n_features, num_classes)
    if name == "Transformer":
        return TransformerClassifier(n_features, num_classes)
    raise ValueError(f"Unknown model: {name}. Choose from {MODEL_NAMES}")