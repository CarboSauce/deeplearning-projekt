"""AlexNet-type 1D CNN dla sekwencji (features, 135 kroków).
Oryginalny AlexNet: 5 conv + 3 FC, ReLU, MaxPool, Dropout.
Adaptacje do 1D i krótkich szeregów: mniejszy stride w conv1, BatchNorm zamiast LRN,
AdaptiveAvgPool1d(6) zamiast stałego rozmiaru wejścia, węższe warstwy FC."""
from torch import nn

NAME = "alexnet"


class AlexNet1D(nn.Module):
    def __init__(self, in_features, num_classes, fc_size=1024, dropout=0.5):
        super().__init__()

        def conv(i, o, k, s=1):
            return [nn.Conv1d(i, o, k, stride=s, padding=k // 2),
                    nn.BatchNorm1d(o), nn.ReLU(inplace=True)]

        self.features = nn.Sequential(
            *conv(in_features, 64, 11, s=2), nn.MaxPool1d(3, 2),   # 135 -> 68 -> 33
            *conv(64, 192, 5),               nn.MaxPool1d(3, 2),   # 33 -> 16
            *conv(192, 384, 3),
            *conv(384, 256, 3),
            *conv(256, 256, 3),              nn.MaxPool1d(3, 2),   # 16 -> 7
            nn.AdaptiveAvgPool1d(6),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(256 * 6, fc_size), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(fc_size, fc_size), nn.ReLU(inplace=True),
            nn.Linear(fc_size, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(num_features, num_classes):
    return AlexNet1D(num_features, num_classes)


def run(loaders, num_features, num_classes):
    from common import fit_and_evaluate
    model = build_model(num_features, num_classes)
    return fit_and_evaluate(model, loaders, NAME, weight_decay=1e-4)
