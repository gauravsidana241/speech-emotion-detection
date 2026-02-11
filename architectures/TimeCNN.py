import torch
import torch.nn as nn


class TimeCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(TimeCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=49, out_channels=64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.AvgPool1d(kernel_size=2),
            # nn.Dropout1d(0.2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            # nn.Dropout1d(0.2),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            # nn.Dropout1d(0.3),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x