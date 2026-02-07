import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_filters=32, kernel_size=3, num_classes=6):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(num_filters, num_filters*2, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(num_filters*2, num_filters*2, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters*2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x