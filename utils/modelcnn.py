import torch
import torch.nn as nn

class CNNTransformerClassifier(nn.Module):
    def __init__(self):
        super(CNNTransformerClassifier, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # cnn.0
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # cnn.3
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Remove the 3rd conv layer
        )



        self.flatten = nn.Flatten(2)                      # → [B, 64, 256]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=2048, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary classification: Fake or Real
        )

    def forward(self, x):
        x = self.cnn(x)                # → [B, 64, 16, 16]
        x = self.flatten(x)            # → [B, 64, 256]
        x = x.permute(2, 0, 1)         # → [256, B, 64]
        x = x[:, :, :32]               # Reduce to match d_model=32 (select first 32 features)
        x = self.transformer(x)        # → [256, B, 32]
        x = x.mean(dim=0)              # → [B, 32]
        return self.fc(x)              # → [B, 2]
