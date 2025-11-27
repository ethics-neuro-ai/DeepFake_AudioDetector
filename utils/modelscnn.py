import torch
from torch import nn
import torchaudio.transforms as T
class CNNTransformerClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, 3, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )
        self.flatten = nn.Flatten(2)
        
       
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = x.permute(2, 0, 1)  
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.fc(x)
