from sklearn.metrics import accuracy_score, roc_curve
import numpy as np


import os
import torch
import torchaudio
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'new_best_model_.pth'
TEST_FOLDER = '/Users/stellafazioli/Documents/test_wamisto'  # your test folder

N_MELS = 64
SAMPLE_RATE = 16000
FIXED_LENGTH = 400  # same as training padding/truncation length

# Your model class (copy-paste from your training script)
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x

class CNNTransformer(torch.nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=1):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        )
        self.linear_proj = torch.nn.Linear(128 * (n_mels // 8), 256)
        self.pos_encoder = PositionalEncoding(256)
        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=256, nhead=8, dropout=0.3)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, n_classes),
            torch.nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.cnn(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, t, c * f)
        x = self.linear_proj(x)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=0)
        x = self.classifier(x)
        return x.squeeze()