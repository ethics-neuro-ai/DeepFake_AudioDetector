import torch
import torchaudio
from utils.cnnt_moodel import CNNTransformer
import torch.nn.functional as F
from pathlib import Path
import sys

# === Configuration ===
SAMPLE_RATE = 16000
N_MELS = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIXED_LENGTH = 400


# PyInstaller-aware base directory
if getattr(sys, 'frozen', False):
    # Running inside PyInstaller bundle
    BASE_DIR = Path(sys.executable).parent.parent / "Resources"
else:
    # Running in normal Python
    BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "new_best_model_.pth"
# === Load Model ===
model = CNNTransformer().to(DEVICE)
model.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE))
model.eval()
# === Transforms ===
mel_spec_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=N_MELS
)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

def preprocess_audio(filepath):
    waveform, sr = torchaudio.load(filepath)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    mel_spec = mel_spec_transform(waveform)
    mel_spec_db = amplitude_to_db(mel_spec)
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-9)

    time_dim = mel_spec_db.shape[-1]
    if time_dim < FIXED_LENGTH:
        mel_spec_db = torch.nn.functional.pad(mel_spec_db, (0, FIXED_LENGTH - time_dim))
    else:
        mel_spec_db = mel_spec_db[:, :, :FIXED_LENGTH]

    return mel_spec_db.unsqueeze(0)


def predict_audio(filepath):
    input_tensor = preprocess_audio(filepath)
    with torch.no_grad():
        model.eval() 
        output = model(input_tensor)  # output is a single sigmoid value
        score = output.item()
        pred = 0 if score >= 0.5 else 1
        prob = [1 - score, score]  # Format as [Fake prob, Real prob]
        confidence = torch.sigmoid(output).item() 
        if score >= 0.5:
            pred = 0  # FAKE
            score_final = score          # confidence for FAKE
        else:
            pred = 1  # REAL
            score_final = 1 - score      # confidence for REAL

        print(f"raw output={output.item():.4f}, pred={pred}")
        return pred, score_final
