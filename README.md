# 🎤 Deepfake Audio Detector

**Status:** Experimental / Research 

This project implements a **deepfake audio detection system** using both machine learning and deep learning techniques.  
It was developed as part of academic research and can classify audio clips as **Real** or **Fake**, compare speakers, generate spectrograms, and transcribe audio.

---

## 📝 Overview

Deepfake audio is increasingly common and can be used maliciously. This project demonstrates how to detect such manipulated audio using a combination of:

- **ML models:** Logistic Regression, MLP, Random Forest  
- **Deep Learning models:** CNN + Transformer  
- **Pretrained speaker recognition:** SpeechBrain ECAPA-TDNN  
- **Automatic transcription:** OpenAI Whisper

The system is wrapped in a **Flask API** with endpoints for predictions, speaker comparison, spectrogram visualization, and transcription.

---

## ✨ Features

- Detect if an audio file is real or deepfake
- Compare two audio files to see if they are from the same speaker
- Generate Mel spectrogram images for visualization
- Transcribe audio to text using Whisper
- Web API with file upload support and temporary storage management

---

## 🛠️ Tech Stack

- Python 3.x
- Flask + Flask-CORS
- PyTorch + Torchaudio
- SpeechBrain (Speaker Recognition)
- Whisper (speech-to-text)
- Joblib (ML model serialization)
- Matplotlib (spectrogram visualization)

---

## ⚙️ Usage

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
Install dependencies:

pip install -r requirements.txt


Start the Flask server:

python app.py


API Endpoints:

Endpoint	Method	Description
/predict	POST	Predicts if a WAV audio file is real or fake using the selected model (model parameter: logisticregression, mlp, randomforest, cnnt_transformer)
/compare	POST	Compares two WAV audio files to determine if they belong to the same speaker
/spectrogram	POST	Generates a Mel spectrogram image for the uploaded audio file
/transcribe	POST	Transcribes WAV audio to text using Whisper

All audio files must be in .wav format.

📦 Directory Structure
project/
├── app.py
├── models/               # Trained deep learning models
├── pretrained_models/    # Pretrained speaker recognition models
├── uploads/              # Temporary audio uploads
├── static/               # Frontend assets
├── templates/            # Flask HTML templates
└── utils/                # Helper scripts for audio processing and ML

📄 requirements.txt
torch>=1.13.1
torchaudio>=0.14.1
flask>=2.3.2
flask-cors>=3.0.10
matplotlib>=3.6.3
joblib>=1.3.2
speechbrain>=0.5.13
whisper @ git+https://github.com/openai/whisper.git
torchvision>=0.14.1
numpy>=1.23.5


Save this as requirements.txt in your repo root.

📊 Notes

The deep learning model CNNTransformer identifies subtle artifacts to detect fake audio.

Speaker comparison uses ECAPA-TDNN embeddings to determine similarity.

Mel spectrogram visualization helps to visually inspect audio characteristics.

Whisper transcription provides an optional textual representation of the audio.

⚠️ Disclaimer

This project is intended for research and educational purposes only.
