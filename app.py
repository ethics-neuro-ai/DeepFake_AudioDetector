from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import torch
import torchaudio
import joblib
import os,sys,io,tempfile
from torchvision import transforms
from speechbrain.inference import EncoderClassifier
from utils.modelcnn import CNNTransformerClassifier

from utils.cnnt_predict import predict_audio as predict3
from utils.cnnt_moodel import CNNTransformer
from speechbrain.pretrained import SpeakerRecognition
from flask import send_file
import matplotlib
matplotlib.use('Agg')  # <- non-GUI backend
import matplotlib.pyplot as plt

import io
from flask import render_template
import whisper
from pathlib import Path
import sklearn
import sklearn.ensemble._forest 


app = Flask(__name__)
CORS(app)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000



if getattr(sys, 'frozen', False):
    # Running as a PyInstaller app
    # sys.executable -> /.../MyDesktopApp.app/Contents/MacOS/MyDesktopApp
    BASE_DIR = Path(sys.executable).parent.parent / "Resources"
else:
    # Running normally in Python
    BASE_DIR = Path(__file__).resolve().parent


# Use this for all model / folder paths
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
PRETRAINED_DIR = BASE_DIR / "pretrained_models"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = Flask(__name__,
            template_folder=str(TEMPLATE_DIR),
            static_folder=str(STATIC_DIR))
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)



label_names = ['Fake', 'Real']
whisper_model = whisper.load_model("base")  # or "small", "medium", "large"


encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")


#model_dir = "models"
model_dir = BASE_DIR / "models"
ml_models = {
    "logisticregression": joblib.load(MODEL_DIR / "logistic_regression_model.pkl"),
    "mlp": joblib.load(MODEL_DIR / "ecapa_mlp_model.pkl"),
    "randomforest": joblib.load(MODEL_DIR / "random_forest_model.pkl"),
}




deep_models = {
   
    "cnnt_transformer": CNNTransformer().to(DEVICE)
}



model_path = MODEL_DIR / "new_best_model_.pth"

if not model_path.exists():
    print(f"⚠️ Model not found at {model_path}")
else:
    print(f"✅ Loading model from {model_path}")

deep_models["cnnt_transformer"].load_state_dict(
    torch.load(str(model_path), map_location=DEVICE)
)



for model in deep_models.values():
    model.eval()


mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=128)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
resize_transform = transforms.Resize((128, 128))


def extract_embedding(path):
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    emb = encoder.encode_batch(wav)
    return emb.squeeze().numpy()


def preprocess_audio(audio_path):
    waveform, sr = torchaudio.load(audio_path)

    
    if waveform.shape[0] > 1:
        waveform = waveform[0:1, :]

    
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    max_len = SAMPLE_RATE * 4 

    if waveform.shape[1] > max_len:
        waveform = waveform[:, :max_len]
    else:
        pad_amount = max_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

    
    mel_spec = mel_spectrogram_transform(waveform)
    mel_spec = amplitude_to_db(mel_spec)

    
    mel_spec = resize_transform(mel_spec)

    
    mel_spec = mel_spec.unsqueeze(0).to(DEVICE, dtype=torch.float)  

    return mel_spec

speaker_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(PRETRAINED_DIR / "spkrec-ecapa-voxceleb")
)


def preprocess_and_save(file_storage, temp_name):
    waveform, sr = torchaudio.load(file_storage)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    torchaudio.save(temp_name, waveform, 16000)
    
def whisper_transcribe(file):
    model = whisper.load_model("base")
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file.save(tmp.name)
        result = model.transcribe(tmp.name)
    
    return result.get("text", "")

@app.route("/predict", methods=["POST"])
def predict_route():
    model_name = request.args.get("model", "").lower()

    file = request.files.get("file")
    if not file or not file.filename.endswith(".wav"):
        return jsonify({"error": "Only WAV files supported"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        file.save(temp_file.name)
        try:
            if model_name in ml_models:
                # ML models use embedding extraction
                embedding = extract_embedding(temp_file.name).reshape(1, -1)
                clf = ml_models[model_name]
                prob = clf.predict_proba(embedding)[0]
                pred = clf.predict(embedding)[0]
                confidence = float(prob[pred])

            elif model_name in deep_models:
                # Deep models use the cnn_predict.predict function
                
                if model_name=="cnnt_transformer":
                    pred,confidence=predict3(temp_file.name)
                    

            else:
                return jsonify({"error": "Invalid model '{}'".format(model_name)}), 400


            return jsonify({
                "model": model_name,
                "prediction": label_names[pred],
                "confidence": confidence
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            os.unlink(temp_file.name)

@app.route('/compare', methods=['POST'])
def compare_speakers():
    try:
        if 'file1' not in request.files or 'file2' not in request.files:
            return jsonify({'error': 'Both audio files are required'}), 400

        file1 = request.files['file1']
        file2 = request.files['file2']

        if not (file1.filename.endswith('.wav') and file2.filename.endswith('.wav')):
            return jsonify({'error': 'Only .wav files are supported'}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f2:

            preprocess_and_save(file1, f1.name)
            preprocess_and_save(file2, f2.name)

            score, _ = speaker_model.verify_files(f1.name, f2.name)
            result = "Same" if score >= 0.75 else "Different"

        return jsonify({
            "result": result,
            "similarity": round(score.item(), 3)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        try:
            os.remove(f1.name)
            os.remove(f2.name)
        except Exception:
            pass
        
        
@app.route("/spectrogram", methods=["POST"])
def generate_spectrogram():
    file = request.files.get("file")
    if not file or not file.filename.endswith(".wav"):
        return jsonify({"error": "Only WAV files supported"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        file.save(temp_file.name)

        waveform, sr = torchaudio.load(temp_file.name)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        # Generate mel spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=128
        )(waveform)
        db_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        # Plot and return as image
        plt.figure(figsize=(10, 4))
        plt.imshow(db_spec.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.title('Mel Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        return send_file(buf, mimetype='image/png')
    



@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    file = request.files['file']
    # Transcribe using Whisper
    transcript = whisper_transcribe(file)  # Replace with your function
    return jsonify({'transcript': transcript})

    
@app.route("/")
def home():
    return render_template("index.html")

        
if __name__ == "__main__":
    app.run(debug=True)
