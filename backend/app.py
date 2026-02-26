from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from pydub import AudioSegment
import os

app = Flask(__name__)
CORS(app)

EMOTIONS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']
SAMPLE_RATE = 16000
DURATION = 5
TARGET_LENGTH = SAMPLE_RATE * DURATION

# Model definition (must match training)
class Wav2Vec2EmotionClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec2.feature_extractor._freeze_parameters()
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        return self.classifier(pooled)

# Load trained weights
model = Wav2Vec2EmotionClassifier()
model.load_state_dict(torch.load('model/emotion_model.pth', map_location='cpu'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    file = request.files['audio']
    temp_path = 'temp_audio.webm'
    wav_path = 'temp_audio.wav'
    file.save(temp_path)

    try:
        # convert webm to wav using ffmpeg
        audio = AudioSegment.from_file(temp_path)
        audio.export(wav_path, format='wav')

        # Same preprocessing as training
        waveform, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

        if len(waveform) < TARGET_LENGTH:
            padding = TARGET_LENGTH - len(waveform)
            offset = padding // 2
            waveform = np.pad(waveform, (offset, padding - offset), 'constant')
        else:
            waveform = waveform[:TARGET_LENGTH]

        # Inference
        input_tensor = torch.FloatTensor(waveform).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = torch.argmax(output, dim=1).item()

        return jsonify({'emotion': EMOTIONS[pred_idx]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=False, port=5000)