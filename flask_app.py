from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SPECTROGRAM_PATH'] = 'static/spectrogram.png'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

def generate_spectrogram(filepath, n_fft, hop_length):
    y, sr = librosa.load(filepath, sr=None)
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hamming'))
    DB = librosa.amplitude_to_db(D, ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', fmin=0, fmax=5000)
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram (n_fft={n_fft}, hop_length={hop_length})")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(app.config['SPECTROGRAM_PATH'])
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        n_fft = int(request.form['n_fft'])
        hop_length = int(request.form['hop_length'])

        file = request.files['audio']
        if not file:
            return "No file uploaded"

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        generate_spectrogram(filepath, n_fft, hop_length)
        return "ok"

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
