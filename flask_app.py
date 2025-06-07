from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
from werkzeug.utils import secure_filename
from scipy.signal import butter, sosfilt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SPECTROGRAM_PATH'] = 'static/spectrogram.png'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

def bandpass_filter(y, sr, low=100, high=3500):
    sos = butter(6, [low, high], btype='bandpass', fs=sr, output='sos')
    return sosfilt(sos, y)

def generate_spectrogram(filepath, n_fft, hop_length, sr):
    y, _ = librosa.load(filepath, sr=sr)

    # Bandpass filter: keep only vehicle-related frequencies
    y = bandpass_filter(y, sr)

    # Trim silence/background
    intervals = librosa.effects.split(y, top_db=30)
    y = np.concatenate([y[start:end] for start, end in intervals]) if len(intervals) > 0 else y

    # Compute STFT
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hamming'))
    DB = librosa.amplitude_to_db(D, ref=np.max)

    # Plot
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.ylim(300, 3500)    # Focus on useful range
    plt.clim(-60, 0)       # Enhance contrast
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram (n_fft={n_fft}, hop_length={hop_length}, sr={sr})")
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
        window_ms = int(request.form['window_size'])
        sr = int(request.form['sampling_rate'])
        hop_length = int(request.form['hop_length'])

        # Calculate n_fft from window size and sampling rate
        n_fft = int((window_ms / 1000) * sr)

        file = request.files['audio']
        if not file:
            return "No file uploaded"

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        generate_spectrogram(filepath, n_fft, hop_length, sr)
        return "ok"

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
