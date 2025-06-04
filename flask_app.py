from flask import Flask, render_template, request, send_from_directory
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os

app = Flask(__name__)

AUDIO_PATH = "1.mp3"
SPECTROGRAM_PATH = "static/spectrogram.png"

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

# Generate a default spectrogram once at startup
def generate_default_spectrogram():
    if not os.path.exists(SPECTROGRAM_PATH):
        print("Generating default spectrogram...")
        y, sr = librosa.load(AUDIO_PATH, sr=None)
        D = np.abs(librosa.stft(y, n_fft=1024, hop_length=256, window='hamming'))
        DB = librosa.amplitude_to_db(D, ref=np.max)

        plt.figure(figsize=(12, 4))
        librosa.display.specshow(DB, sr=sr, hop_length=256, x_axis='time', y_axis='linear', fmin=0, fmax=5000)
        plt.colorbar(format="%+2.0f dB")
        plt.title("Default Spectrogram (Limited to 0–5000 Hz)")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency (Hz)")
        plt.tight_layout()
        plt.savefig(SPECTROGRAM_PATH)
        plt.close()

generate_default_spectrogram()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        n_fft = int(request.form['n_fft'])
        hop_length = int(request.form['hop_length'])

        y, sr = librosa.load(AUDIO_PATH, sr=None)
        D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hamming'))
        DB = librosa.amplitude_to_db(D, ref=np.max)

        plt.figure(figsize=(12, 4))
        librosa.display.specshow(DB, sr=sr, hop_length=256, x_axis='time', y_axis='hz', fmin=0, fmax=5000)
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Spectrogram (n_fft={n_fft}, hop_length={hop_length}, 0–5000 Hz)")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency (Hz)")
        plt.tight_layout()
        plt.savefig(SPECTROGRAM_PATH)
        plt.close()

        return "ok"

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
