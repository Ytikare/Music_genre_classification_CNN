import tensorflow as tf
from tensorflow import keras
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pickle
import cv2
import os
from pathlib import Path
import sys


def predict_genre(audio_path):
    # Load model and label encoder
    model = keras.models.load_model("music_genre_model.h5")
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    
    # Create output directory
    output_dir = Path("test_files_spectrograms/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get song name without extension
    song_name = Path(audio_path).stem
    spec_path = output_dir / f"{song_name}.png"
    
    # Load only first 30 seconds and create spectrogram
    y, sr = librosa.load(audio_path, sr=22050, duration=30.0)
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    
    # Save spectrogram with song name
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram_db, aspect='auto', origin='lower', cmap='viridis')
    plt.axis('off')
    plt.savefig(spec_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    
    # Load image and predict
    spec = cv2.imread(str(spec_path), cv2.IMREAD_GRAYSCALE)
    spec = spec.astype('float32') / 255.0
    spec = spec[..., np.newaxis]
    spec = np.expand_dims(spec, axis=0)
    
    prediction = model.predict(spec)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    genre = le.inverse_transform([predicted_class])[0]
    
    return genre, confidence

if __name__ == "__main__":
    music_file = sys.argv[1]
    genre, confidence = predict_genre(music_file)
    print(f"Genre: {genre}, Confidence: {confidence:.3f}")