import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

model = keras.models.load_model("music_genre_model.h5")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

def predict_genre(spectrogram_path):
    spec = cv2.imread(spectrogram_path, cv2.IMREAD_GRAYSCALE)
    spec = spec.astype('float32') / 255.0
    spec = spec[..., np.newaxis]
    spec = np.expand_dims(spec, axis=0)
    
    prediction = model.predict(spec)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    
    genre = le.inverse_transform([predicted_class])[0]
    
    return genre, confidence

print(f"Available genres: {list(le.classes_)}")

genre, confidence = predict_genre("path/to/new/spectrogram.png")
print(f"Predicted genre: {genre} (confidence: {confidence:.3f})")