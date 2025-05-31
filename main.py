import os
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import pickle

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU found: {len(gpus)}")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPU found")

class MusicGenreCNN:
    def __init__(self, input_shape=(128, 431, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        
    def build_model(self):
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16
        )

def load_data(data_dir):
    import cv2
    X, y = [], []
    for genre in os.listdir(data_dir):
        genre_path = os.path.join(data_dir, genre)
        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                if file.endswith('.png'):
                    spec = cv2.imread(os.path.join(genre_path, file), cv2.IMREAD_GRAYSCALE)
                    X.append(spec)
                    y.append(genre)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    
    X, y = load_data("genres_spectrograms")
    
    X = X.astype('float32') / 255.0
    if len(X.shape) == 3:
        X = X[..., np.newaxis]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_cat = keras.utils.to_categorical(y_encoded)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_cat, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    cnn = MusicGenreCNN(input_shape=X.shape[1:], num_classes=len(le.classes_))
    cnn.compile_model()
    
    print("Training...")
    history = cnn.train(X_train, y_train, X_val, y_val, epochs=50)
    
    test_loss, test_acc = cnn.model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.3f}")
    
    cnn.model.save("music_genre_model.h5")
    
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)