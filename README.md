# Music Genre Classification

A CNN-based music genre classifier using spectrograms of audio files.

## Pre-usage
   ```bash
   pip install -r requirments.txt
   ```
   Install requirments

## Usage Order

1. **Download Dataset**
   ```bash
   python download_data.py
   ```
   Downloads the GTZAN dataset from Kaggle.

2. **Convert Audio to Spectrograms**
   ```bash
   python music_to_spectrograms.py
   ```
   Converts audio files to spectrogram images for training.

3. **Train Model**
   ```bash
   python main.py
   ```
   Trains the CNN model and saves `music_genre_model.h5` and `label_encoder.pkl`.

4. **Make Predictions**
   ```bash
   python prediction_script.py <audio_file.wav>
   ```
   Predicts genre for a new audio file. 
   Prediction script gets the .wav file, creates and uses a spectrogram of it for prediction.