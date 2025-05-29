
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool

def audio_to_spectrogram(args):
    audio_path, output_path = args
    sr, n_fft, hop_length = 22050, 2048, 512
    
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        spectrogram = np.abs(stft)
        spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
        
        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram_db, aspect='auto', origin='lower', cmap='viridis')
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        
        return f"Converted: {audio_path.name}"
    except Exception as e:
        return f"Error processing {audio_path.name}: {e}"

def process_gtzan_dataset(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    file_pairs = []
    
    for genre_folder in input_path.iterdir():
        if genre_folder.is_dir():
            genre_output = output_path / genre_folder.name
            genre_output.mkdir(exist_ok=True)
            
            for audio_file in genre_folder.glob("*.wav"):
                output_file = genre_output / f"{audio_file.stem}.png"
                file_pairs.append((audio_file, output_file))
    
    with Pool(processes=4) as pool:
        results = pool.map(audio_to_spectrogram, file_pairs)
        for result in results:
            print(result)

if __name__ == "__main__":
    input_directory = "./genres_original"
    output_directory = "./genres_spectrograms"
    
    process_gtzan_dataset(input_directory, output_directory)