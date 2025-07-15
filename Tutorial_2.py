# write program to visualize the spectrogram of a audio file
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def main():
    audio_path = Path('data') / 'test.mp3'
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return

    y, sr = librosa.load(str(audio_path))

    # Compute the spectrogram (Short-time Fourier transform magnitude)
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(abs(S), ref=np.max)

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (dB)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

