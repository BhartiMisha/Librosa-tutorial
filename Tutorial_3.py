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

    # 🧠 Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # You can also try 20 or 40

    print(f"MFCC shape: {mfccs.shape}")  # (20, Time frames)

    # 🎨 Plot the MFCCs
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title("MFCC (Mel-Frequency Cepstral Coefficients)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
