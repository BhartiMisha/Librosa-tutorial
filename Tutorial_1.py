from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt


def main():
    audio_path = Path('data') / 'test.mp3'
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return

    y, sr = librosa.load(str(audio_path))

    print("These are the numbers for your voice:", y[:10])  # just first 10
    print("Sampling rate:", sr)

    if len(y) > 10010:
        print("Middle of audio:", y[10000:10010])
    else:
        print("Audio is shorter than 10010 samples.")

    print("End of audio:", y[-10:])

    librosa.display.waveshow(y, sr=sr)
    plt.show()

if __name__ == "__main__":
    main()
