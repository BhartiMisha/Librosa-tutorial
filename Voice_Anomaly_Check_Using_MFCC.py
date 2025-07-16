import librosa
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path

def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def compare_mfcc(mfcc1, mfcc2):
    # Compute the mean of each MFCC across time
    mfcc1_mean = mfcc1.mean(axis=1)
    mfcc2_mean = mfcc2.mean(axis=1)
    
    # Compute the Euclidean distance
    distance = euclidean(mfcc1_mean, mfcc2_mean)
    return distance

def main():
    # File paths
    normal_file = Path('data') / 'test.mp3'
    test_file = Path('data') / 'suspicious_audio.mp3'

    # Check if files exist
    if not normal_file.exists() or not test_file.exists():
        print("Voice files not found. Please check paths.")
        return

    # Extract MFCCs
    print("Extracting MFCCs...")
    mfcc_normal = extract_mfcc(str(normal_file))
    mfcc_test = extract_mfcc(str(test_file))

    # Compare MFCCs
    print("Comparing voices...")
    distance = compare_mfcc(mfcc_normal, mfcc_test)
    print(f"Euclidean Distance between voices: {distance:.2f}")

    # Set a threshold — tune based on trial (usually 20–50 for voice)
    THRESHOLD = 35

    if distance < THRESHOLD:
        print("Voice is NORMAL")
    else:
        print("Voice is ANOMALOUS")

    # Visualize both MFCCs
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    librosa.display.specshow(mfcc_normal, ax=axs[0], x_axis='time')
    axs[0].set_title("Normal Voice MFCC")
    axs[0].set_ylabel("MFCC Coefficients")

    librosa.display.specshow(mfcc_test, ax=axs[1], x_axis='time')
    axs[1].set_title("Test Voice MFCC")

    for ax in axs:
        ax.set_xlabel("Time")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
