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

    # Calculate MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Calculate Pitch (using librosa.pyin, fallback to zero if fails)
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    except Exception:
        f0 = np.zeros(y.shape[0] // 512 + 1)  # fallback

    # Calculate Energy (Root Mean Square)
    energy = librosa.feature.rms(y=y)[0]

    # Calculate Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # Calculate Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    # Plot Spectrogram
    plt.figure(figsize=(14, 10))

    # 1. Spectrogram
    plt.subplot(5, 1, 1)
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (dB)')

    # 2. MFCCs
    plt.subplot(5, 1, 2)
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('MFCC')

    # 3. Pitch (F0)
    plt.subplot(5, 1, 3)
    times = librosa.times_like(f0, sr=sr)
    plt.plot(times, f0, label='Pitch (F0)', color='g')
    plt.ylabel('Hz')
    plt.title('Pitch (F0)')
    plt.xlim([0, times[-1]])
    plt.legend(loc='upper right')

    # 4. Energy
    plt.subplot(5, 1, 4)
    times_rms = librosa.times_like(energy, sr=sr)
    plt.plot(times_rms, energy, label='Energy', color='r')
    plt.ylabel('RMS')
    plt.title('Energy')
    plt.xlim([0, times_rms[-1]])
    plt.legend(loc='upper right')

    # 5. ZCR and Bandwidth
    plt.subplot(5, 1, 5)
    times_zcr = librosa.times_like(zcr, sr=sr)
    times_bw = librosa.times_like(bandwidth, sr=sr)
    plt.plot(times_zcr, zcr, label='Zero Crossing Rate', color='b')
    plt.plot(times_bw, bandwidth / np.max(bandwidth), label='Bandwidth (norm)', color='m', alpha=0.7)
    plt.ylabel('Value')
    plt.xlabel('Time (s)')
    plt.title('ZCR & Bandwidth')
    plt.xlim([0, max(times_zcr[-1], times_bw[-1])])
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
