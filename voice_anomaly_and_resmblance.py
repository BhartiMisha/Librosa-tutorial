import librosa
import numpy as np
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav

def analyze_voice(reference_path, test_path):
    """
    Analyze the test voice against a reference voice and return a dict with similarity, behavior match, risk, etc.
    """
    # --- Voice similarity using resemblyzer ---
    encoder = VoiceEncoder()
    wav_ref = preprocess_wav(Path(reference_path))
    wav_test = preprocess_wav(Path(test_path))
    embed_ref = encoder.embed_utterance(wav_ref)
    embed_test = encoder.embed_utterance(wav_test)
    cosine_sim = np.dot(embed_ref, embed_test) / (np.linalg.norm(embed_ref) * np.linalg.norm(embed_test))
    voice_similarity = float(np.clip(cosine_sim * 100, 0, 100))  # as percentage

    # --- Behavioral features: MFCC, pitch, energy, zcr, bandwidth ---
    def extract_features(audio_path):
        y, sr = librosa.load(str(audio_path))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = mfccs.mean(axis=1)
        try:
            f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            pitch_mean = np.nanmean(f0)
        except Exception:
            pitch_mean = 0.0
        energy = librosa.feature.rms(y=y)[0]
        energy_mean = np.mean(energy)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        bandwidth_mean = np.mean(bandwidth)
        return np.concatenate([mfccs_mean, [pitch_mean, energy_mean, zcr_mean, bandwidth_mean]])

    feat_ref = extract_features(reference_path)
    feat_test = extract_features(test_path)
    # Euclidean distance for behavior
    behavior_dist = np.linalg.norm(feat_ref - feat_test)
    # Convert to a "match" score (higher is better, 0-100)
    # Assume typical voice-to-voice distance is 20-50, so we map 0->100, 50->0
    behavior_match = float(np.clip(100 - (behavior_dist * 2), 0, 100))

    # --- Risk score: combine both, higher means more risk ---
    # If either similarity is low or behavior match is low, risk is high
    risk_score = 1.0 - (0.4 * (voice_similarity/100) + 0.6 * (behavior_match/100))
    risk_score = float(np.clip(risk_score, 0, 1))

    # --- Risk level and verdict ---
    if risk_score > 0.75:
        risk_level = "HIGH"
        verdict = "Possible impersonation"
    elif risk_score > 0.4:
        risk_level = "MEDIUM"
        verdict = "Some anomaly detected"
    else:
        risk_level = "LOW"
        verdict = "Voice matches reference"

    return {
        "voice_similarity": round(voice_similarity, 1),
        "behavior_match": round(behavior_match, 1),
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "verdict": verdict
    }

# def extract_features(audio_path):
#     y, sr = librosa.load(str(audio_path))
#     # MFCCs
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     mfccs_mean = mfccs.mean(axis=1)
#     # Pitch (F0)
#     try:
#         f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
#         pitch_mean = np.nanmean(f0)
#     except Exception:
#         pitch_mean = 0.0
#     # Energy (RMS)
#     energy = librosa.feature.rms(y=y)[0]
#     energy_mean = np.mean(energy)
#     # Zero Crossing Rate
#     zcr = librosa.feature.zero_crossing_rate(y)[0]
#     zcr_mean = np.mean(zcr)
#     # Spectral Bandwidth
#     bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
#     bandwidth_mean = np.mean(bandwidth)
#     # Combine all features
#     features = np.concatenate([mfccs_mean, [pitch_mean, energy_mean, zcr_mean, bandwidth_mean]])
#     return features

# def calculate_voice_similarity(file1, file2):
#     encoder = VoiceEncoder()
#     wav1 = preprocess_wav(Path(file1))
#     wav2 = preprocess_wav(Path(file2))
#     embed1 = encoder.embed_utterance(wav1)
#     embed2 = encoder.embed_utterance(wav2)
#     similarity = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
#     return similarity

# def plot_features(audio_path):
#     y, sr = librosa.load(str(audio_path))
#     plt.figure(figsize=(14, 10))
#     # 1. Spectrogram
#     plt.subplot(5, 1, 1)
#     S = librosa.stft(y)
#     S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
#     librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Spectrogram (dB)')
#     # 2. MFCCs
#     plt.subplot(5, 1, 2)
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     librosa.display.specshow(mfccs, x_axis='time', sr=sr)
#     plt.colorbar()
#     plt.title('MFCC')
#     # 3. Pitch (F0)
#     plt.subplot(5, 1, 3)
#     try:
#         f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
#     except Exception:
#         f0 = np.zeros(y.shape[0] // 512 + 1)
#     times = librosa.times_like(f0, sr=sr)
#     plt.plot(times, f0, label='Pitch (F0)', color='g')
#     plt.ylabel('Hz')
#     plt.title('Pitch (F0)')
#     plt.xlim([0, times[-1]])
#     plt.legend(loc='upper right')
#     # 4. Energy
#     plt.subplot(5, 1, 4)
#     energy = librosa.feature.rms(y=y)[0]
#     times_rms = librosa.times_like(energy, sr=sr)
#     plt.plot(times_rms, energy, label='Energy', color='r')
#     plt.ylabel('RMS')
#     plt.title('Energy')
#     plt.xlim([0, times_rms[-1]])
#     plt.legend(loc='upper right')
#     # 5. ZCR and Bandwidth
#     plt.subplot(5, 1, 5)
#     zcr = librosa.feature.zero_crossing_rate(y)[0]
#     bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
#     times_zcr = librosa.times_like(zcr, sr=sr)
#     times_bw = librosa.times_like(bandwidth, sr=sr)
#     plt.plot(times_zcr, zcr, label='Zero Crossing Rate', color='b')
#     plt.plot(times_bw, bandwidth / np.max(bandwidth), label='Bandwidth (norm)', color='m', alpha=0.7)
#     plt.ylabel('Value')
#     plt.xlabel('Time (s)')
#     plt.title('ZCR & Bandwidth')
#     plt.xlim([0, max(times_zcr[-1], times_bw[-1])])
#     plt.legend(loc='upper right')
#     plt.tight_layout()
#     plt.show()

# def train_model(voice_files, labels, model_path="voice_knn_model.joblib"):
#     X = []
#     for file in voice_files:
#         features = extract_features(file)
#         X.append(features)
#     X = np.array(X)
#     y = np.array(labels)
#     knn = KNeighborsClassifier(n_neighbors=3)
#     knn.fit(X, y)
#     joblib.dump(knn, model_path)
#     print(f"Model trained and saved to {model_path}")

# def predict_voice(file, model_path="voice_knn_model.joblib"):
#     features = extract_features(file).reshape(1, -1)
#     knn = joblib.load(model_path)
#     pred = knn.predict(features)
#     print(f"Predicted label: {pred[0]}")
#     return pred[0]

if __name__ == "__main__":
    import json
    # Use two example files
    file1 = "data/test.mp3"
    file2 = "data/heavy_voice.mp3"
    result = analyze_voice(file1, file2)
    print(json.dumps(result, indent=2))
