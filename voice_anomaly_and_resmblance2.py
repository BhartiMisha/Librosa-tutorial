import librosa
import numpy as np
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import euclidean

def calculate_voice_resemblance(audio1, audio2):
    encoder = VoiceEncoder()
    wav_1 = preprocess_wav(Path(audio1))
    wav_2 = preprocess_wav(Path(audio2))
    embed_audio1 = encoder.embed_utterance(wav_1)
    embed_audio2 = encoder.embed_utterance(wav_2)

    cosine_sim = np.dot(embed_audio1, embed_audio2) / (np.linalg.norm(embed_audio1) * np.linalg.norm(embed_audio2))
    voice_similarity = float(np.clip(cosine_sim * 100, 0, 100))
    return voice_similarity

def extract_MFCC(y, sr):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

def extract_pitch(y):
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    except Exception:
        f0 = np.zeros(y.shape[0] // 512 + 1)
    return f0

def extract_energy(y):
    return librosa.feature.rms(y=y)[0]

def extract_zcr(y):
    return librosa.feature.zero_crossing_rate(y=y)[0]

def extract_bandwidth(y, sr):
    return librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

def estimate_emotion(pitch, energy, zcr, bandwidth):
    if pitch > 200 and energy > 0.04:
        if zcr > 0.1:
            return "Angry"
        else:
            return "Happy"
    elif pitch < 150 and energy < 0.02:
        return "Sad"
    elif zcr > 0.15 and bandwidth > 3000:
        return "Fearful"
    else:
        return "Neutral"

def extract_feature(audio_path):
    y, sr = librosa.load(str(audio_path))
    pitch = np.nan_to_num(extract_pitch(y))
    pitch_mean = pitch.mean()
    energy_mean = extract_energy(y).mean()
    zcr_mean = extract_zcr(y).mean()
    bandwidth_mean = extract_bandwidth(y, sr).mean()
    mfcc_mean = extract_MFCC(y, sr).mean(axis=1)

    emotion = estimate_emotion(pitch_mean, energy_mean, zcr_mean, bandwidth_mean)

    return {
        'mfcc': mfcc_mean,
        'pitch': pitch_mean,
        'energy': energy_mean,
        'zcr': zcr_mean,
        'bandwidth': bandwidth_mean,
        'emotion': emotion
    }


def calculate_behavior_match(audio1, audio2):
    feat1 = extract_feature(audio1)
    feat2 = extract_feature(audio2)

    diff_scores = {
        "mfcc_distance": euclidean(feat1['mfcc'], feat2['mfcc']),
        "pitch_diff": abs(feat1['pitch'] - feat2['pitch']),
        "energy_diff": abs(feat1['energy'] - feat2['energy']),
        "zcr_diff": abs(feat1['zcr'] - feat2['zcr']),
        "bandwidth_diff": abs(feat1['bandwidth'] - feat2['bandwidth']),
    }

    # Normalization factors (based on typical audio ranges)
    max_vals = {
        "mfcc_distance": 100.0,
        "pitch_diff": 200.0,
        "energy_diff": 0.1,
        "zcr_diff": 0.5,
        "bandwidth_diff": 5000.0,
    }

    norm_diffs = [
        diff_scores[key] / max_vals[key]
        for key in diff_scores
    ]

    behavior_anomaly = np.clip(np.mean(norm_diffs) * 100, 0, 100)
    behavior_similarity = 100 - behavior_anomaly

    emotion1 = feat1['emotion']
    emotion2 = feat2['emotion']

    return behavior_similarity, (emotion1, emotion2)

def calculate_anomaly(voice_score, behavior_score, voice_threshold=75, behavior_threshold=70):
    if voice_score > voice_threshold and behavior_score > behavior_threshold:
        return "Normal"
    elif voice_score > voice_threshold:
        return "Voice Match but Behavioral Anomaly"
    elif behavior_score > behavior_threshold:
        return "Behavior Match but Voice Anomaly"
    else:
        return "Possible Fraud or Impersonation"

# ========== Main ==========
if __name__ == "__main__":
    audio1 = "data/normal.mp3"
    audio2 = "data/random_girl_humming.mp3"

    voice_resemblance = calculate_voice_resemblance(audio1, audio2)
    behaviour_match, (emotion1, emotion2) = calculate_behavior_match(audio1, audio2)
    anomaly_result = calculate_anomaly(voice_resemblance, behaviour_match)

    print("📊 Voice & Emotion Analysis")
    print("-" * 40)
    print(f"Voice resemblance     : {voice_resemblance:.2f}%")
    print(f"Behavior similarity   : {behaviour_match:.2f}%")
    print(f"Emotion in Audio 1    : {emotion1}")
    print(f"Emotion in Audio 2    : {emotion2}")
    print(f"Anomaly Detection     : {anomaly_result}")
    if emotion1 != emotion2:
        print(f"⚠️ Emotion mismatch detected: {emotion1} → {emotion2}")





'''
Emotion Mapping Logic (Rule-Based)
------------------------------------
Emotion	    Pitch	    Energy	    ZCR	            Bandwidth
Happy	    High	    High	    Medium-High	    High
Sad	Low	    Low	Low	    Low
Angry	    High	    High	    High	        High
Neutral	    Medium	    Medium	    Medium	        Medium
Fearful	    High	    Low/Med	    High	        Med/High
'''