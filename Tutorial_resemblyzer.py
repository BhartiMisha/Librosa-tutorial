from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

# Load encoder model
encoder = VoiceEncoder()

# Load and preprocess your audio files
wav1 = preprocess_wav(Path("data/test.mp3"))
wav2 = preprocess_wav(Path("data/weird_accent.mp3"))

# Generate voice embeddings
embed1 = encoder.embed_utterance(wav1)
embed2 = encoder.embed_utterance(wav2)

# Cosine similarity (1.0 = identical, lower = more different)
similarity = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))

print(f"Cosine similarity: {similarity:.3f}")
