"""
audio_extractor.py
==================
Server-side audio feature extraction using librosa.
Used when the client uploads an audio file (not browser-recorded).

The frontend (Web Audio API) computes simplified features for
browser-recorded audio. This module computes full librosa features
for uploaded .wav/.mp3 files processed server-side.

Install: pip install librosa soundfile
"""

import numpy as np
from pathlib import Path

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False


def extract_from_file(filepath: str, sr: int = 22050, duration: float = 4.0) -> dict:
    """
    Extract MFCC + spectral features from an audio file.
    Returns dict matching audio_features.csv schema.
    """
    if not LIBROSA_OK:
        raise RuntimeError(
            "librosa not installed. Run: pip install librosa soundfile"
        )

    y, sr = librosa.load(filepath, sr=sr, mono=True, duration=duration)

    # ── 13 MFCCs ─────────────────────────────────────────────────
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = {}
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i+1}_std"]  = float(np.std(mfccs[i]))

    # ── Spectral centroid ─────────────────────────────────────────
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid_mean"] = float(np.mean(sc))
    features["spectral_centroid_std"]  = float(np.std(sc))

    # ── Spectral bandwidth ────────────────────────────────────────
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["spectral_bandwidth_mean"] = float(np.mean(sb))
    features["spectral_bandwidth_std"]  = float(np.std(sb))

    # ── Spectral rolloff ──────────────────────────────────────────
    sr_feat = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    features["spectral_rolloff_mean"] = float(np.mean(sr_feat))
    features["spectral_rolloff_std"]  = float(np.std(sr_feat))

    # ── Zero crossing rate ────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zero_crossing_rate_mean"] = float(np.mean(zcr))
    features["zero_crossing_rate_std"]  = float(np.std(zcr))

    # ── RMS energy ────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y)[0]
    features["rms_energy_mean"] = float(np.mean(rms))
    features["rms_energy_std"]  = float(np.std(rms))

    # ── Dominant frequency ────────────────────────────────────────
    fft_mag = np.abs(np.fft.rfft(y))
    freqs   = np.fft.rfftfreq(len(y), 1.0 / sr)
    features["dominant_freq_hz"] = float(freqs[np.argmax(fft_mag)])

    # ── Tempo ─────────────────────────────────────────────────────
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo_bpm"] = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    # ── Metadata ──────────────────────────────────────────────────
    features["sample_rate_hz"]   = int(sr)
    features["duration_sec"]     = float(len(y) / sr)
    features["recording_env"]    = "uploaded"
    features["ambient_noise_db"] = float(
        20 * np.log10(np.mean(np.abs(y)) + 1e-9) + 96
    )

    return features


def extract_from_bytes(audio_bytes: bytes, fmt: str = "wav") -> dict:
    """Extract features from raw bytes (for API uploads)."""
    import io, soundfile as sf, tempfile, os

    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmppath = tmp.name

    try:
        return extract_from_file(tmppath)
    finally:
        os.unlink(tmppath)
