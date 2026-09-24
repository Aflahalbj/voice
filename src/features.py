"""
features.py
-----------
MFCC-based feature extraction.

Each audio segment becomes ONE fixed-length vector:
    MFCC (40) + delta (40) + delta-delta (40)
    + spectral centroid + spectral rolloff + zero crossing rate + chroma (12)
    -> mean and standard deviation of every row  => 2 * (120 + 3 + 12) = 270 values

Standalone usage:
    python src/features.py path/to/audio.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np

from src.preprocess import SAMPLE_RATE, AudioProcessingError, preprocess_file

N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
N_CHROMA = 12
FEATURE_DIM = 2 * (3 * N_MFCC + 3 + N_CHROMA)  # 270


def compute_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return the MFCC matrix (N_MFCC x frames). Also used for heatmap visualization."""
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)


def _delta(features: np.ndarray, order: int) -> np.ndarray:
    """Delta features that also work on very short segments."""
    n_frames = features.shape[1]
    if n_frames < 3:
        return np.zeros_like(features)
    width = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
    return librosa.feature.delta(features, width=width, order=order, mode="interp")


def _mean_std(matrix: np.ndarray) -> np.ndarray:
    """Aggregate a (rows x frames) matrix into [mean of each row, std of each row]."""
    return np.concatenate([matrix.mean(axis=1), matrix.std(axis=1)])


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract a fixed-length feature vector (FEATURE_DIM,) from one segment."""
    if y.size < N_FFT:
        y = np.pad(y, (0, N_FFT - y.size))

    # One STFT shared by all spectral features
    magnitude = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    power = magnitude ** 2

    mel = librosa.feature.melspectrogram(S=power, sr=sr)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC)
    mfcc_delta = _delta(mfcc, order=1)
    mfcc_delta2 = _delta(mfcc, order=2)

    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr, roll_percent=0.85)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    chroma = librosa.feature.chroma_stft(S=power, sr=sr, n_chroma=N_CHROMA)

    vector = np.concatenate([
        _mean_std(mfcc),
        _mean_std(mfcc_delta),
        _mean_std(mfcc_delta2),
        _mean_std(centroid),
        _mean_std(rolloff),
        _mean_std(zcr),
        _mean_std(chroma),
    ])
    return np.nan_to_num(vector).astype(np.float32)


def extract_features_batch(
    segments: list[np.ndarray],
    sr: int = SAMPLE_RATE,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Extract features for many segments -> array of shape (n_segments, FEATURE_DIM)."""
    if not segments:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)
    rows = []
    total = len(segments)
    for i, seg in enumerate(segments, 1):
        rows.append(extract_features(seg, sr))
        if progress_callback is not None:
            progress_callback(i, total)
    return np.vstack(rows)


def feature_names() -> list[str]:
    """Human-readable name of each position in the feature vector."""
    blocks = [
        ("mfcc", N_MFCC), ("mfcc_delta", N_MFCC), ("mfcc_delta2", N_MFCC),
        ("spectral_centroid", 1), ("spectral_rolloff", 1), ("zcr", 1), ("chroma", N_CHROMA),
    ]
    names: list[str] = []
    for prefix, size in blocks:
        for stat in ("mean", "std"):
            names += [f"{prefix}_{i + 1}_{stat}" if size > 1 else f"{prefix}_{stat}" for i in range(size)]
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract MFCC features from an audio file.")
    parser.add_argument("audio", help="Path to an audio file")
    parser.add_argument("--no-noise-reduction", action="store_true", help="Disable noise reduction")
    args = parser.parse_args(argv)

    try:
        _, segments = preprocess_file(args.audio, noise_reduction=not args.no_noise_reduction)
    except AudioProcessingError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    X = extract_features_batch(segments)
    names = feature_names()
    print(f"Segments       : {X.shape[0]}")
    print(f"Feature vector : {X.shape[1]} values (expected {FEATURE_DIM})")
    print("First segment preview:")
    for name, value in list(zip(names, X[0]))[:8]:
        print(f"  {name:<24} {value:10.4f}")
    print("  ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
