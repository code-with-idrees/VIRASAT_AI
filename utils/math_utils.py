#!/usr/bin/env python3
"""
math_utils.py — DSP Mathematical Utilities for VIRASAT AI
==========================================================
Core mathematical functions used across the project:
- STFT computation
- Pitch Class Profile (PCP)
- Frequency ↔ pitch class mapping
- Cosine similarity
- Normalization functions
"""

import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# ─── Constants ────────────────────────────────────────────────

PITCH_CLASS_NAMES = [
    "Sa", "Re♭", "Re", "Ga♭", "Ga", "Ma", "Ma#",
    "Pa", "Dha♭", "Dha", "Ni♭", "Ni"
]

WESTERN_NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F", "F#",
    "G", "G#", "A", "A#", "B"
]

DEFAULT_TONIC_HZ = 261.63  # C4


# ─── Frequency / Pitch Conversions ──────────────────────────

def hz_to_pitch_class(freq, tonic_hz=DEFAULT_TONIC_HZ):
    """Map frequency (Hz) to pitch class (0-11)."""
    if freq <= 0 or tonic_hz <= 0:
        return -1
    return int(round(12 * np.log2(freq / tonic_hz))) % 12


def pitch_class_to_hz(pc, tonic_hz=DEFAULT_TONIC_HZ, octave=4):
    """Map pitch class (0-11) to frequency (Hz) in given octave."""
    return tonic_hz * (2 ** (pc / 12)) * (2 ** (octave - 4))


def hz_to_midi(freq):
    """Convert frequency to MIDI note number."""
    if freq <= 0:
        return 0
    return int(round(69 + 12 * np.log2(freq / 440.0)))


def midi_to_hz(midi_note):
    """Convert MIDI note number to frequency."""
    return 440.0 * (2 ** ((midi_note - 69) / 12))


# ─── Spectral Analysis ──────────────────────────────────────

def compute_stft(y, n_fft=2048, hop_length=512):
    """
    Compute Short-Time Fourier Transform.

    STFT{x[n]}(m, ω) = Σ x[n] · w[n-m] · e^(-jωn)

    Returns:
        Complex STFT matrix, frequency array
    """
    if LIBROSA_AVAILABLE:
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=44100, n_fft=n_fft)
        return S, freqs
    else:
        # Manual STFT using numpy
        window = np.hanning(n_fft)
        n_frames = 1 + (len(y) - n_fft) // hop_length
        S = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)

        for i in range(n_frames):
            start = i * hop_length
            frame = y[start:start + n_fft] * window
            S[:, i] = np.fft.rfft(frame)

        freqs = np.fft.rfftfreq(n_fft, 1.0 / 44100)
        return S, freqs


def compute_magnitude_spectrogram(S):
    """Magnitude spectrogram from complex STFT."""
    return np.abs(S)


def compute_power_spectrogram(S):
    """Power spectrogram (|S|²) from complex STFT."""
    return np.abs(S) ** 2


def compute_pcp(S_power, freqs, tonic_hz=DEFAULT_TONIC_HZ):
    """
    Compute Pitch Class Profile from power spectrogram.

    PCP[k] = Σ |S(ω)|² for all ω mapping to pitch class k
    """
    pcp = np.zeros(12)
    for i, freq in enumerate(freqs):
        if freq < 50 or freq > 8000:
            continue
        pc = hz_to_pitch_class(freq, tonic_hz)
        if 0 <= pc < 12:
            pcp[pc] += np.sum(S_power[i, :])

    total = np.sum(pcp)
    if total > 0:
        pcp /= total
    return pcp


def compute_spectral_centroid(S_mag, freqs):
    """
    Spectral centroid — "center of mass" of the spectrum.
    centroid = Σ(f * |S(f)|) / Σ|S(f)|

    Higher centroid → brighter sound.
    """
    weighted_sum = np.sum(freqs[:, np.newaxis] * S_mag, axis=0)
    magnitude_sum = np.sum(S_mag, axis=0)
    centroid = weighted_sum / (magnitude_sum + 1e-10)
    return centroid


def compute_spectral_bandwidth(S_mag, freqs, centroid):
    """
    Spectral bandwidth — spread of the spectrum around centroid.
    bandwidth = sqrt(Σ((f - centroid)² * |S(f)|) / Σ|S(f)|)
    """
    deviation = freqs[:, np.newaxis] - centroid[np.newaxis, :]
    weighted_var = np.sum(deviation ** 2 * S_mag, axis=0)
    magnitude_sum = np.sum(S_mag, axis=0)
    bandwidth = np.sqrt(weighted_var / (magnitude_sum + 1e-10))
    return bandwidth


# ─── Similarity & Distance ──────────────────────────────────

def cosine_similarity(a, b):
    """
    Cosine similarity: sim(a,b) = (a·b) / (||a|| × ||b||)
    Range: [0, 1] for non-negative vectors
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(dot / (norm_a * norm_b))


def euclidean_distance(a, b):
    """Euclidean distance: d(a,b) = ||a - b||"""
    return float(np.linalg.norm(a - b))


def kl_divergence(p, q):
    """
    Kullback-Leibler divergence: D_KL(P || Q) = Σ P(i) * log(P(i)/Q(i))
    Measures how one probability distribution differs from another.
    """
    p = np.asarray(p, dtype=float) + 1e-10
    q = np.asarray(q, dtype=float) + 1e-10
    return float(np.sum(p * np.log(p / q)))


# ─── Normalization ───────────────────────────────────────────

def normalize_min_max(x, x_min=None, x_max=None, scale=100):
    """Min-max normalization to [0, scale]."""
    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)
    if x_max == x_min:
        return 0.0 if np.isscalar(x) else np.zeros_like(x)
    result = (x - x_min) / (x_max - x_min) * scale
    return np.clip(result, 0, scale)


def normalize_z_score(x):
    """Z-score normalization: z = (x - μ) / σ"""
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return np.zeros_like(x)
    return (x - mean) / std


def db_to_linear(db):
    """Convert decibels to linear scale: linear = 10^(dB/10)"""
    return 10 ** (db / 10)


def linear_to_db(linear):
    """Convert linear scale to decibels: dB = 10*log10(linear)"""
    return 10 * np.log10(np.maximum(linear, 1e-10))
