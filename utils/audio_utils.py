#!/usr/bin/env python3
"""
audio_utils.py — Common Audio I/O Utilities
============================================
Shared audio loading, saving, resampling, and format
conversion functions used across all VIRASAT AI scripts.
"""

import os
from pathlib import Path

import numpy as np

try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


# ─── Constants ────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BIT_DEPTH = 16


# ─── Audio I/O ───────────────────────────────────────────────

def load_audio(path, sr=DEFAULT_SAMPLE_RATE, mono=True):
    """
    Load audio file with consistent settings.

    Returns:
        tuple: (audio_array, sample_rate)
    """
    if not AUDIO_AVAILABLE:
        raise ImportError("librosa and soundfile required. Install: pip install librosa soundfile")

    path = str(path)
    y, sr_loaded = librosa.load(path, sr=sr, mono=mono)
    return y, sr_loaded


def save_audio(audio, path, sr=DEFAULT_SAMPLE_RATE):
    """Save audio array to WAV file."""
    if not AUDIO_AVAILABLE:
        raise ImportError("soundfile required")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype='PCM_16')


def get_audio_info(path):
    """Get audio file metadata without loading full file."""
    if not AUDIO_AVAILABLE:
        raise ImportError("soundfile required")

    info = sf.info(str(path))
    return {
        "path": str(path),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "duration_seconds": round(info.duration, 2),
        "frames": info.frames,
        "format": info.format,
        "subtype": info.subtype,
    }


def find_audio_files(directory, recursive=True):
    """Find all audio files in a directory."""
    directory = Path(directory)
    files = []

    if recursive:
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(directory.rglob(f"*{ext}"))
    else:
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(directory.glob(f"*{ext}"))

    return sorted(files)


def resample(audio, sr_original, sr_target):
    """Resample audio to target sample rate."""
    if sr_original == sr_target:
        return audio
    if not AUDIO_AVAILABLE:
        raise ImportError("librosa required for resampling")
    return librosa.resample(audio, orig_sr=sr_original, target_sr=sr_target)


def to_mono(audio):
    """Convert stereo to mono by averaging channels."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=0)


def normalize_volume(audio, target_db=-20):
    """Normalize audio volume to target dB level."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return audio
    target_rms = 10 ** (target_db / 20)
    return audio * (target_rms / rms)


def trim_silence(audio, sr=DEFAULT_SAMPLE_RATE, top_db=30):
    """Trim leading and trailing silence."""
    if not AUDIO_AVAILABLE:
        raise ImportError("librosa required")
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed
