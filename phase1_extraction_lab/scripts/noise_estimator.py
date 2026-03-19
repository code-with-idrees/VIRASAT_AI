#!/usr/bin/env python3
"""
noise_estimator.py — SNR Estimation & Pre-processing Decision Engine
=====================================================================
Estimates the Signal-to-Noise Ratio of heritage recordings and decides
which noise reduction pipeline to apply before Demucs separation.

Mathematical basis (from Implementation Plan Part B5):
  SNR = 10 * log10(P_signal / P_noise)

Decision flow:
  SNR > 30 dB  → No pre-processing needed (direct Demucs)
  SNR 15-30 dB → Wiener Filter → Demucs
  SNR 5-15 dB  → Spectral Subtraction → Wiener Filter → Demucs
  SNR < 5 dB   → Adobe Enhance API → Demucs

Usage:
  python noise_estimator.py --input data/raw/song.wav
  python noise_estimator.py --input data/raw/ --report
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


# ─── Configuration ────────────────────────────────────────────

FRAME_SIZE = 2048
HOP_SIZE = 512
SAMPLE_RATE = 44100

# SNR decision thresholds
SNR_THRESHOLDS = {
    "good":     {"min_snr": 30, "preprocessing": "none",
                 "description": "Good quality — direct Demucs"},
    "moderate": {"min_snr": 15, "preprocessing": "wiener_filter",
                 "description": "Moderate noise — apply Wiener Filter first"},
    "heavy":    {"min_snr": 5,  "preprocessing": "spectral_subtraction_then_wiener",
                 "description": "Heavy noise — Spectral Subtraction → Wiener → Demucs"},
    "severe":   {"min_snr": 0,  "preprocessing": "adobe_enhance_api",
                 "description": "Severe noise — use Adobe Enhance API (AI-based)"},
}


# ─── Mathematical Functions ──────────────────────────────────

def estimate_snr_from_signal(y, sr=SAMPLE_RATE, method="percentile"):
    """
    Estimate SNR from a single audio signal.

    Method 1: Percentile-based
      - Signal power = mean power of top 70% frames (assumed to contain signal)
      - Noise power = mean power of bottom 10% frames (assumed to be noise/silence)

    Method 2: VAD-based (Voice Activity Detection)
      - Use energy threshold to separate speech from silence
      - More accurate but slower

    Parameters:
        y:       Audio signal (numpy array)
        sr:      Sample rate
        method:  'percentile' or 'vad'

    Returns:
        float: Estimated SNR in dB
    """
    # Compute frame-level RMS energy
    rms = librosa.feature.rms(y=y, frame_length=FRAME_SIZE, hop_length=HOP_SIZE)[0]
    rms_power = rms ** 2

    if len(rms_power) == 0:
        return 0.0

    if method == "percentile":
        # Signal = top 70% energy frames
        signal_threshold = np.percentile(rms_power, 30)
        signal_frames = rms_power[rms_power >= signal_threshold]
        signal_power = np.mean(signal_frames) if len(signal_frames) > 0 else 1e-10

        # Noise = bottom 10% energy frames
        noise_threshold = np.percentile(rms_power, 10)
        noise_frames = rms_power[rms_power <= noise_threshold]
        noise_power = np.mean(noise_frames) if len(noise_frames) > 0 else 1e-10

    elif method == "vad":
        # Simple energy-based VAD
        energy_threshold = np.mean(rms_power) * 0.1  # 10% of mean energy
        signal_frames = rms_power[rms_power > energy_threshold]
        noise_frames = rms_power[rms_power <= energy_threshold]

        signal_power = np.mean(signal_frames) if len(signal_frames) > 0 else 1e-10
        noise_power = np.mean(noise_frames) if len(noise_frames) > 0 else 1e-10

    else:
        raise ValueError(f"Unknown method: {method}")

    # Avoid extreme values
    if noise_power < 1e-12:
        return 60.0  # Cap at 60 dB (very clean)

    snr = 10 * np.log10(signal_power / noise_power)
    return round(float(np.clip(snr, -10, 60)), 2)


def estimate_spectral_snr(y, sr=SAMPLE_RATE):
    """
    Frequency-band SNR estimation.
    Computes SNR separately for low, mid, and high frequency bands.

    Returns:
        dict: {overall_snr, low_band_snr, mid_band_snr, high_band_snr}
    """
    S = np.abs(librosa.stft(y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE))
    S_power = S ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FRAME_SIZE)

    bands = {
        "low":  (50, 300),    # Bass, hum
        "mid":  (300, 3400),  # Speech band
        "high": (3400, 8000), # Hiss, sibilance
    }

    band_snrs = {}
    for band_name, (f_low, f_high) in bands.items():
        band_mask = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        if len(band_mask) == 0:
            band_snrs[f"{band_name}_band_snr_db"] = 0.0
            continue

        band_power = S_power[band_mask, :]
        frame_energies = np.mean(band_power, axis=0)

        # Same percentile approach per band
        if len(frame_energies) > 0:
            signal_power = np.mean(frame_energies[frame_energies >= np.percentile(frame_energies, 30)])
            noise_frames = frame_energies[frame_energies <= np.percentile(frame_energies, 10)]
            noise_power = np.mean(noise_frames) if len(noise_frames) > 0 else 1e-12

            snr = 10 * np.log10(signal_power / noise_power + 1e-10)
            band_snrs[f"{band_name}_band_snr_db"] = round(float(np.clip(snr, -10, 60)), 2)
        else:
            band_snrs[f"{band_name}_band_snr_db"] = 0.0

    return band_snrs


def detect_noise_type(y, sr=SAMPLE_RATE):
    """
    Detect common noise types in heritage recordings.

    Checks for:
      - 50/60 Hz hum (power line interference)
      - Broadband hiss (tape noise)
      - Vinyl crackle (impulsive noise)
    """
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=HOP_SIZE))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    noise_types = []

    # Check for 50 Hz hum
    hum_50_mask = np.where((freqs >= 48) & (freqs <= 52))[0]
    hum_60_mask = np.where((freqs >= 58) & (freqs <= 62))[0]
    overall_mean = np.mean(S)

    for hum_mask, hum_freq in [(hum_50_mask, 50), (hum_60_mask, 60)]:
        if len(hum_mask) > 0:
            hum_energy = np.mean(S[hum_mask, :])
            if hum_energy > overall_mean * 3:
                noise_types.append(f"{hum_freq}hz_hum")

    # Check for broadband hiss (elevated energy above 4 kHz)
    high_mask = np.where(freqs >= 4000)[0]
    mid_mask = np.where((freqs >= 500) & (freqs <= 2000))[0]
    if len(high_mask) > 0 and len(mid_mask) > 0:
        high_energy = np.mean(S[high_mask, :])
        mid_energy = np.mean(S[mid_mask, :])
        if high_energy > mid_energy * 0.5:
            noise_types.append("broadband_hiss")

    # Check for crackle (impulsive noise — high crest factor)
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
    peak = np.max(np.abs(y))
    rms_mean = np.mean(rms)
    if rms_mean > 0 and peak / rms_mean > 15:
        noise_types.append("vinyl_crackle")

    return noise_types


def decide_preprocessing(snr_db):
    """
    Decide which preprocessing pipeline to apply based on SNR.

    Returns:
        dict: {category, preprocessing, description}
    """
    for category, info in SNR_THRESHOLDS.items():
        if snr_db >= info["min_snr"]:
            return {
                "category": category,
                "preprocessing": info["preprocessing"],
                "description": info["description"],
            }

    # Default to severe
    return {
        "category": "severe",
        "preprocessing": "adobe_enhance_api",
        "description": "Severe noise — use Adobe Enhance API",
    }


# ─── Analysis ────────────────────────────────────────────────

def analyze_recording(audio_path, sr=SAMPLE_RATE):
    """Full noise analysis of a recording."""
    if not AUDIO_AVAILABLE:
        raise ImportError("librosa required. Install: pip install librosa")

    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)

    # Overall SNR
    snr_percentile = estimate_snr_from_signal(y, sr, method="percentile")
    snr_vad = estimate_snr_from_signal(y, sr, method="vad")
    snr_avg = round((snr_percentile + snr_vad) / 2, 2)

    # Per-band SNR
    band_snrs = estimate_spectral_snr(y, sr)

    # Noise type detection
    noise_types = detect_noise_type(y, sr)

    # Preprocessing decision
    decision = decide_preprocessing(snr_avg)

    result = {
        "file": str(audio_path),
        "duration_seconds": round(len(y) / sr, 2),
        "snr_db": {
            "percentile_method": snr_percentile,
            "vad_method": snr_vad,
            "average": snr_avg,
        },
        "band_snr_db": band_snrs,
        "detected_noise_types": noise_types,
        "preprocessing_decision": decision,
    }

    return result


def analyze_directory(input_dir, sr=SAMPLE_RATE):
    """Analyze all audio files in a directory."""
    input_dir = Path(input_dir)
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = [f for f in sorted(input_dir.rglob("*")) if f.suffix.lower() in exts]

    if not files:
        print(f"❌ No audio files found in {input_dir}")
        return []

    results = []
    for audio_file in files:
        print(f"\n🔍 Analyzing: {audio_file.name}")
        result = analyze_recording(audio_file, sr)

        snr = result["snr_db"]["average"]
        decision = result["preprocessing_decision"]
        noise_types = result["detected_noise_types"]

        print(f"   SNR: {snr:.1f} dB → {decision['description']}")
        if noise_types:
            print(f"   Detected: {', '.join(noise_types)}")

        results.append(result)

    return results


# ─── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VIRASAT AI — Noise Estimation & Pre-processing Decision Engine",
    )

    parser.add_argument("--input", "-i", required=True,
                        help="Audio file or directory to analyze")
    parser.add_argument("--save-json", default=None,
                        help="Save results as JSON")
    parser.add_argument("--report", action="store_true",
                        help="Generate detailed report")

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        results = analyze_directory(args.input)
    elif input_path.is_file():
        result = analyze_recording(args.input)
        results = [result]

        snr = result["snr_db"]["average"]
        decision = result["preprocessing_decision"]
        noise_types = result["detected_noise_types"]

        print(f"\n📊 Noise Analysis: {input_path.name}")
        print(f"   Overall SNR: {snr:.1f} dB")
        print(f"   Band SNR: {json.dumps(result['band_snr_db'], indent=2)}")
        print(f"   Noise types: {noise_types or 'None detected'}")
        print(f"   ➡️  Recommendation: {decision['description']}")
        print(f"      Pipeline: {decision['preprocessing']}")
    else:
        print(f"❌ Not found: {args.input}")
        return

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved: {args.save_json}")


if __name__ == "__main__":
    main()
