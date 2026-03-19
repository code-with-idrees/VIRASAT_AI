#!/usr/bin/env python3
"""
audio_enhancer.py — Noise Reduction Engine for Heritage Audio
=============================================================
Implements spectral subtraction and Wiener filter for cleaning
old recordings before stem separation.

Mathematical basis (from Implementation Plan Part B5):
  Spectral Subtraction: |Ŝ_clean|² = |S_noisy|² - α·|N̂|²
  Wiener Filter: H(ω) = SNR_prior(ω) / (1 + SNR_prior(ω))

Usage:
  python audio_enhancer.py --input noisy_song.wav --output clean_song.wav
  python audio_enhancer.py --input noisy_song.wav --method wiener
"""

import argparse
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

N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 44100


# ─── Mathematical Core ───────────────────────────────────────

def estimate_noise_spectrum(y, sr=SAMPLE_RATE, noise_duration=0.5):
    """
    Estimate noise spectrum from the first N seconds of audio.
    Assumes the first segment is a "silence" or noise-only region.

    Parameters:
        y:              Audio signal
        sr:             Sample rate
        noise_duration: Seconds of audio to use as noise estimate

    Returns:
        Noise magnitude spectrum (averaged over frames)
    """
    noise_samples = int(noise_duration * sr)
    noise_segment = y[:min(noise_samples, len(y))]

    S_noise = librosa.stft(noise_segment, n_fft=N_FFT, hop_length=HOP_LENGTH)
    noise_mag = np.mean(np.abs(S_noise), axis=1)  # Average across frames

    return noise_mag


def spectral_subtraction(y, sr=SAMPLE_RATE, alpha=1.5, beta=0.02, noise_duration=0.5):
    """
    Spectral Subtraction noise reduction.

    |Ŝ_clean(ω)|² = max(|S_noisy(ω)|² - α·|N̂(ω)|², β·|S_noisy(ω)|²)

    Parameters:
        y:     Input audio signal
        sr:    Sample rate
        alpha: Over-subtraction factor (1.0-2.0, higher = more aggressive)
        beta:  Spectral floor (0.01-0.05, prevents musical noise)
        noise_duration: Seconds for noise estimation

    Returns:
        Enhanced audio signal (numpy array)
    """
    # Estimate noise spectrum
    noise_mag = estimate_noise_spectrum(y, sr, noise_duration)

    # STFT of full signal
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_mag = np.abs(S)
    S_phase = np.angle(S)

    # Power spectra
    S_power = S_mag ** 2
    N_power = (noise_mag ** 2)[:, np.newaxis]

    # Spectral subtraction with spectral floor
    S_clean_power = np.maximum(S_power - alpha * N_power, beta * S_power)

    # Reconstruct magnitude
    S_clean_mag = np.sqrt(S_clean_power)

    # Reconstruct complex STFT (using original phase)
    S_clean = S_clean_mag * np.exp(1j * S_phase)

    # Inverse STFT
    y_clean = librosa.istft(S_clean, hop_length=HOP_LENGTH, length=len(y))

    return y_clean


def wiener_filter(y, sr=SAMPLE_RATE, noise_duration=0.5):
    """
    Wiener Filter noise reduction.

    H(ω) = SNR_prior(ω) / (1 + SNR_prior(ω))
    Ŝ_clean(ω) = H(ω) · S_noisy(ω)

    The Wiener filter minimizes mean squared error between
    estimated and true clean signal.

    Parameters:
        y:     Input audio signal
        sr:    Sample rate
        noise_duration: Seconds for noise estimation

    Returns:
        Enhanced audio signal (numpy array)
    """
    # Estimate noise spectrum
    noise_mag = estimate_noise_spectrum(y, sr, noise_duration)

    # STFT of full signal
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_mag = np.abs(S)
    S_phase = np.angle(S)

    # Power spectra
    S_power = S_mag ** 2
    N_power = (noise_mag ** 2)[:, np.newaxis]

    # A priori SNR estimate
    SNR_prior = np.maximum(S_power / (N_power + 1e-10) - 1, 0)

    # Wiener filter gain
    H = SNR_prior / (1 + SNR_prior)

    # Apply filter
    S_clean_mag = H * S_mag

    # Reconstruct
    S_clean = S_clean_mag * np.exp(1j * S_phase)
    y_clean = librosa.istft(S_clean, hop_length=HOP_LENGTH, length=len(y))

    return y_clean


def cascaded_enhancement(y, sr=SAMPLE_RATE, noise_duration=0.5):
    """
    Two-stage enhancement: Spectral Subtraction → Wiener Filter.
    Used for heavily degraded recordings (SNR 5-15 dB).
    """
    # Stage 1: Spectral subtraction (aggressive, alpha=2.0)
    y_stage1 = spectral_subtraction(y, sr, alpha=2.0, beta=0.03, noise_duration=noise_duration)

    # Stage 2: Wiener filter (refine)
    y_stage2 = wiener_filter(y_stage1, sr, noise_duration=noise_duration)

    return y_stage2


# ─── Pipeline ────────────────────────────────────────────────

def enhance_audio(input_path, output_path=None, method="auto", noise_duration=0.5):
    """
    Full audio enhancement pipeline.

    Parameters:
        input_path:     Path to noisy audio
        output_path:    Path for enhanced output (auto-generated if None)
        method:         'spectral_subtraction', 'wiener', 'cascaded', or 'auto'
        noise_duration: Seconds for noise estimation

    Returns:
        dict: {success, output_path, method_used, snr_before, snr_after}
    """
    if not AUDIO_AVAILABLE:
        raise ImportError("librosa and soundfile required")

    input_path = Path(input_path)

    # Load audio
    y, sr = librosa.load(str(input_path), sr=SAMPLE_RATE, mono=True)

    # Estimate SNR for auto mode
    from noise_estimator import estimate_snr_from_signal
    snr_before = estimate_snr_from_signal(y, sr)

    # Auto-select method based on SNR
    if method == "auto":
        if snr_before > 30:
            print(f"   SNR: {snr_before:.1f} dB — No enhancement needed")
            return {
                "success": True,
                "method_used": "none",
                "snr_before": snr_before,
                "snr_after": snr_before,
                "output_path": str(input_path),
            }
        elif snr_before > 15:
            method = "wiener"
        else:
            method = "cascaded"

    print(f"   SNR before: {snr_before:.1f} dB")
    print(f"   Method: {method}")

    # Apply enhancement
    if method == "spectral_subtraction":
        y_clean = spectral_subtraction(y, sr, noise_duration=noise_duration)
    elif method == "wiener":
        y_clean = wiener_filter(y, sr, noise_duration=noise_duration)
    elif method == "cascaded":
        y_clean = cascaded_enhancement(y, sr, noise_duration=noise_duration)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Measure improvement
    snr_after = estimate_snr_from_signal(y_clean, sr)

    # Save output
    if output_path is None:
        output_path = input_path.parent.parent / "enhanced" / f"{input_path.stem}_enhanced.wav"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), y_clean, sr)

    improvement = snr_after - snr_before
    print(f"   SNR after:  {snr_after:.1f} dB (Δ {improvement:+.1f} dB)")
    print(f"   ✅ Saved: {output_path}")

    return {
        "success": True,
        "method_used": method,
        "snr_before": snr_before,
        "snr_after": snr_after,
        "improvement_db": round(improvement, 2),
        "output_path": str(output_path),
    }


# ─── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VIRASAT AI — Audio Enhancement (Noise Reduction)",
    )

    parser.add_argument("--input", "-i", required=True, help="Input audio file")
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    parser.add_argument("--method", "-m", default="auto",
                        choices=["auto", "spectral_subtraction", "wiener", "cascaded"],
                        help="Enhancement method")
    parser.add_argument("--noise-duration", type=float, default=0.5,
                        help="Seconds of audio for noise estimation (default: 0.5)")

    args = parser.parse_args()

    result = enhance_audio(
        input_path=args.input,
        output_path=args.output,
        method=args.method,
        noise_duration=args.noise_duration,
    )

    if result["method_used"] != "none":
        print(f"\n📊 Enhancement Result:")
        print(f"   Method:      {result['method_used']}")
        print(f"   SNR before:  {result['snr_before']:.1f} dB")
        print(f"   SNR after:   {result['snr_after']:.1f} dB")
        print(f"   Improvement: {result['improvement_db']:+.1f} dB")


if __name__ == "__main__":
    main()
