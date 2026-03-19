#!/usr/bin/env python3
"""
bleed_detector.py — Instrument Bleed Detection Engine
=====================================================
Spectral analysis script to measure instrument bleed in vocal stems.
Uses frequency-domain energy ratios with instrument-specific fingerprints.

Mathematical basis (from Implementation Plan Part B3):
  Bleed_i = 10 * log10(E_instrument_band / E_vocal_band)
  Where E = sum of |STFT(x)|^2 over the instrument's frequency range

Usage:
  python bleed_detector.py --input data/stems/vocals.wav
  python bleed_detector.py --input data/stems/ --report
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ─── Configuration ────────────────────────────────────────────

# Load instrument profiles from config
CONFIG_DIR = Path(__file__).parent.parent / "config"
DEFAULT_PROFILES_PATH = CONFIG_DIR / "instrument_profiles.json"
DEFAULT_THRESHOLDS_PATH = CONFIG_DIR / "quality_thresholds.json"

# Fallback frequency ranges if config not found
FALLBACK_INSTRUMENTS = {
    "sitar": {"low_hz": 200, "high_hz": 700, "threshold_db": -15},
    "tabla": {"low_hz": 60, "high_hz": 300, "threshold_db": -12},
    "harmonium": {"low_hz": 200, "high_hz": 800, "threshold_db": -15},
    "tanpura": {"low_hz": 60, "high_hz": 250, "threshold_db": -10},
    "dhol": {"low_hz": 50, "high_hz": 150, "threshold_db": -10},
    "sarangi": {"low_hz": 250, "high_hz": 1200, "threshold_db": -15},
    "electric_guitar": {"low_hz": 80, "high_hz": 1200, "threshold_db": -15},
    "synthesizer": {"low_hz": 200, "high_hz": 4000, "threshold_db": -12},
}

# Vocal reference band (speech intelligibility)
VOCAL_BAND = {"low_hz": 300, "high_hz": 3400}

# STFT parameters
N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 44100


# ─── Core Functions ──────────────────────────────────────────

def load_instrument_profiles():
    """Load instrument frequency profiles from config or use fallback."""
    if DEFAULT_PROFILES_PATH.exists():
        with open(DEFAULT_PROFILES_PATH) as f:
            data = json.load(f)

        profiles = {}
        for name, info in data.get("instruments", {}).items():
            ranges = info.get("frequency_ranges", {})
            # Use the fundamental range for bleed detection
            if "fundamental" in ranges:
                profiles[name] = {
                    "low_hz": ranges["fundamental"]["low_hz"],
                    "high_hz": ranges["fundamental"]["high_hz"],
                    "threshold_db": info.get("bleed_threshold_db", -15),
                }
            elif "fundamental_reed" in ranges:
                profiles[name] = {
                    "low_hz": ranges["fundamental_reed"]["low_hz"],
                    "high_hz": ranges["fundamental_reed"]["high_hz"],
                    "threshold_db": info.get("bleed_threshold_db", -15),
                }
            elif "bass_side" in ranges:
                profiles[name] = {
                    "low_hz": ranges["bass_side"]["low_hz"],
                    "high_hz": ranges["bass_side"]["high_hz"],
                    "threshold_db": info.get("bleed_threshold_db", -10),
                }
            elif "bayan_bass" in ranges:
                profiles[name] = {
                    "low_hz": ranges["bayan_bass"]["low_hz"],
                    "high_hz": ranges["dayan_treble"]["high_hz"],
                    "threshold_db": info.get("bleed_threshold_db", -12),
                }

        return profiles if profiles else FALLBACK_INSTRUMENTS

    return FALLBACK_INSTRUMENTS


def compute_spectral_energy(S_power, freqs, low_hz, high_hz):
    """
    Compute total spectral energy within a frequency band.

    E_band = Σ |S(ω)|² for ω ∈ [low_hz, high_hz]

    Parameters:
        S_power: Power spectrogram (magnitude squared)
        freqs:   Frequency array from librosa.fft_frequencies
        low_hz:  Lower frequency bound
        high_hz: Upper frequency bound

    Returns:
        Total energy (float)
    """
    band_mask = np.where((freqs >= low_hz) & (freqs <= high_hz))[0]
    if len(band_mask) == 0:
        return 1e-10  # Avoid division by zero
    return np.sum(S_power[band_mask, :])


def compute_bleed_scores(audio_path, sr=SAMPLE_RATE, instruments=None):
    """
    Compute instrument bleed scores for a vocal stem.

    For each instrument i:
      Bleed_i = 10 * log10(E_instrument_band / E_vocal_band)

    Parameters:
        audio_path:  Path to vocal stem WAV file
        sr:          Sample rate
        instruments: Dict of instrument profiles (name → {low_hz, high_hz, threshold_db})

    Returns:
        dict: {
            instrument_name: {
                energy_ratio_dB: float,
                is_bleeding: bool,
                severity: str ('CLEAN' | 'MILD' | 'SEVERE'),
                threshold_db: float
            }
        }
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required. Install: pip install librosa")

    if instruments is None:
        instruments = load_instrument_profiles()

    # Load audio
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)

    # Compute STFT → power spectrogram
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_power = np.abs(S) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    # Compute vocal band energy (reference)
    E_vocal = compute_spectral_energy(
        S_power, freqs,
        VOCAL_BAND["low_hz"], VOCAL_BAND["high_hz"]
    )

    # Compute per-instrument bleed scores
    bleed_scores = {}
    for name, profile in instruments.items():
        E_inst = compute_spectral_energy(
            S_power, freqs,
            profile["low_hz"], profile["high_hz"]
        )

        # Bleed ratio in dB
        bleed_db = 10 * np.log10(E_inst / E_vocal + 1e-10)
        threshold = profile["threshold_db"]

        # Classify severity
        if bleed_db < threshold - 5:
            severity = "CLEAN"
        elif bleed_db < threshold:
            severity = "MILD"
        else:
            severity = "SEVERE"

        bleed_scores[name] = {
            "energy_ratio_dB": round(float(bleed_db), 2),
            "is_bleeding": bleed_db >= threshold,
            "severity": severity,
            "threshold_db": threshold,
            "frequency_range_hz": f"{profile['low_hz']}-{profile['high_hz']}",
        }

    return bleed_scores


def compute_overall_bleed_score(bleed_scores):
    """
    Compute an overall bleed score (0-100 scale, lower = cleaner).

    Score = 100 if any SEVERE, else weighted average based on margin to threshold.
    """
    if not bleed_scores:
        return 0

    has_severe = any(s["severity"] == "SEVERE" for s in bleed_scores.values())
    if has_severe:
        # Count how many are severe
        severe_count = sum(1 for s in bleed_scores.values() if s["severity"] == "SEVERE")
        return min(100, 50 + severe_count * 15)

    # Compute score based on distance from thresholds
    margins = []
    for name, score in bleed_scores.items():
        # How far below the threshold (positive = good, negative = bad)
        margin = score["threshold_db"] - score["energy_ratio_dB"]
        margins.append(max(0, 20 - margin) * 5)  # Scale to 0-100

    return min(100, int(np.mean(margins)))


def generate_spectrogram_plot(audio_path, bleed_scores, output_path=None):
    """Generate a visual spectrogram with instrument frequency bands overlaid."""
    if not MATPLOTLIB_AVAILABLE or not LIBROSA_AVAILABLE:
        print("⚠️  matplotlib and librosa required for visualization")
        return None

    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)))

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot spectrogram
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="hz",
        ax=ax, cmap="magma"
    )

    # Overlay instrument frequency bands
    colors = {
        "CLEAN": "#00ff88",
        "MILD": "#ffaa00",
        "SEVERE": "#ff3333",
    }

    instruments = load_instrument_profiles()
    for name, score in bleed_scores.items():
        if name in instruments:
            profile = instruments[name]
            color = colors.get(score["severity"], "#ffffff")
            ax.axhspan(
                profile["low_hz"], profile["high_hz"],
                alpha=0.15, color=color,
                label=f"{name} ({score['severity']}: {score['energy_ratio_dB']:.1f} dB)"
            )

    # Vocal band reference
    ax.axhspan(
        VOCAL_BAND["low_hz"], VOCAL_BAND["high_hz"],
        alpha=0.1, color="#00aaff",
        label=f"Vocal band ({VOCAL_BAND['low_hz']}-{VOCAL_BAND['high_hz']} Hz)",
        linestyle="--", linewidth=2
    )

    ax.set_title(f"VIRASAT AI — Bleed Analysis: {Path(audio_path).stem}", fontsize=14)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📊 Spectrogram saved: {output_path}")
    else:
        plt.show()

    plt.close(fig)
    return output_path


def analyze_path(input_path, generate_plots=False, report_dir=None):
    """Analyze a file or directory of vocal stems."""
    input_path = Path(input_path)
    results = {}

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("**/*vocals*.wav")) + \
                sorted(input_path.glob("**/*vocal*.wav"))
        if not files:
            # Fallback: try all WAV files
            files = sorted(input_path.glob("**/*.wav"))
    else:
        print(f"❌ Not found: {input_path}")
        return results

    for audio_file in files:
        print(f"\n🎤 Analyzing: {audio_file.name}")
        print(f"   {'─' * 40}")

        bleed_scores = compute_bleed_scores(audio_file)
        overall_score = compute_overall_bleed_score(bleed_scores)

        # Print results
        for name, score in bleed_scores.items():
            icon = "✅" if score["severity"] == "CLEAN" else "⚠️" if score["severity"] == "MILD" else "❌"
            print(f"   {icon} {name:15s} │ {score['energy_ratio_dB']:+7.2f} dB │ "
                  f"threshold: {score['threshold_db']:+6.1f} dB │ {score['severity']}")

        # Overall assessment
        if overall_score < 20:
            print(f"\n   🏆 Overall: CLEAN (score: {overall_score}/100)")
        elif overall_score < 50:
            print(f"\n   ⚠️  Overall: MILD BLEED (score: {overall_score}/100)")
        else:
            print(f"\n   ❌ Overall: SIGNIFICANT BLEED (score: {overall_score}/100)")

        # Generate plot if requested
        if generate_plots:
            plot_dir = report_dir or (input_path.parent / "reports")
            plot_dir = Path(plot_dir)
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / f"bleed_{audio_file.stem}.png"
            generate_spectrogram_plot(audio_file, bleed_scores, plot_path)

        results[str(audio_file)] = {
            "bleed_scores": bleed_scores,
            "overall_score": overall_score,
        }

    return results


# ─── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VIRASAT AI — Instrument Bleed Detection Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", "-i", required=True,
                        help="Vocal stem WAV file or directory")
    parser.add_argument("--report", "-r", action="store_true",
                        help="Generate spectrogram plots")
    parser.add_argument("--report-dir", default=None,
                        help="Directory for report output")
    parser.add_argument("--save-json", default=None,
                        help="Save results as JSON")

    args = parser.parse_args()

    results = analyze_path(
        args.input,
        generate_plots=args.report,
        report_dir=args.report_dir,
    )

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved: {output_path}")


if __name__ == "__main__":
    main()
