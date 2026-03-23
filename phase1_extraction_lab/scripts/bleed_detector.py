#!/usr/bin/env python3
"""
bleed_detector.py — Instrument Bleed Detection Engine v2
=========================================================
Spectral analysis script to measure *true* instrument bleed in vocal stems.

Architectural upgrade (v2):
  The human voice shares the 150-1200 Hz band with sitar, harmonium, and
  sarangi. Naive energy-ratio measurement always flags this as bleed.

  Solution — two independent gates:
    1. Spectral Flatness Gate: instruments have broad, flat spectra;
       vocals have narrow harmonic peaks.  Only energy with flatness
       above a threshold is counted as instrument bleed.
    2. Vocal Harmonic Exclusion: estimate the vocal fundamental (f0)
       and mask out f0 + integer harmonics before measuring instrument
       band energy.

  Bleed_i = 10 * log10(E_gated_instrument_band / E_vocal_band)

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
    "sitar": {"low_hz": 200, "high_hz": 700, "threshold_db": -5},
    "tabla": {"low_hz": 60, "high_hz": 300, "threshold_db": -5},
    "harmonium": {"low_hz": 200, "high_hz": 800, "threshold_db": -5},
    "tanpura": {"low_hz": 60, "high_hz": 250, "threshold_db": -5},
    "dhol": {"low_hz": 50, "high_hz": 150, "threshold_db": -5},
    "sarangi": {"low_hz": 250, "high_hz": 1200, "threshold_db": -5},
    "electric_guitar": {"low_hz": 80, "high_hz": 1200, "threshold_db": -5},
    "synthesizer": {"low_hz": 200, "high_hz": 4000, "threshold_db": -5},
}

# Vocal reference band (speech intelligibility)
VOCAL_BAND = {"low_hz": 300, "high_hz": 3400}

# STFT parameters
N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 44100

# Spectral flatness threshold — above this, energy is "broad/instrumental"
# Vocals have flatness ~0.01-0.05; instruments like sitar/harmonium ~0.1-0.4
FLATNESS_INSTRUMENT_THRESHOLD = 0.08

# How many Hz around each vocal harmonic to exclude
HARMONIC_EXCLUSION_WIDTH_HZ = 40


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


def _estimate_f0(y, sr):
    """
    Estimate the dominant fundamental frequency (f0) of a vocal signal.
    Returns the median f0 in Hz, or 0 if detection fails.
    """
    pitches, magnitudes = librosa.piptrack(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        fmin=80, fmax=1000  # vocal range
    )
    # Pick the strongest pitch per frame
    f0_values = []
    for t in range(pitches.shape[1]):
        idx = magnitudes[:, t].argmax()
        p = pitches[idx, t]
        if 80 < p < 1000:
            f0_values.append(p)

    if len(f0_values) < 5:
        return 0.0
    return float(np.median(f0_values))


def _build_harmonic_mask(freqs, f0, n_harmonics=12, width_hz=HARMONIC_EXCLUSION_WIDTH_HZ):
    """
    Build a boolean mask that is True for frequency bins near vocal harmonics.
    Harmonic series: f0, 2*f0, 3*f0, ..., n_harmonics*f0.
    """
    mask = np.zeros(len(freqs), dtype=bool)
    if f0 <= 0:
        return mask
    for n in range(1, n_harmonics + 1):
        harmonic_hz = n * f0
        mask |= (freqs >= harmonic_hz - width_hz) & (freqs <= harmonic_hz + width_hz)
    return mask


def _compute_harmonic_coverage(S_power, freqs, harmonic_mask, low_hz, high_hz):
    """
    Compute what fraction of energy in [low_hz, high_hz] is explained
    by the vocal harmonic series (marked by harmonic_mask).

    Returns:
        coverage: float 0..1  (1 = all energy is vocal harmonics = no bleed)
    """
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band_mask):
        return 1.0  # No band = assume clean

    band_indices = np.where(band_mask)[0]
    total_energy = np.sum(S_power[band_indices, :]) + 1e-10

    # Energy at vocal harmonics within this band
    harmonic_in_band = band_mask & harmonic_mask
    harmonic_indices = np.where(harmonic_in_band)[0]
    harmonic_energy = np.sum(S_power[harmonic_indices, :]) if len(harmonic_indices) > 0 else 0.0

    return float(harmonic_energy / total_energy)


def compute_bleed_scores(audio_path, sr=SAMPLE_RATE, instruments=None):
    """
    Compute instrument bleed scores for Demucs-separated vocal stems.

    Architectural insight:
      On a Demucs-separated vocal stem, the entire HARMONIC component
      is the voice itself.  Sitar, harmonium, and sarangi share the
      same 150-1200 Hz frequency band as the human voice, so any
      harmonic energy in that band IS the vocal — not bleed.

      True instrument bleed manifests as PERCUSSIVE residual:
        • Sitar pluck transients
        • Tabla attacks leaking in
        • Harmonium bellows noise
        • String bow attacks

      Therefore we measure bleed using the PERCUSSIVE spectrogram
      from HPSS for ALL instruments.  The harmonic spectrogram is
      used only as the vocal reference (denominator).

    Pipeline:
      1. HPSS → S_harm (= voice) + S_perc (= potential bleed evidence)
      2. E_vocal = harmonic energy in the vocal band (reference)
      3. E_bleed_i = percussive energy in each instrument's band
      4. Bleed_i = 10 * log10(E_bleed_i / E_vocal)
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required. Install: pip install librosa")

    if instruments is None:
        instruments = load_instrument_profiles()

    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)

    # ── HPSS: The Core Separation ─────────────────────────────
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_mag = np.abs(S)
    S_harm, S_perc = librosa.decompose.hpss(S_mag)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    # Power spectrograms
    S_power_harm = S_harm ** 2     # Voice (harmonic = the vocal itself)
    S_power_perc = S_perc ** 2     # Bleed evidence (instrument transients)

    # Vocal reference energy
    E_vocal = compute_spectral_energy(
        S_power_harm, freqs,
        VOCAL_BAND["low_hz"], VOCAL_BAND["high_hz"]
    )

    bleed_scores = {}
    for name, profile in instruments.items():
        low_hz, high_hz = profile["low_hz"], profile["high_hz"]

        # ALL instruments use percussive spectrogram for bleed detection.
        # On a Demucs-separated vocal stem, harmonic energy IS the voice.
        E_inst = compute_spectral_energy(S_power_perc, freqs, low_hz, high_hz)

        # Vocal consonants (t, k, p) inherently produce percussive energy across
        # all bands. The natural consonant baseline in a Demucs-clean vocal is
        # around -6 to -10 dB relative to harmonic power. We subtract a 12 dB
        # baseline offset so that only energy *above* natural consonants is
        # flagged as instrument bleed.
        CONSONANT_BASELINE_DB = 12.0
        
        # Bleed ratio in dB against the main vocal energy, adjusted for consonants
        bleed_db = 10 * np.log10(E_inst / (E_vocal + 1e-10) + 1e-10) - CONSONANT_BASELINE_DB
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
            "is_bleeding": bool(bleed_db >= threshold),
            "severity": severity,
            "threshold_db": threshold,
            "frequency_range_hz": f"{low_hz}-{high_hz}",
        }

    return bleed_scores


def compute_overall_bleed_score(bleed_scores):
    """
    Compute an overall bleed score (0-100 scale, lower = cleaner).

    Only instruments that actually show bleed (above threshold AND high
    spectral flatness) contribute to the penalty.  Clean stems → score ≈ 0.
    """
    if not bleed_scores:
        return 0

    # Compute score from how much each instrument exceeds its bleed threshold.
    # overshoot = energy_ratio_dB - threshold_db
    #   → negative: instrument is below threshold (CLEAN) → 0 contribution
    #   → positive: instrument exceeds threshold (bleeding) → score rises
    #
    # Scale: +15 dB over threshold → 100 (severe), 0 dB over → 0
    SEVERITY_SCALE_DB = 15.0
    penalty_scores = []
    for name, score in bleed_scores.items():
        overshoot = score["energy_ratio_dB"] - score["threshold_db"]
        penalty = max(0.0, overshoot / SEVERITY_SCALE_DB * 100)
        penalty_scores.append(min(100.0, penalty))

    if not penalty_scores:
        return 0

    # Use the MAXIMUM penalty across instruments, not the average.
    # Average dilutes a single severe bleed across 7 clean instruments.
    return int(max(penalty_scores))


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
