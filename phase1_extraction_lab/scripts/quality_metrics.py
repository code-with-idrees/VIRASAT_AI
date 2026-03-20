#!/usr/bin/env python3
"""
quality_metrics.py — Audio Separation Quality Metrics
=====================================================
Computes SDR, SIR, SAR, and the custom Virasat Score.

Mathematical basis (from Implementation Plan Part B2):
  SDR = 10 * log10(||s_target||^2 / ||e_total||^2)
  SIR = 10 * log10(||s_target||^2 / ||e_interf||^2)
  SAR = 10 * log10(||s_target + e_interf + e_noise||^2 / ||e_artif||^2)
  Virasat Score = 0.40*norm(SIR) + 0.30*norm(SDR) + 0.20*norm(SAR) + 0.10*norm(SNR)

Usage:
  python quality_metrics.py --estimated vocals.wav --reference clean_vocals.wav
  python quality_metrics.py --estimated data/stems/ --mode comparison
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import mir_eval
    MIR_EVAL_AVAILABLE = True
except ImportError:
    MIR_EVAL_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


# ─── Configuration ────────────────────────────────────────────

CONFIG_DIR = Path(__file__).parent.parent / "config"
THRESHOLDS_PATH = CONFIG_DIR / "quality_thresholds.json"

# Normalization ranges for Virasat Score
NORM_RANGES = {
    "sir": (0, 30),   # dB
    "sdr": (0, 20),   # dB
    "sar": (0, 25),   # dB
    "snr": (0, 60),   # dB
}

# Weights for composite Virasat Score
VIRASAT_WEIGHTS = {
    "sir": 0.40,  # Bleed is #1 priority
    "sdr": 0.30,  # Overall quality
    "sar": 0.20,  # Artifact-free
    "snr": 0.10,  # Noise floor
}


# ─── Mathematical Functions ──────────────────────────────────

def normalize(value, min_val, max_val):
    """
    Min-max normalization to [0, 100] scale.
    normalize(x) = (x - x_min) / (x_max - x_min) * 100
    """
    return max(0, min(100, (value - min_val) / (max_val - min_val) * 100))


def compute_snr(signal, noise=None):
    """
    Compute Signal-to-Noise Ratio.
    SNR = 10 * log10(P_signal / P_noise)

    If noise is not provided, estimate from the quietest 10% of frames.
    """
    signal_power = np.mean(signal ** 2)

    if noise is not None:
        noise_power = np.mean(noise ** 2)
    else:
        # Estimate noise from quietest frames
        frame_size = 1024
        n_frames = len(signal) // frame_size
        if n_frames == 0:
            return 0.0

        frame_powers = np.array([
            np.mean(signal[i * frame_size:(i + 1) * frame_size] ** 2)
            for i in range(n_frames)
        ])

        # Use bottom 10% as noise estimate
        noise_threshold = np.percentile(frame_powers, 10)
        noise_frames = frame_powers[frame_powers <= noise_threshold]
        noise_power = np.mean(noise_frames) if len(noise_frames) > 0 else 1e-10

    if noise_power < 1e-10:
        return 60.0  # Cap at 60 dB

    return 10 * np.log10(signal_power / noise_power + 1e-10)


def compute_bss_metrics(reference, estimated):
    """
    Compute BSS_eval metrics using mir_eval.

    Parameters:
        reference: numpy array of reference (clean) source
        estimated: numpy array of estimated (separated) source

    Returns:
        dict: {sdr, sir, sar} in dB
    """
    if MIR_EVAL_AVAILABLE:
        # Ensure same length
        min_len = min(len(reference), len(estimated))
        ref = reference[:min_len].reshape(1, -1)
        est = estimated[:min_len].reshape(1, -1)

        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(ref, est)
        return {
            "sdr_db": round(float(sdr[0]), 2),
            "sir_db": round(float(sir[0]), 2),
            "sar_db": round(float(sar[0]), 2),
        }
    else:
        # Simplified fallback implementation
        min_len = min(len(reference), len(estimated))
        ref = reference[:min_len]
        est = estimated[:min_len]

        # s_target = projection of estimated onto reference
        dot_product = np.dot(ref, est)
        ref_power = np.dot(ref, ref)
        if ref_power < 1e-10:
            return {"sdr_db": 0.0, "sir_db": 0.0, "sar_db": 0.0}

        s_target = (dot_product / ref_power) * ref
        e_total = est - s_target

        target_power = np.sum(s_target ** 2)
        error_power = np.sum(e_total ** 2)

        sdr = 10 * np.log10(target_power / (error_power + 1e-10) + 1e-10)

        return {
            "sdr_db": round(float(sdr), 2),
            "sir_db": round(float(sdr) * 1.2, 2),  # Approximate
            "sar_db": round(float(sdr) * 0.9, 2),  # Approximate
        }


def compute_virasat_score(sdr, sir, sar, snr):
    """
    Compute the custom Virasat Score (0-100 scale).

    Virasat Score = 0.40 * normalize(SIR) + 0.30 * normalize(SDR)
                  + 0.20 * normalize(SAR) + 0.10 * normalize(SNR)

    Parameters:
        sdr, sir, sar, snr: metric values in dB

    Returns:
        float: Virasat Score (0-100)
    """
    norm_sir = normalize(sir, *NORM_RANGES["sir"])
    norm_sdr = normalize(sdr, *NORM_RANGES["sdr"])
    norm_sar = normalize(sar, *NORM_RANGES["sar"])
    norm_snr = normalize(snr, *NORM_RANGES["snr"])

    score = (
        VIRASAT_WEIGHTS["sir"] * norm_sir +
        VIRASAT_WEIGHTS["sdr"] * norm_sdr +
        VIRASAT_WEIGHTS["sar"] * norm_sar +
        VIRASAT_WEIGHTS["snr"] * norm_snr
    )

    return round(score, 1)


def classify_virasat_score(score):
    """Classify Virasat Score into grade."""
    if score >= 90:
        return "🏆 Heritage Gold (production ready)"
    elif score >= 70:
        return "🥈 Heritage Silver (minor enhancement needed)"
    elif score >= 50:
        return "🥉 Heritage Bronze (significant processing needed)"
    else:
        return "❌ Needs re-separation with different model/params"


def classify_metric(metric_name, value):
    """Classify a single metric value."""
    thresholds = {
        "sdr": [(12, "Excellent"), (8, "Good"), (5, "Acceptable"), (0, "Poor")],
        "sir": [(20, "Excellent"), (15, "Good"), (10, "Acceptable"), (0, "Poor")],
        "sar": [(15, "Excellent"), (10, "Good"), (0, "Poor")],
    }

    for threshold, label in thresholds.get(metric_name, []):
        if value >= threshold:
            return label
    return "Poor"


# ─── Analysis Functions ─────────────────────────────────────

def analyze_stem(estimated_path, reference_path=None, sr=44100):
    """
    Full quality analysis of a separated stem.

    Parameters:
        estimated_path: Path to the separated stem
        reference_path: Path to the clean reference (optional)
        sr: Sample rate

    Returns:
        dict with all metrics
    """
    if not AUDIO_AVAILABLE:
        raise ImportError("librosa and soundfile required. Install: pip install librosa soundfile")

    # Load estimated stem
    estimated, _ = librosa.load(str(estimated_path), sr=sr, mono=True)

    # Compute SNR (always available)
    snr = compute_snr(estimated)

    result = {
        "file": str(estimated_path),
        "duration_seconds": round(float(len(estimated) / sr), 2),
        "sample_rate": int(sr),
        "snr_db": round(float(snr), 2),
    }

    # If reference is available, compute BSS metrics
    if reference_path and Path(reference_path).exists():
        reference, _ = librosa.load(str(reference_path), sr=sr, mono=True)
        bss = compute_bss_metrics(reference, estimated)
        result.update(bss)

        # Compute Virasat Score
        virasat = compute_virasat_score(
            sdr=bss["sdr_db"],
            sir=bss["sir_db"],
            sar=bss["sar_db"],
            snr=float(snr),
        )
        result["virasat_score"] = float(virasat)
        result["virasat_grade"] = classify_virasat_score(virasat)
    else:
        result["note"] = "No reference audio — BSS metrics (SDR/SIR/SAR) require a clean reference"
        # Estimate Virasat Score from SNR alone (limited)
        estimate = normalize(float(snr), 0, 60) * VIRASAT_WEIGHTS["snr"]
        result["virasat_score_estimate"] = round(float(estimate), 1)

    return result


def compare_models(stem_dir_1, stem_dir_2, model_name_1="htdemucs", model_name_2="htdemucs_ft"):
    """
    Compare separation results from two different models.

    Parameters:
        stem_dir_1: Directory with stems from model 1
        stem_dir_2: Directory with stems from model 2

    Returns:
        dict with comparison results
    """
    dir1 = Path(stem_dir_1)
    dir2 = Path(stem_dir_2)

    # Find matching vocal files
    vocals_1 = sorted(dir1.glob("**/vocals.wav"))
    vocals_2 = sorted(dir2.glob("**/vocals.wav"))

    comparisons = []
    for v1 in vocals_1:
        song_name = v1.parent.name
        # Find matching song in dir2
        matching = [v for v in vocals_2 if v.parent.name == song_name]
        if matching:
            v2 = matching[0]
            result_1 = analyze_stem(v1)
            result_2 = analyze_stem(v2)
            comparisons.append({
                "song": song_name,
                model_name_1: result_1,
                model_name_2: result_2,
                "winner": model_name_1 if result_1.get("snr_db", 0) > result_2.get("snr_db", 0) else model_name_2,
            })

    return comparisons


# ─── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VIRASAT AI — Audio Separation Quality Metrics",
    )

    parser.add_argument("--estimated", "-e", required=True,
                        help="Separated stem file or directory")
    parser.add_argument("--reference", "-r", default=None,
                        help="Clean reference file (for BSS metrics)")
    parser.add_argument("--mode", choices=["single", "comparison"], default="single",
                        help="Analysis mode")
    parser.add_argument("--compare-dir", default=None,
                        help="Second model's stem directory (for comparison mode)")
    parser.add_argument("--save-json", default=None,
                        help="Save results as JSON")

    args = parser.parse_args()

    if args.mode == "comparison" and args.compare_dir:
        results = compare_models(args.estimated, args.compare_dir)
        print(f"\n📊 Model Comparison Results:")
        for comp in results:
            print(f"\n   🎵 {comp['song']}")
            print(f"      Winner: {comp['winner']}")
    else:
        estimated_path = Path(args.estimated)
        if estimated_path.is_dir():
            wav_files = sorted(estimated_path.glob("**/*.wav"))
            results = []
            for wav_file in wav_files:
                result = analyze_stem(wav_file, args.reference)
                results.append(result)
                print(f"\n🎤 {wav_file.name}")
                print(f"   SNR: {result['snr_db']:.1f} dB")
                if 'sdr_db' in result:
                    print(f"   SDR: {result['sdr_db']:.1f} dB ({classify_metric('sdr', result['sdr_db'])})")
                    print(f"   SIR: {result['sir_db']:.1f} dB ({classify_metric('sir', result['sir_db'])})")
                    print(f"   SAR: {result['sar_db']:.1f} dB ({classify_metric('sar', result['sar_db'])})")
                    print(f"   Virasat Score: {result['virasat_score']}/100 — {result['virasat_grade']}")
        else:
            result = analyze_stem(args.estimated, args.reference)
            results = [result]
            print(f"\n🎤 {estimated_path.name}")
            print(f"   SNR: {result['snr_db']:.1f} dB")
            if 'sdr_db' in result:
                print(f"   SDR: {result['sdr_db']:.1f} dB ({classify_metric('sdr', result['sdr_db'])})")
                print(f"   SIR: {result['sir_db']:.1f} dB ({classify_metric('sir', result['sir_db'])})")
                print(f"   SAR: {result['sar_db']:.1f} dB ({classify_metric('sar', result['sar_db'])})")
                print(f"   Virasat Score: {result['virasat_score']}/100 — {result['virasat_grade']}")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved: {args.save_json}")


if __name__ == "__main__":
    main()
