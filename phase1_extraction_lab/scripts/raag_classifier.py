#!/usr/bin/env python3
"""
raag_classifier.py — Raag Detection via Pitch Class Profile
============================================================
Detects which Raag a recording is in using Pitch Class Profiles
and cosine similarity matching.

Mathematical basis (from Implementation Plan Part B4):
  1. Compute STFT → magnitude spectrogram
  2. Map frequencies to 12 pitch classes: pc(f) = round(12*log2(f/f_tonic)) mod 12
  3. Sum energy per pitch class → Pitch Class Profile (PCP)
  4. Match against Raag templates: Match = cosine_similarity(PCP, Template)

Usage:
  python raag_classifier.py --input vocal_stem.wav
  python raag_classifier.py --input data/stems/ --top 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# ─── Configuration ────────────────────────────────────────────

CONFIG_DIR = Path(__file__).parent.parent / "config"
RAAG_MAPS_PATH = CONFIG_DIR / "raag_maps.json"

N_FFT = 4096  # Higher resolution for pitch detection
HOP_LENGTH = 512
SAMPLE_RATE = 44100

# Default tonic (Sa = C4)
DEFAULT_TONIC_HZ = 261.63


# ─── Mathematical Functions ──────────────────────────────────

def frequency_to_pitch_class(freq, tonic_hz=DEFAULT_TONIC_HZ):
    """
    Map a frequency to its pitch class (0-11).

    pitch_class(f) = round(12 * log2(f / f_tonic)) mod 12

    Parameters:
        freq:     Frequency in Hz
        tonic_hz: Tonic frequency (Sa)

    Returns:
        int: Pitch class (0=Sa, 2=Re, 4=Ga, 5=Ma, 7=Pa, 9=Dha, 11=Ni)
    """
    if freq <= 0 or tonic_hz <= 0:
        return -1
    return int(round(12 * np.log2(freq / tonic_hz))) % 12


def compute_pitch_class_profile(y, sr=SAMPLE_RATE, tonic_hz=DEFAULT_TONIC_HZ):
    """
    Compute the Pitch Class Profile (PCP) of an audio signal.

    PCP[k] = Σ |S(ω)|² for all ω mapping to pitch class k
             across all frames

    Parameters:
        y:        Audio signal
        sr:       Sample rate
        tonic_hz: Tonic frequency

    Returns:
        numpy array of shape (12,) — normalized PCP
    """
    # Compute STFT
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_power = np.abs(S) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    # Initialize PCP
    pcp = np.zeros(12)

    # Map each frequency bin to pitch class and accumulate energy
    for i, freq in enumerate(freqs):
        if freq < 50 or freq > 8000:  # Skip very low/high frequencies
            continue
        pc = frequency_to_pitch_class(freq, tonic_hz)
        if 0 <= pc < 12:
            pcp[pc] += np.sum(S_power[i, :])

    # Normalize
    total = np.sum(pcp)
    if total > 0:
        pcp = pcp / total

    return pcp


def compute_chroma_pcp(y, sr=SAMPLE_RATE):
    """
    Alternative PCP using librosa's chroma features.
    More robust than manual computation for polyphonic audio.

    Returns:
        numpy array of shape (12,) — normalized PCP
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    pcp = np.mean(chroma, axis=1)  # Average across frames
    pcp = pcp / (np.sum(pcp) + 1e-10)
    return pcp


def cosine_similarity(a, b):
    """
    Cosine similarity between two vectors.

    sim(a, b) = (a · b) / (||a|| × ||b||)

    Returns:
        float: similarity score (0 to 1)
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return float(dot / (norm_a * norm_b))


def build_raag_template(pitch_classes):
    """
    Build a 12-dimensional template vector from Raag pitch classes.

    Template[k] = 1 if k ∈ Raag, else 0

    Parameters:
        pitch_classes: List of allowed pitch class integers (0-11)

    Returns:
        numpy array of shape (12,)
    """
    template = np.zeros(12)
    for pc in pitch_classes:
        if 0 <= pc < 12:
            template[pc] = 1.0
    # Normalize
    template = template / (np.sum(template) + 1e-10)
    return template


def load_raag_database():
    """Load Raag definitions from config."""
    if RAAG_MAPS_PATH.exists():
        with open(RAAG_MAPS_PATH) as f:
            data = json.load(f)
        return data.get("raags", {})
    else:
        # Fallback minimal set
        return {
            "yaman": {"pitch_classes": [0, 2, 4, 6, 7, 9, 11], "name": "Raag Yaman"},
            "bhairavi": {"pitch_classes": [0, 1, 3, 5, 7, 8, 10], "name": "Raag Bhairavi"},
            "kafi": {"pitch_classes": [0, 2, 3, 5, 7, 9, 10], "name": "Raag Kafi"},
        }


# ─── Classification ─────────────────────────────────────────

def classify_raag(audio_path, tonic_hz=DEFAULT_TONIC_HZ, top_n=3, use_chroma=True):
    """
    Classify which Raag an audio recording is in.

    Steps:
      1. Compute PCP of the audio
      2. Build templates for all known Raags
      3. Compute cosine similarity between PCP and each template
      4. Return top matches

    Parameters:
        audio_path: Path to audio file
        tonic_hz:   Tonic frequency (Sa)
        top_n:      Number of top matches to return
        use_chroma: Use librosa chroma features (more robust)

    Returns:
        list of {raag_name, similarity, pitch_classes, ...}
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa required. Install: pip install librosa")

    # Load audio
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

    # Compute PCP
    if use_chroma:
        pcp = compute_chroma_pcp(y, sr)
    else:
        pcp = compute_pitch_class_profile(y, sr, tonic_hz)

    # Load Raag database
    raags = load_raag_database()

    # Match against all Raags
    matches = []
    for raag_id, raag_info in raags.items():
        template = build_raag_template(raag_info["pitch_classes"])
        similarity = cosine_similarity(pcp, template)

        matches.append({
            "raag_id": raag_id,
            "raag_name": raag_info.get("name", raag_id),
            "similarity": round(similarity, 4),
            "pitch_classes": raag_info["pitch_classes"],
            "mood": raag_info.get("mood", "Unknown"),
            "time": raag_info.get("time", "Unknown"),
        })

    # Sort by similarity (highest first)
    matches.sort(key=lambda x: x["similarity"], reverse=True)

    return matches[:top_n], pcp


def estimate_tonic(y, sr=SAMPLE_RATE):
    """
    Estimate the tonic (Sa) frequency of a recording.
    Uses pitch tracking and finds the most stable pitch.

    Returns:
        float: Estimated tonic frequency in Hz
    """
    # Use librosa pitch tracking
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Find the most common strong pitch
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 50 and pitch < 1000:  # Reasonable vocal range
            pitch_values.append(pitch)

    if not pitch_values:
        return DEFAULT_TONIC_HZ

    # Find mode using histogram
    hist, bin_edges = np.histogram(pitch_values, bins=100)
    peak_bin = np.argmax(hist)
    estimated_tonic = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2

    return round(float(estimated_tonic), 2)


# ─── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VIRASAT AI — Raag Classifier (Pitch Class Profile)",
    )

    parser.add_argument("--input", "-i", required=True, help="Audio file to classify")
    parser.add_argument("--top", type=int, default=3, help="Top N Raag matches to show")
    parser.add_argument("--tonic", type=float, default=None, help="Tonic frequency (Hz)")
    parser.add_argument("--auto-tonic", action="store_true", help="Auto-detect tonic")
    parser.add_argument("--save-json", default=None, help="Save results as JSON")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Not found: {args.input}")
        return

    # Auto-detect tonic if requested
    tonic = args.tonic or DEFAULT_TONIC_HZ
    if args.auto_tonic:
        print("🎵 Detecting tonic...")
        y, sr = librosa.load(str(input_path), sr=SAMPLE_RATE, mono=True)
        tonic = estimate_tonic(y, sr)
        print(f"   Estimated tonic: {tonic:.1f} Hz")

    # Classify
    print(f"\n🔍 Classifying: {input_path.name}")
    print(f"   Tonic: {tonic:.1f} Hz")

    matches, pcp = classify_raag(input_path, tonic_hz=tonic, top_n=args.top)

    print(f"\n📊 Pitch Class Profile (0=Sa ... 11=Ni):")
    note_names = ["Sa", "Re♭", "Re", "Ga♭", "Ga", "Ma", "Ma#", "Pa", "Dha♭", "Dha", "Ni♭", "Ni"]
    for i, (name, val) in enumerate(zip(note_names, pcp)):
        bar = "█" * int(val * 50)
        print(f"   {name:4s} │ {val:.3f} {bar}")

    print(f"\n🎶 Top {args.top} Raag Matches:")
    for i, match in enumerate(matches):
        confidence = "HIGH" if match["similarity"] > 0.85 else "MEDIUM" if match["similarity"] > 0.70 else "LOW"
        print(f"\n   {i+1}. {match['raag_name']}")
        print(f"      Similarity: {match['similarity']:.4f} ({confidence})")
        print(f"      Mood: {match['mood']}")
        print(f"      Time: {match['time']}")

    if args.save_json:
        # Calculate relative confidence percentages for the top matches
        total_sim = sum(abs(m["similarity"]) for m in matches)
        if total_sim > 0:
            for m in matches:
                m["confidence"] = round((m["similarity"] / total_sim) * 100, 1)
        else:
            for m in matches:
                m["confidence"] = 0.0

        top_confidence = matches[0]["confidence"] if matches else 0.0

        # Create output matching the blueprint's detect_raag format
        tonic_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        # Find closest standard pitch class index for the raw frequency
        # Formula: pc = round(12 * log2(f / 261.63)) % 12
        if tonic and tonic > 0:
            tonic_pc = int(round(12 * np.log2(tonic / 261.63))) % 12
            tonic_name = tonic_names[tonic_pc]
        else:
            tonic_name = "Unknown"

        result = {
            "file": str(input_path),
            "detected_tonic": tonic_name,
            "tonic_hz": tonic,
            "top_raags": [(m["raag_name"], m["confidence"]) for m in matches],
            "confidence": top_confidence,
            "is_certain": top_confidence > 60.0,
            "chroma_vector": pcp.tolist(),
            "matches_detailed": matches,
        }
        with open(args.save_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved: {args.save_json}")


if __name__ == "__main__":
    main()
