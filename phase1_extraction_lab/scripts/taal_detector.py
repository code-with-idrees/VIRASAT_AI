#!/usr/bin/env python3
"""
taal_detector.py — Rhythmic Cycle (Taal) Detection
====================================================
Detects taal (rhythmic cycle) patterns in audio using
onset detection and beat tracking.

Usage:
  python taal_detector.py --input song.wav
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


# ─── Common Taal Patterns ─────────────────────────────────────

TAAL_DATABASE = {
    "teentaal": {
        "name": "Teentaal",
        "beats": 16,
        "divisions": [4, 4, 4, 4],
        "clap_pattern": ["Dha", "Dhin", "Dhin", "Dha",
                          "Dha", "Dhin", "Dhin", "Dha",
                          "Dha", "Tin", "Tin", "Ta",
                          "Ta", "Dhin", "Dhin", "Dha"],
        "common_in": ["Ghazal", "Classical", "Film songs"],
        "tempo_range_bpm": [50, 200],
    },
    "rupak": {
        "name": "Rupak Taal",
        "beats": 7,
        "divisions": [3, 2, 2],
        "common_in": ["Light classical", "Thumri"],
        "tempo_range_bpm": [40, 160],
    },
    "dadra": {
        "name": "Dadra",
        "beats": 6,
        "divisions": [3, 3],
        "common_in": ["Thumri", "Folk", "Film songs"],
        "tempo_range_bpm": [60, 180],
    },
    "keherwa": {
        "name": "Keherwa",
        "beats": 8,
        "divisions": [4, 4],
        "common_in": ["Folk", "Qawwali", "Light classical"],
        "tempo_range_bpm": [80, 220],
    },
    "jhaptaal": {
        "name": "Jhaptaal",
        "beats": 10,
        "divisions": [2, 3, 2, 3],
        "common_in": ["Classical vocal", "Ghazal"],
        "tempo_range_bpm": [40, 160],
    },
}


def detect_tempo_and_beats(audio_path, sr=44100):
    """Detect tempo and beat positions."""
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa required")

    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Onset strength for rhythm analysis
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    return {
        "tempo_bpm": round(float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo), 1),
        "num_beats": len(beat_times),
        "beat_times": beat_times.tolist(),
        "duration_seconds": round(len(y) / sr, 2),
    }


def classify_taal(tempo_bpm, num_beats, duration):
    """Simple taal classification based on tempo and beat grouping."""
    matches = []
    for taal_id, taal in TAAL_DATABASE.items():
        tempo_range = taal["tempo_range_bpm"]
        if tempo_range[0] <= tempo_bpm <= tempo_range[1]:
            # Check if beat count is divisible by taal beats
            remainder = num_beats % taal["beats"]
            fit_score = 1.0 - (remainder / taal["beats"])
            matches.append({
                "taal_id": taal_id,
                "taal_name": taal["name"],
                "beats": taal["beats"],
                "fit_score": round(fit_score, 3),
                "common_in": taal.get("common_in", []),
            })

    matches.sort(key=lambda x: x["fit_score"], reverse=True)
    return matches[:3]


def main():
    parser = argparse.ArgumentParser(description="VIRASAT AI — Taal Detector")
    parser.add_argument("--input", "-i", required=True, help="Audio file")
    parser.add_argument("--save-json", default=None, help="Save results")
    args = parser.parse_args()

    result = detect_tempo_and_beats(args.input)
    taal_matches = classify_taal(result["tempo_bpm"], result["num_beats"], result["duration_seconds"])

    print(f"\n🥁 Taal Analysis: {Path(args.input).name}")
    print(f"   Tempo: {result['tempo_bpm']} BPM")
    print(f"   Beats: {result['num_beats']}")

    for i, match in enumerate(taal_matches):
        print(f"\n   {i+1}. {match['taal_name']} ({match['beats']} beats)")
        print(f"      Fit: {match['fit_score']:.3f}")
        print(f"      Common in: {', '.join(match['common_in'])}")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump({"rhythm": result, "taal_matches": taal_matches}, f, indent=2)


if __name__ == "__main__":
    main()
