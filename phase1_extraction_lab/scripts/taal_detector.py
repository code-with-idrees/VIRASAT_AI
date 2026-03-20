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

# Taal definitions mapping from time signature
TAAL_MAP = {
    '4/4': 'Keherwa (8 beats) or Teentaal (16 beats)',
    '6/8': 'Dadra (6 beats)',
    '7/8': 'Rupak (7 beats)',
    '10/8': 'Jhaptaal (10 beats)',
}

def detect_taal(audio_path, sr=22050):
    '''
    Detects BPM and maps to the most likely Taal.
    Eastern music does not always follow strict tempo — we use confidence scoring.
    '''
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa required")

    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    
    # 1. Detect tempo (BPM)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
    bpm = float(tempo[0] if hasattr(tempo, '__len__') else tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # 2. Analyze beat intervals for consistency
    if len(beat_times) > 1:
        intervals = np.diff(beat_times)
        interval_std = float(np.std(intervals))
        is_consistent = interval_std < 0.15  # Less than 150ms variance = consistent tempo
    else:
        is_consistent = False
        interval_std = 999.0
    
    # 3. Detect time signature (look for 3-beat vs 4-beat groupings)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # Pad to ensure enough length for correlation pattern
    if len(onset_env) < 20:
        pad = np.zeros(20 - len(onset_env))
        onset_env = np.concatenate([onset_env, pad])
        
    # Check correlation with beat patterns
    score_3 = float(np.correlate(onset_env, np.array([1,0,0,1,0,0,1,0,0]), mode='valid')[0])
    score_4 = float(np.correlate(onset_env, np.array([1,0,0,0,1,0,0,0]), mode='valid')[0])
    score_6 = float(np.correlate(onset_env, np.array([1,0,0,1,0,0]), mode='valid')[0])
    score_7 = float(np.correlate(onset_env, np.array([1,0,0,1,0,1,0]), mode='valid')[0])
    
    # 4. Map to Taal
    pattern_scores = {'4/4': score_4, '6/8': score_6, '7/8': score_7}
    detected_sig = max(pattern_scores, key=pattern_scores.get)
    
    return {
        'bpm': round(bpm, 1),
        'time_signature': detected_sig,
        'likely_taal': TAAL_MAP.get(detected_sig, 'Unknown'),
        'tempo_consistent': bool(is_consistent),
        'tempo_variance_ms': round(float(interval_std) * 1000, 1),
        'warning': None if is_consistent else 'Tempo drifts — DTW sync required in Phase 2'
    }


def main():
    parser = argparse.ArgumentParser(description="VIRASAT AI — Taal Detector")
    parser.add_argument("--input", "-i", required=True, help="Audio file")
    parser.add_argument("--save-json", default=None, help="Save results")
    args = parser.parse_args()

    result = detect_taal(args.input)

    print(f"\n🥁 Taal Analysis: {Path(args.input).name}")
    print(f"   BPM: {result['bpm']} BPM")
    print(f"   Time Signature: {result['time_signature']}")
    print(f"   Likely Taal: {result['likely_taal']}")
    print(f"   Tempo Consistent: {result['tempo_consistent']} (variance: {result['tempo_variance_ms']} ms)")
    
    if result.get("warning"):
        print(f"   ⚠️  WARNING: {result['warning']}")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
