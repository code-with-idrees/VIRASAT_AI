#!/usr/bin/env python3
"""
audio_pipeline.py — Combined Output Report Generator
======================================================
Runs the final analysis pipeline (Raag, Taal, Bleed) on enhanced audio tracks
and generates the analysis_reports.json exactly as required for Phase 4.
"""

import argparse
import json
import sys
from pathlib import Path

# Import our modular analysis tools
from raag_classifier import estimate_tonic, classify_raag
from taal_detector import detect_taal
from bleed_detector import compute_bleed_scores, compute_overall_bleed_score


def detect_raag_wrapper(audio_path):
    """Wraps classify_raag to return the exact dict shape needed by the pipeline."""
    import librosa
    import numpy as np
    
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    tonic = estimate_tonic(y, sr)
    matches, pcp = classify_raag(audio_path, tonic_hz=tonic, top_n=3)
    
    # Calculate confidence percentages
    total_sim = sum(abs(m["similarity"]) for m in matches)
    if total_sim > 0:
        for m in matches:
            m["confidence"] = round((m["similarity"] / total_sim) * 100, 1)
    else:
        for m in matches:
            m["confidence"] = 0.0
            
    top_confidence = matches[0]["confidence"] if matches else 0.0
    
    tonic_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    if tonic and tonic > 0:
        tonic_pc = int(round(12 * np.log2(tonic / 261.63))) % 12
        tonic_name = tonic_names[tonic_pc]
    else:
        tonic_name = "Unknown"
        
    return {
        'detected_tonic': tonic_name,
        'top_raags': [(m["raag_name"], m["confidence"]) for m in matches],
        'confidence': top_confidence,
        'is_certain': top_confidence > 60.0
    }


def analyze_heritage_track(audio_path, song_name='Unknown'):
    print(f"\n=== VIRASAT.AI ANALYSIS: {song_name} ===")
    
    # 1. Detect Raag
    print("   🎵 Analyzing Raag...")
    raag_result = detect_raag_wrapper(audio_path)
    
    # 2. Detect Taal
    print("   🥁 Analyzing Taal...")
    taal_result = detect_taal(audio_path)
    
    # 3. Detect Bleed Ratio
    print("   🔬 Analyzing Instrument Bleed...")
    bleed_scores = compute_bleed_scores(audio_path)
    overall_bleed_penalty = compute_overall_bleed_score(bleed_scores)
    bleed_ratio = float(overall_bleed_penalty) / 100.0
    
    # Dynamic restoration score (0-100)
    # Non-linear curve: score = 100 * (1 - bleed_ratio)^0.5
    # This rewards clean stems more aggressively.
    base_score = int(100 * ((1.0 - bleed_ratio) ** 0.5))
    
    # Cleanliness bonus: if no instrument shows bleed, add +2
    any_bleeding = any(s.get("is_bleeding", False) for s in bleed_scores.values())
    if not any_bleeding:
        base_score = min(100, base_score + 2)
    
    if not taal_result.get('tempo_consistent', False):
        base_score = min(base_score, 95)  # Cap at 95 if tempo drifts (not 85)
    
    restoration_score = base_score
        
    report = {
        'song': song_name,
        'raag': raag_result['top_raags'][0][0] if raag_result['top_raags'] else "Unknown",
        'raag_confidence': raag_result['confidence'],
        'tonic': raag_result['detected_tonic'],
        'bpm': taal_result['bpm'],
        'taal': taal_result['likely_taal'],
        'time_signature': taal_result['time_signature'],
        'tempo_drift': not taal_result.get('tempo_consistent', False),
        'bleed_ratio': round(bleed_ratio, 3),
        'restoration_score': restoration_score,
        'ready_for_phase2': restoration_score >= 45,
        'dtw_required': not taal_result.get('tempo_consistent', False),
        'warnings': [w for w in [taal_result.get('warning')] if w],
    }
    
    print("\n📝 Result Summary:")
    print(json.dumps(report, indent=2))
    
    if not report['ready_for_phase2']:
        print("\n>> RECOMMENDATION: This recording needs manual restoration first.")
        print("   Restoration score below 45 — AI cannot reliably fix this track.")
    else:
        print("\n>> READY FOR PHASE 2: Raag-Lock + Taal Quantizer")
        
    return report

def main():
    parser = argparse.ArgumentParser(description="VIRASAT AI — Combined Audio Pipeline Analysis")
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Input directory of enhanced WAVs or exact single WAV file")
    parser.add_argument("--save-json", type=str, default="analysis_reports.json", 
                        help="Output JSON file path (default: analysis_reports.json)")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    reports = {}
    
    if input_path.is_file():
        name = input_path.stem
        reports[name] = analyze_heritage_track(input_path, song_name=name)
    elif input_path.is_dir():
        # Find all WAV files
        wav_files = list(input_path.rglob("*.wav"))
        for wav in wav_files:
            # Skip Demucs output folders if pointed at root (only look for specific files if you want)
            # but usually you point this to /content/enhanced/
            name = wav.stem
            reports[name] = analyze_heritage_track(wav, song_name=name)
    else:
        print(f"❌ Input path {input_path} does not exist.")
        sys.exit(1)
        
    out_json = Path(args.save_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(reports, f, indent=2)
        
    print(f"\n✅ All reports saved to {out_json}")

if __name__ == "__main__":
    main()
