#!/usr/bin/env python3
"""
stem_separator.py — Demucs v4 Stem Separation Engine
=====================================================
Wrapper around Facebook's Demucs v4 for audio source separation.
Supports both htdemucs and htdemucs_ft models.

Mathematical basis:
  Hybrid Transformer Demucs uses dual U-Net architecture:
  - Time-domain branch: temporal convolutions on raw waveform
  - Frequency-domain branch: convolutions on STFT spectrogram
  - Cross-domain Transformer encoder with self-attention and cross-attention

Usage:
  python stem_separator.py --input data/raw/ --output data/stems/ --model htdemucs_ft
  python stem_separator.py --input data/raw/song.mp3 --model htdemucs --two-stems
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ─── Configuration ────────────────────────────────────────────

SUPPORTED_MODELS = {
    "htdemucs": {
        "description": "Hybrid Transformer Demucs — general purpose",
        "best_for": "Modern recordings (post-2000)",
        "speed": "Faster",
        "benchmark_sdr_db": 7.5,
    },
    "htdemucs_ft": {
        "description": "Hybrid Transformer Demucs — fine-tuned",
        "best_for": "Old/mono recordings (pre-2000)",
        "speed": "Slower (2x htdemucs)",
        "benchmark_sdr_db": 9.2,
    },
}

SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


# ─── Core Functions ──────────────────────────────────────────

def detect_device():
    """Detect GPU/CPU and return device info."""
    device_info = {
        "device": "cpu",
        "device_name": "CPU",
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory_gb": None,
    }

    if TORCH_AVAILABLE and torch.cuda.is_available():
        device_info["device"] = "cuda"
        device_info["cuda_available"] = True
        device_info["gpu_name"] = torch.cuda.get_device_name(0)
        device_info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
        )
        device_info["device_name"] = f"GPU: {device_info['gpu_name']} ({device_info['gpu_memory_gb']} GB)"

    return device_info


def find_audio_files(input_path):
    """Find all supported audio files in a directory or return single file."""
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            return [input_path]
        else:
            print(f"⚠️  Unsupported format: {input_path.suffix}")
            return []

    if input_path.is_dir():
        files = []
        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            # rglob = recursive: finds WAVs in subdirs (e.g. data/raw/Noor_Jehan/song.wav)
            files.extend(input_path.rglob(f"*{ext}"))
        files = sorted(set(files))  # deduplicate
        if files:
            print(f"\n📂 Found {len(files)} audio file(s) in {input_path}:")
            for f in files:
                print(f"   • {f.name}  ({f.parent.name}/)")
        return files

    print(f"❌ Path not found: {input_path}")
    return []


def run_demucs(
    input_file,
    output_dir,
    model="htdemucs_ft",
    two_stems=True,
    device=None,
    shifts=1,
    overlap=0.25,
):
    """
    Run Demucs v4 on a single audio file.

    Parameters:
        input_file:  Path to input audio file
        output_dir:  Directory for output stems
        model:       Demucs model name (htdemucs or htdemucs_ft)
        two_stems:   If True, output only vocals + no_vocals (faster, cleaner)
        device:      Force device (cuda/cpu), auto-detect if None
        shifts:      Number of random shifts for better quality (1=fast, 5=best)
        overlap:     Overlap between processing segments (0.25 default)

    Returns:
        dict with results: {success, output_path, duration_seconds, model_used}
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)

    if model not in SUPPORTED_MODELS:
        print(f"❌ Unknown model: {model}. Use: {list(SUPPORTED_MODELS.keys())}")
        return {"success": False, "error": f"Unknown model: {model}"}

    # Build demucs command
    cmd = [
        sys.executable, "-m", "demucs",
        "--name", model,
        "--out", str(output_dir),
        "--shifts", str(shifts),
        "--overlap", str(overlap),
    ]

    if two_stems:
        cmd.extend(["--two-stems", "vocals"])

    if device:
        cmd.extend(["--device", device])

    cmd.append(str(input_file))

    # Run separation
    print(f"\n🎵 Processing: {input_file.name}")
    print(f"   Model:  {model} ({SUPPORTED_MODELS[model]['description']})")
    print(f"   Mode:   {'Two-stems (vocals/no_vocals)' if two_stems else 'Full 4-stem'}")
    print(f"   Shifts: {shifts} | Overlap: {overlap}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        duration = round(time.time() - start_time, 1)

        if result.returncode == 0:
            # Find output directory
            stem_dir = output_dir / model / input_file.stem
            print(f"   ✅ Complete in {duration}s → {stem_dir}")

            return {
                "success": True,
                "output_path": str(stem_dir),
                "input_file": str(input_file),
                "model": model,
                "duration_seconds": duration,
                "two_stems": two_stems,
            }
        else:
            print(f"   ❌ Error: {result.stderr[:200]}")
            return {
                "success": False,
                "error": result.stderr[:500],
                "duration_seconds": duration,
            }

    except subprocess.TimeoutExpired:
        print("   ❌ Timeout (>10 min)")
        return {"success": False, "error": "Timeout exceeded 10 minutes"}
    except FileNotFoundError:
        print("   ❌ Demucs not installed. Run: pip install demucs")
        return {"success": False, "error": "Demucs not installed"}


def batch_separate(
    input_dir,
    output_dir,
    models=None,
    two_stems=True,
    device=None,
):
    """
    Batch process all audio files with one or more models.
    When models list has 2+ entries, runs comparison mode (same songs, different models).

    Returns:
        list of result dicts
    """
    if models is None:
        models = ["htdemucs_ft"]

    audio_files = find_audio_files(input_dir)
    if not audio_files:
        print("❌ No audio files found.")
        return []

    print(f"\n{'='*60}")
    print(f"🎼 VIRASAT AI — Batch Stem Separation")
    print(f"{'='*60}")
    print(f"   Files:  {len(audio_files)}")
    print(f"   Models: {', '.join(models)}")
    print(f"   Device: {detect_device()['device_name']}")
    print(f"{'='*60}\n")

    all_results = []

    for audio_file in audio_files:
        for model in models:
            result = run_demucs(
                input_file=audio_file,
                output_dir=output_dir,
                model=model,
                two_stems=two_stems,
                device=device,
            )
            result["audio_file"] = audio_file.name
            all_results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 Summary")
    print(f"{'='*60}")
    success_count = sum(1 for r in all_results if r.get("success"))
    print(f"   Total:     {len(all_results)}")
    print(f"   Success:   {success_count}")
    print(f"   Failed:    {len(all_results) - success_count}")

    return all_results


def save_results(results, output_path):
    """Save batch results to JSON for quality analysis pipeline."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings for JSON serialization
    serializable = []
    for r in results:
        sr = {}
        for k, v in r.items():
            sr[k] = str(v) if isinstance(v, Path) else v
        serializable.append(sr)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n💾 Results saved: {output_path}")


# ─── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VIRASAT AI — Demucs v4 Stem Separation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Separate a single file with fine-tuned model
  python stem_separator.py --input song.mp3 --model htdemucs_ft

  # Batch process with both models for comparison
  python stem_separator.py --input data/raw/ --models htdemucs htdemucs_ft

  # Full 4-stem separation (vocals, drums, bass, other)
  python stem_separator.py --input song.mp3 --no-two-stems
        """,
    )

    parser.add_argument("--input", "-i", required=True, help="Input audio file or directory")
    parser.add_argument("--output", "-o", default="phase1_extraction_lab/data/stems",
                        help="Output directory for stems")
    parser.add_argument("--model", "-m", default="htdemucs_ft",
                        choices=list(SUPPORTED_MODELS.keys()),
                        help="Demucs model to use")
    parser.add_argument("--models", nargs="+",
                        choices=list(SUPPORTED_MODELS.keys()),
                        help="Multiple models for comparison mode")
    parser.add_argument("--no-two-stems", action="store_true",
                        help="Full 4-stem separation instead of vocals-only")
    parser.add_argument("--device", choices=["cuda", "cpu"],
                        help="Force device (auto-detect if not set)")
    parser.add_argument("--shifts", type=int, default=1,
                        help="Random shifts for quality (1=fast, 5=best)")
    parser.add_argument("--overlap", type=float, default=0.25,
                        help="Segment overlap (0.0-0.5)")
    parser.add_argument("--save-results", default=None,
                        help="Path to save results JSON")
    parser.add_argument("--info", action="store_true",
                        help="Show device and model info, then exit")

    args = parser.parse_args()

    # Info mode
    if args.info:
        device = detect_device()
        print(f"\n🖥️  Device: {device['device_name']}")
        print(f"\n📦 Available Models:")
        for name, info in SUPPORTED_MODELS.items():
            print(f"   {name}: {info['description']}")
            print(f"      Best for: {info['best_for']}")
            print(f"      Benchmark SDR: {info['benchmark_sdr_db']} dB")
        return

    # Determine models
    models = args.models if args.models else [args.model]
    two_stems = not args.no_two_stems

    # Check input exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input not found: {args.input}")
        sys.exit(1)

    # Run separation
    if input_path.is_dir() or args.models:
        results = batch_separate(
            input_dir=args.input,
            output_dir=args.output,
            models=models,
            two_stems=two_stems,
            device=args.device,
        )
    else:
        result = run_demucs(
            input_file=args.input,
            output_dir=args.output,
            model=models[0],
            two_stems=two_stems,
            device=args.device,
            shifts=args.shifts,
            overlap=args.overlap,
        )
        results = [result]

    # Save results
    if args.save_results:
        save_results(results, args.save_results)
    elif results:
        default_path = Path(args.output).parent / "reports" / "separation_results.json"
        save_results(results, default_path)


if __name__ == "__main__":
    main()
