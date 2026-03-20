#!/usr/bin/env python3
"""
run_pipeline.py — VIRASAT AI End-to-End Phase 1 Pipeline
=========================================================
Standalone script to run the full Phase 1 pipeline:
  1. (Optional) Download songs from test_songs.json
  2. Run Demucs v4 stem separation
  3. Compute quality metrics (SNR / Virasat Score estimate)
  4. Run instrument bleed detection
  5. Print a final summary report

Works on both:
  - LOCAL CPU  → python run_pipeline.py --skip-download
  - CLOUD GPU  → python run_pipeline.py (Kaggle / Colab)

Usage:
  # Full pipeline (download + separate + analyze)
  python phase1_extraction_lab/scripts/run_pipeline.py

  # Skip download (songs already in data/raw/)
  python phase1_extraction_lab/scripts/run_pipeline.py --skip-download

  # Run on a single already-downloaded WAV file
  python phase1_extraction_lab/scripts/run_pipeline.py \\
      --skip-download \\
      --single-file phase1_extraction_lab/data/raw/Vital_Signs_Dil_Dil_Pakistan.wav

  # Use faster model (good for testing on CPU)
  python phase1_extraction_lab/scripts/run_pipeline.py \\
      --skip-download --model htdemucs
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Ensure project root is importable ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR      = PROJECT_ROOT / "phase1_extraction_lab" / "data" / "raw"
STEMS_DIR    = PROJECT_ROOT / "phase1_extraction_lab" / "data" / "stems"
REPORTS_DIR  = PROJECT_ROOT / "phase1_extraction_lab" / "data" / "reports"
CONFIG_PATH  = PROJECT_ROOT / "phase1_extraction_lab" / "config" / "test_songs.json"

SCRIPTS_DIR  = PROJECT_ROOT / "phase1_extraction_lab" / "scripts"


# ─── Utilities ────────────────────────────────────────────────────────────────

def banner(title: str) -> None:
    """Print a section banner."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def detect_device() -> str:
    """Return 'cuda' if GPU available, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            print(f"   🚀 GPU detected: {gpu} ({mem} GB)")
            return "cuda"
    except ImportError:
        pass
    print("   💻 No GPU detected — running on CPU (Demucs will be slower)")
    return "cpu"


# ─── Pipeline Steps ───────────────────────────────────────────────────────────

def step_download(config_path: Path) -> list:
    """Step 1: Download songs using download_songs.py functions."""
    banner("STEP 1 ▸ Download Songs")
    from phase1_extraction_lab.scripts.download_songs import download_from_config
    results = download_from_config(config_path)
    success = sum(1 for r in results if r.get("success"))
    print(f"\n   ✅ {success}/{len(results)} songs downloaded to {RAW_DIR}")
    return results


def step_separate(
    input_dir: Path,
    output_dir: Path,
    model: str,
    device: str,
    single_file: Path = None,
) -> list:
    """Step 2: Run Demucs stem separation."""
    banner("STEP 2 ▸ Demucs Stem Separation")

    from phase1_extraction_lab.scripts.stem_separator import batch_separate, run_demucs

    if single_file:
        print(f"   Single-file mode: {single_file.name}")
        result = run_demucs(
            input_file=single_file,
            output_dir=output_dir,
            model=model,
            two_stems=True,
            device=device if device == "cuda" else None,
            shifts=1,   # Use 1 shift for speed (increase to 5 for best quality)
            overlap=0.25,
        )
        return [result]

    return batch_separate(
        input_dir=input_dir,
        output_dir=output_dir,
        models=[model],
        two_stems=True,
        device=device if device == "cuda" else None,
    )


def step_quality(stems_dir: Path, model: str) -> list:
    """Step 3: Compute quality metrics on extracted stems."""
    banner("STEP 3 ▸ Quality Metrics")

    from phase1_extraction_lab.scripts.quality_metrics import analyze_stem, classify_metric

    model_stems = stems_dir / model
    if not model_stems.exists():
        print(f"   ❌ Stems directory not found: {model_stems}")
        print("      Did Step 2 (separation) complete successfully?")
        return []

    vocal_files = sorted(model_stems.rglob("vocals.wav"))
    if not vocal_files:
        print(f"   ⚠️  No vocals.wav files found under {model_stems}")
        return []

    print(f"   Analyzing {len(vocal_files)} vocal stem(s)...\n")
    results = []
    for vf in vocal_files:
        song_name = vf.parent.name
        try:
            result = analyze_stem(vf)
            result["song"] = song_name
            results.append(result)

            snr = result.get("snr_db", 0)
            score = result.get("virasat_score") or result.get("virasat_score_estimate", "N/A")
            print(f"   🎤 {song_name}")
            print(f"      SNR: {snr:.1f} dB")
            if "sdr_db" in result:
                print(f"      SDR: {result['sdr_db']:.1f} dB "
                      f"({classify_metric('sdr', result['sdr_db'])})")
                print(f"      SIR: {result['sir_db']:.1f} dB "
                      f"({classify_metric('sir', result['sir_db'])})")
                print(f"      Virasat Score: {score}/100 — {result.get('virasat_grade','')}")
            else:
                note = result.get("note", "")
                print(f"      Virasat Score estimate: {score}/100")
                print(f"      ℹ️  {note}")
        except Exception as e:
            print(f"   ❌ Error analyzing {song_name}: {e}")

    return results


def step_bleed(stems_dir: Path, model: str, generate_plots: bool = True) -> dict:
    """Step 4: Instrument bleed detection."""
    banner("STEP 4 ▸ Instrument Bleed Detection")

    from phase1_extraction_lab.scripts.bleed_detector import analyze_path

    model_stems = stems_dir / model
    if not model_stems.exists():
        print(f"   ❌ Stems directory not found: {model_stems}")
        return {}

    return analyze_path(
        input_path=model_stems,
        generate_plots=generate_plots,
        report_dir=str(REPORTS_DIR),
    )


def step_save_report(quality_results: list, bleed_results: dict, output_path: Path) -> None:
    """Save combined summary report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "quality_metrics": quality_results,
        "bleed_detection": bleed_results,
    }
    # Convert Path objects for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, Path):
            return str(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=make_serializable)
    print(f"\n   💾 Summary report saved: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VIRASAT AI — Full Phase 1 Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline on CPU (skip download, separate 1 file quickly)
  python run_pipeline.py --skip-download --single-file data/raw/Vital_Signs_Dil_Dil_Pakistan.wav

  # Full pipeline (download all, separate with htdemucs_ft, analyze)
  python run_pipeline.py --model htdemucs_ft

  # Quick CPU smoke-test using fast model
  python run_pipeline.py --skip-download --model htdemucs
        """,
    )

    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip Step 1 (songs already downloaded in data/raw/)",
    )
    parser.add_argument(
        "--single-file", default=None, type=Path,
        help="Run on a single WAV file only (skips batch download)",
    )
    parser.add_argument(
        "--model", default="htdemucs_ft",
        choices=["htdemucs", "htdemucs_ft"],
        help="Demucs model to use (htdemucs = faster, htdemucs_ft = higher quality)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip spectrogram plot generation (faster on CPU)",
    )
    parser.add_argument(
        "--report", default=None,
        help="Path to save final JSON summary report",
    )

    args = parser.parse_args()

    # ── Header ────────────────────────────────────────────────────────────────
    print("\n" + "🎼 " * 20)
    print("  VIRASAT AI — Phase 1 Pipeline Runner")
    print("🎼 " * 20)
    print(f"\n  Model    : {args.model}")
    print(f"  Input    : {RAW_DIR}")
    print(f"  Output   : {STEMS_DIR / args.model}")
    print(f"  Reports  : {REPORTS_DIR}")

    t_total = time.time()
    device = detect_device()

    # ── Step 1: Download ──────────────────────────────────────────────────────
    if not args.skip_download and not args.single_file:
        step_download(CONFIG_PATH)
    elif args.skip_download:
        print("\n⏭️  Step 1 skipped (--skip-download)")
    elif args.single_file:
        print(f"\n⏭️  Step 1 skipped (single-file mode: {args.single_file.name})")

    # ── Step 2: Separate ──────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sep_results = step_separate(
        input_dir=RAW_DIR,
        output_dir=STEMS_DIR,
        model=args.model,
        device=device,
        single_file=args.single_file,
    )

    success_seps = sum(1 for r in sep_results if r.get("success"))
    if success_seps == 0:
        print("\n❌ No files were successfully separated. Check errors above.")
        sys.exit(1)

    # ── Step 3: Quality Metrics ───────────────────────────────────────────────
    quality_results = step_quality(STEMS_DIR, args.model)

    # ── Step 4: Bleed Detection ───────────────────────────────────────────────
    bleed_results = step_bleed(
        STEMS_DIR,
        args.model,
        generate_plots=not args.no_plots,
    )

    # ── Step 5: Final Report ──────────────────────────────────────────────────
    banner("STEP 5 ▸ Final Summary")

    report_path = Path(args.report) if args.report else \
        REPORTS_DIR / "pipeline_summary.json"
    step_save_report(quality_results, bleed_results, report_path)

    total_time = round(time.time() - t_total)
    print(f"\n{'='*60}")
    print(f"  ✅ Pipeline complete in {total_time}s ({total_time//60}m {total_time%60}s)")
    print(f"  📁 Stems: {STEMS_DIR / args.model}")
    print(f"  📊 Report: {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
