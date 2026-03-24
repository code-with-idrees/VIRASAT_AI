#!/usr/bin/env python3
"""
pipeline.py — Phase 2 Master Pipeline Orchestrator
=====================================================
Ties all Phase 2 modules together into a single 7-step pipeline.
Running this script executes the complete Eastern Soul Engine workflow:

    Old vocal in → Raag-locked modern fusion track out

Steps:
    1. Generate Eastern backing track (Sonauto/fallback)
    2. Convert to MIDI (Basic Pitch/librosa)
    3. Apply Raag-Lock filter (nearest-note correction)
    4. Render Raag-locked MIDI to audio
    5. Generate and sync Taal drums (DTW alignment)
    6. Voice conversion — Ghost Collaboration (RVC/fallback)
    7. Final mix — combine all stems

Usage:
    python pipeline.py --raag Bhairavi --taal Keherwa --bpm 90 \\
        --style coke_studio --vocal path/to/vocal.wav --model path/to/model.pth

    # Quick test (no GPU needed):
    python pipeline.py --raag Bhairavi --taal Keherwa --bpm 90 --test
"""

import argparse
import os
import sys
import time

# Ensure phase2 modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from prompt_generator import generate_eastern_prompt
from sonauto_client import generate_track_sonauto, generate_track_local_fallback
from audio_to_midi import wav_to_midi
from raag_lock import apply_raag_lock
from midi_to_audio import midi_to_wav
from taal_quantizer import generate_taal_midi
from beat_sync import dtw_sync_beat
from rvc_inference import rvc_inference
from simple_mixer import simple_mix, mix_with_fades


# ─── Pipeline ────────────────────────────────────────────────────────────────

def run_phase2_pipeline(raag, taal, bpm, style,
                        vocal_path=None, voice_model_path=None,
                        output_dir='outputs/pipeline_run/',
                        use_api=False, duration=60):
    """
    Execute the complete Phase 2 pipeline.

    Args:
        raag:              Target Raag name (e.g. 'Bhairavi')
        taal:              Target Taal name (e.g. 'Keherwa')
        bpm:               Tempo in BPM
        style:             Style preset (e.g. 'coke_studio')
        vocal_path:        Path to clean vocal WAV (Phase 1 output)
        voice_model_path:  Path to trained RVC .pth model
        output_dir:        Where to save all outputs
        use_api:           If True, use Sonauto API; else use local fallback
        duration:          Duration in seconds

    Returns:
        dict with paths to all generated files
    """
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    print(f'\n{"=" * 55}')
    print(f'  🎵 VIRASAT.AI — PHASE 2 PIPELINE')
    print(f'{"=" * 55}')
    print(f'  Raag:  {raag}')
    print(f'  Taal:  {taal}')
    print(f'  BPM:   {bpm}')
    print(f'  Style: {style}')
    print(f'  Vocal: {vocal_path or "None (skip sync)"}')
    print(f'  Model: {voice_model_path or "None (use fallback)"}')
    print(f'{"=" * 55}\n')

    results = {}

    # ── STEP 1: Generate backing track ────────────────────────────────────
    print(f'\n[Step 1/7] 🎹 Generating Eastern backing track...')
    config = generate_eastern_prompt(raag, taal, style=style, bpm=bpm)

    if use_api:
        backing_wav = generate_track_sonauto(config, output_dir)
    else:
        backing_wav = generate_track_local_fallback(config, output_dir)

    if not backing_wav:
        print('❌ Track generation failed. Aborting pipeline.')
        return None

    results['backing_wav'] = backing_wav

    # ── STEP 2: Convert to MIDI ───────────────────────────────────────────
    print(f'\n[Step 2/7] 🎼 Converting backing track to MIDI...')
    backing_midi = wav_to_midi(backing_wav)
    results['backing_midi'] = backing_midi

    # ── STEP 3: Apply Raag-Lock ───────────────────────────────────────────
    print(f'\n[Step 3/7] 🔒 Applying Raag-Lock filter ({raag})...')
    locked = apply_raag_lock(backing_midi, raag, strategy='nearest')
    results['raag_lock_stats'] = locked['stats']
    results['locked_midi'] = locked['output_path']

    # ── STEP 4: Render locked MIDI to audio ───────────────────────────────
    print(f'\n[Step 4/7] 🔊 Rendering Raag-locked MIDI to audio...')
    locked_wav = midi_to_wav(locked['output_path'])
    results['locked_wav'] = locked_wav

    # ── STEP 5: Generate Taal drums and sync ──────────────────────────────
    print(f'\n[Step 5/7] 🥁 Generating Taal pattern ({taal}) and syncing...')
    drum_midi = generate_taal_midi(
        taal, duration, bpm,
        os.path.join(output_dir, 'drums.mid'),
    )
    drum_wav = midi_to_wav(drum_midi)
    results['drum_midi'] = drum_midi
    results['drum_wav'] = drum_wav

    # Sync to vocal if provided
    if vocal_path and os.path.exists(vocal_path):
        synced_drums = dtw_sync_beat(
            vocal_path, drum_wav,
            os.path.join(output_dir, 'drums_synced.wav'),
        )
        results['synced_drums'] = synced_drums
    else:
        synced_drums = drum_wav
        results['synced_drums'] = drum_wav
        print('   ℹ No vocal provided — skipping DTW sync')

    # ── STEP 6: Voice conversion (Ghost Collaboration) ────────────────────
    print(f'\n[Step 6/7] 👻 Ghost Collaboration — voice conversion...')
    ghost_output = os.path.join(output_dir, 'ghost_vocal.wav')
    ghost_vocal = rvc_inference(
        locked_wav,
        voice_model_path or '',
        output_path=ghost_output,
    )
    results['ghost_vocal'] = ghost_vocal

    # ── STEP 7: Final mix ─────────────────────────────────────────────────
    print(f'\n[Step 7/7] 🎚️ Mixing final track...')

    mix_tracks = [
        (ghost_vocal, 1.0),    # Ghost vocal at full volume
        (locked_wav, 0.5),     # Raag-locked backing at 50%
        (synced_drums, 0.7),   # Drums at 70%
    ]

    final_path = os.path.join(output_dir, f'FINAL_{raag}_{taal}_{bpm}bpm.wav')
    final_mix = mix_with_fades(mix_tracks, final_path)
    results['final_mix'] = final_mix

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - start_time

    print(f'\n{"=" * 55}')
    print(f'  ✅ PIPELINE COMPLETE')
    print(f'{"=" * 55}')
    print(f'  Time: {elapsed:.1f}s')
    print(f'\n  📁 Output files:')
    for key, path in results.items():
        if isinstance(path, str) and os.path.exists(path):
            size = os.path.getsize(path)
            print(f'     {key:20s} → {os.path.basename(path)} ({size/1024:.0f} KB)')
    print(f'\n  🎵 Final mix: {final_mix}')
    print(f'{"=" * 55}\n')

    return results


# ─── Quick Test Mode ─────────────────────────────────────────────────────────

def run_quick_test(output_dir='outputs/quick_test/'):
    """
    Run a quick test of the pipeline without API or GPU.
    Uses local fallback synthesizer and passthrough voice conversion.
    """
    print('🧪 Running quick pipeline test...')
    print('   (No API key or GPU needed — uses local fallbacks)\n')

    return run_phase2_pipeline(
        raag='Bhairavi',
        taal='Keherwa',
        bpm=90,
        style='coke_studio',
        vocal_path=None,
        voice_model_path=None,
        output_dir=output_dir,
        use_api=False,
        duration=30,
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='VIRASAT.AI — Phase 2 Master Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Quick test (no GPU/API needed):
  python pipeline.py --test

  # Full pipeline with vocal and model:
  python pipeline.py --raag Bhairavi --taal Keherwa --bpm 90 \\
      --vocal ../virasat_vocals/Ghulam_Ali_Chupke_Chupke/vocals.wav \\
      --model models/ghulam_ali_v1.pth

  # Use Sonauto API for track generation:
  python pipeline.py --raag Yaman --taal Teentaal --bpm 80 --api
        '''
    )

    parser.add_argument('--raag', default='Bhairavi',
                        help='Target Raag (default: Bhairavi)')
    parser.add_argument('--taal', default='Keherwa',
                        help='Target Taal (default: Keherwa)')
    parser.add_argument('--bpm', type=int, default=90,
                        help='Tempo in BPM (default: 90)')
    parser.add_argument('--style', default='coke_studio',
                        help='Style preset (default: coke_studio)')
    parser.add_argument('--vocal', default=None,
                        help='Clean vocal WAV from Phase 1')
    parser.add_argument('--model', default=None,
                        help='Trained RVC .pth model')
    parser.add_argument('--output', default='outputs/pipeline_run/',
                        help='Output directory')
    parser.add_argument('--api', action='store_true',
                        help='Use Sonauto API (requires SONAUTO_KEY)')
    parser.add_argument('--duration', type=int, default=60,
                        help='Song duration in seconds')
    parser.add_argument('--test', action='store_true',
                        help='Quick test mode (no API/GPU needed)')

    args = parser.parse_args()

    if args.test:
        results = run_quick_test()
    else:
        results = run_phase2_pipeline(
            args.raag, args.taal, args.bpm, args.style,
            args.vocal, args.model, args.output,
            args.api, args.duration,
        )

    if results:
        print('🎉 Phase 2 Eastern Soul Engine — Pipeline Success!')
    else:
        print('❌ Pipeline failed. Check errors above.')
        sys.exit(1)
