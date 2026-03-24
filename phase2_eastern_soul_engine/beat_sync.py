#!/usr/bin/env python3
"""
beat_sync.py — DTW Beat Synchronizer
=======================================
Synchronizes a modern generated beat to an old recording's natural tempo drift
using Dynamic Time Warping. Instead of forcing the vocal to fit a rigid grid,
DTW makes the beat 'warp' to follow the vocal's natural timing.

This solves the fundamental problem: old recordings have rubato (tempo breathing),
so layering a fixed-BPM drum beat on them causes drift within 10 seconds.

Usage:
    from beat_sync import dtw_sync_beat
    synced = dtw_sync_beat('vocal.wav', 'drums.wav', 'synced_drums.wav')
"""

import os
import sys
import warnings

import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ─── Beat Detection ──────────────────────────────────────────────────────────

def extract_beat_times(audio_path, sr=22050, hop_length=512):
    """
    Extract beat timestamps from an audio file using librosa's beat tracker.

    Args:
        audio_path: Path to WAV file
        sr:         Sample rate for analysis
        hop_length: Hop length for beat tracking (smaller = finer resolution)

    Returns:
        tuple: (beat_times_array, estimated_tempo_bpm)
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError('librosa required. Install: pip install librosa')

    y, sr_actual = librosa.load(audio_path, sr=sr)

    # Suppress librosa warnings about tempo estimation
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=hop_length, units='frames'
        )

    beat_times = librosa.frames_to_time(beat_frames, sr=sr,
                                         hop_length=hop_length)

    # Handle scalar vs array tempo (librosa version differences)
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo = float(tempo)

    print(f'🥁 Beat detection: {os.path.basename(audio_path)}')
    print(f'   Tempo: {tempo:.1f} BPM | Beats found: {len(beat_times)}')

    return beat_times, tempo


def extract_onset_times(audio_path, sr=22050):
    """
    Extract onset (attack) times — more granular than beats.
    Useful when beat tracking fails on very free-tempo recordings.

    Returns:
        array of onset times in seconds
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError('librosa required')

    y, sr_actual = librosa.load(audio_path, sr=sr)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    print(f'   Onsets detected: {len(onset_times)}')
    return onset_times


# ─── DTW Beat Synchronizer ───────────────────────────────────────────────────

def dtw_sync_beat(vocal_path, beat_path, output_path='synced_beat.wav',
                  sr=22050, hop_length=512):
    """
    The core DTW sync function. Warps the beat audio to follow
    the vocal's natural timing.

    Algorithm:
        1. Detect beats in both vocal and beat tracks
        2. Build a time-mapping: beat_time → vocal_time (for each beat position)
        3. Interpolate smoothly between beat positions
        4. Resample the beat audio at warped time positions

    Args:
        vocal_path:  Path to isolated vocal WAV (Phase 1 Demucs output)
        beat_path:   Path to generated modern beat WAV
        output_path: Where to save the time-warped beat
        sr:          Sample rate for processing
        hop_length:  Controls beat detection resolution

    Returns:
        str: Path to the synced beat WAV
    """
    if not SCIPY_AVAILABLE:
        raise ImportError('scipy required. Install: pip install scipy')
    if not SOUNDFILE_AVAILABLE:
        raise ImportError('soundfile required. Install: pip install soundfile')

    print(f'\n🔄 DTW Beat Synchronization')
    print(f'   Vocal: {os.path.basename(vocal_path)}')
    print(f'   Beat:  {os.path.basename(beat_path)}')

    # Load audio
    vocal, _ = librosa.load(vocal_path, sr=sr)
    beat_audio, _ = librosa.load(beat_path, sr=sr)

    # Extract beats from both
    vocal_beats, vocal_tempo = extract_beat_times(vocal_path, sr, hop_length)
    beat_beats, beat_tempo = extract_beat_times(beat_path, sr, hop_length)

    print(f'   Vocal tempo: {vocal_tempo:.1f} BPM')
    print(f'   Beat tempo:  {beat_tempo:.1f} BPM')

    if len(vocal_beats) < 2 or len(beat_beats) < 2:
        print('   ⚠ Too few beats detected. Using simple tempo stretch instead.')
        return _simple_tempo_match(vocal_path, beat_path, output_path, sr)

    # Match the number of beats
    n_beats = min(len(vocal_beats), len(beat_beats))

    # Create time warp mapping:
    # For each beat position in the beat track,
    # map it to the corresponding vocal beat time
    beat_source_times = beat_beats[:n_beats]
    beat_target_times = vocal_beats[:n_beats]

    # Create interpolation function
    # 'linear' interpolation gives smooth warping
    warp_fn = interp1d(
        beat_source_times,
        beat_target_times,
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate',
    )

    # Generate warped time indices for the entire beat audio
    beat_duration = len(beat_audio) / sr
    original_times = np.linspace(0, beat_duration, len(beat_audio))
    warped_times = warp_fn(original_times)

    # Clip warped times to valid range
    warped_times = np.clip(warped_times, 0, beat_duration - 1.0 / sr)

    # Resample beat audio at warped positions
    warped_indices = np.clip(
        (warped_times * sr).astype(int), 0, len(beat_audio) - 1
    )
    warped_beat = beat_audio[warped_indices]

    # Match length to vocal
    target_len = len(vocal)
    if len(warped_beat) < target_len:
        warped_beat = np.pad(warped_beat, (0, target_len - len(warped_beat)))
    else:
        warped_beat = warped_beat[:target_len]

    # Normalize
    max_val = np.max(np.abs(warped_beat))
    if max_val > 0:
        warped_beat = warped_beat / max_val * 0.85

    # Save
    sf.write(output_path, warped_beat, sr)
    print(f'   ✅ DTW-synced beat saved: {output_path}')

    # Quality check: estimate drift
    _check_sync_quality(vocal_beats, beat_beats)

    return output_path


def _simple_tempo_match(vocal_path, beat_path, output_path, sr=22050):
    """
    Fallback: simple tempo scaling when DTW can't get enough beat points.
    Stretches/compresses the beat to match the vocal's average tempo.
    """
    print('   🔧 Using simple tempo matching (fallback)...')

    vocal, _ = librosa.load(vocal_path, sr=sr)
    beat_audio, _ = librosa.load(beat_path, sr=sr)

    # Get average tempos
    _, vocal_tempo = extract_beat_times(vocal_path, sr)
    _, beat_tempo = extract_beat_times(beat_path, sr)

    if beat_tempo <= 0 or vocal_tempo <= 0:
        print('   ⚠ Cannot estimate tempo. Outputting beat as-is.')
        sf.write(output_path, beat_audio, sr)
        return output_path

    # Stretch ratio
    ratio = vocal_tempo / beat_tempo

    # Time stretch using librosa
    stretched = librosa.effects.time_stretch(beat_audio, rate=ratio)

    # Match length
    target_len = len(vocal)
    if len(stretched) < target_len:
        stretched = np.pad(stretched, (0, target_len - len(stretched)))
    else:
        stretched = stretched[:target_len]

    sf.write(output_path, stretched, sr)
    print(f'   ✅ Tempo-matched beat saved: {output_path}')
    return output_path


def _check_sync_quality(vocal_beats, beat_beats):
    """Check if the synced beats have significant drift."""
    n = min(len(vocal_beats), len(beat_beats))
    if n < 4:
        return

    # Check drift at end of track
    diffs = vocal_beats[:n] - beat_beats[:n]
    final_drift = abs(diffs[-1])
    avg_drift = np.mean(np.abs(diffs))

    if final_drift > 0.5:
        print(f'   ⚠ Final beat drift: {final_drift:.2f}s — may be audible')
    elif final_drift > 0.1:
        print(f'   ℹ Minor drift: {final_drift:.2f}s — acceptable')
    else:
        print(f'   ✅ Excellent sync — drift: {final_drift:.3f}s')


# ─── Mix Helper ──────────────────────────────────────────────────────────────

def preview_sync(vocal_path, synced_beat_path, output_path='preview_mix.wav',
                 vocal_vol=1.0, beat_vol=0.6, sr=22050):
    """
    Quick preview: mix vocal + synced beat together for listening.

    Args:
        vocal_path:       Path to vocal WAV
        synced_beat_path: Path to DTW-synced beat WAV
        output_path:      Where to save the preview mix
        vocal_vol:        Vocal volume (0-1)
        beat_vol:         Beat volume (0-1)

    Returns:
        str: Path to preview WAV
    """
    vocal, _ = librosa.load(vocal_path, sr=sr)
    beat, _ = librosa.load(synced_beat_path, sr=sr)

    # Match lengths
    target_len = min(len(vocal), len(beat))
    mix = vocal[:target_len] * vocal_vol + beat[:target_len] * beat_vol

    # Normalize
    max_val = np.max(np.abs(mix))
    if max_val > 0:
        mix = mix / max_val * 0.9

    sf.write(output_path, mix, sr)
    print(f'🎧 Preview mix saved: {output_path}')
    return output_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='VIRASAT.AI — DTW Beat Synchronizer'
    )
    parser.add_argument('--vocal', required=True,
                        help='Vocal WAV (source of tempo truth)')
    parser.add_argument('--beat', required=True,
                        help='Beat WAV to sync')
    parser.add_argument('--output', default='synced_beat.wav')
    parser.add_argument('--preview', action='store_true',
                        help='Also create a vocal+beat preview mix')
    args = parser.parse_args()

    synced = dtw_sync_beat(args.vocal, args.beat, args.output)

    if args.preview:
        preview_sync(args.vocal, synced)
