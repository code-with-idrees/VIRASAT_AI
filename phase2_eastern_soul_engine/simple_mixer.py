#!/usr/bin/env python3
"""
simple_mixer.py — Multi-Track Audio Mixer
============================================
Combines multiple WAV files with individual volume controls into a
single mixed output. Used for the final stage of the Phase 2 pipeline:
mixing ghost vocal + backing track + drums.

For production mixing, use DaVinci Resolve or Audacity.
This provides a working prototype mix for demos.

Usage:
    from simple_mixer import simple_mix
    final = simple_mix([
        ('ghost_vocal.wav', 1.0),   # Full volume
        ('sitar_backing.wav', 0.6),  # 60% volume
        ('drums_synced.wav', 0.7),   # 70% volume
    ], 'final_mix.wav')
"""

import os
import sys

import numpy as np

try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


# ─── Simple Mixer ────────────────────────────────────────────────────────────

def simple_mix(tracks_and_volumes, output_path='final_mix.wav',
               sr=22050, normalize=True, headroom_db=-1.0):
    """
    Mix multiple WAV files together with individual volume controls.

    Args:
        tracks_and_volumes: List of (wav_path, volume) tuples.
                            Volume is 0.0 to 1.0 (or higher for boost).
        output_path:        Where to save the mixed output
        sr:                 Sample rate for output
        normalize:          Normalize to prevent clipping
        headroom_db:        Headroom in dB below 0 dBFS (default: -1 dB)

    Returns:
        str: Path to the mixed WAV file

    Raises:
        ImportError: If librosa/soundfile not installed
        ValueError: If tracks_and_volumes is empty
    """
    if not AUDIO_AVAILABLE:
        raise ImportError(
            'librosa and soundfile required. '
            'Install: pip install librosa soundfile'
        )

    if not tracks_and_volumes:
        raise ValueError('No tracks provided for mixing')

    print(f'🎚️ Mixing {len(tracks_and_volumes)} tracks...')

    # Load all tracks and find the longest duration
    loaded_tracks = []
    max_len = 0

    for track_path, volume in tracks_and_volumes:
        if not os.path.exists(track_path):
            print(f'   ⚠ Skipping missing track: {track_path}')
            continue

        audio, _ = librosa.load(track_path, sr=sr)
        loaded_tracks.append((audio, volume, os.path.basename(track_path)))
        max_len = max(max_len, len(audio))

        duration = len(audio) / sr
        print(f'   📎 {os.path.basename(track_path)}: '
              f'{duration:.1f}s @ vol={volume:.1f}')

    if not loaded_tracks:
        raise ValueError('No valid tracks loaded for mixing')

    # Mix
    mix = np.zeros(max_len)

    for audio, volume, name in loaded_tracks:
        # Pad shorter tracks with silence
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))
        else:
            audio = audio[:max_len]

        mix += audio * volume

    # Normalize to prevent clipping
    if normalize:
        max_val = np.max(np.abs(mix))
        if max_val > 0:
            # Convert headroom from dB to linear
            headroom_linear = 10 ** (headroom_db / 20.0)
            mix = mix / max_val * headroom_linear

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    sf.write(output_path, mix, sr)

    duration = max_len / sr
    print(f'   ✅ Final mix saved: {output_path}')
    print(f'   Duration: {duration:.1f}s | Sample rate: {sr}Hz')

    return output_path


# ─── Fade Utilities ──────────────────────────────────────────────────────────

def apply_fade(audio, sr, fade_in_sec=0.5, fade_out_sec=1.0):
    """
    Apply fade-in and fade-out to audio.

    Args:
        audio:        Numpy audio array
        sr:           Sample rate
        fade_in_sec:  Fade-in duration in seconds
        fade_out_sec: Fade-out duration in seconds

    Returns:
        Audio with fades applied
    """
    audio = audio.copy()
    fade_in_samples = int(fade_in_sec * sr)
    fade_out_samples = int(fade_out_sec * sr)

    # Fade in
    if fade_in_samples > 0 and fade_in_samples < len(audio):
        fade_in = np.linspace(0, 1, fade_in_samples)
        audio[:fade_in_samples] *= fade_in

    # Fade out
    if fade_out_samples > 0 and fade_out_samples < len(audio):
        fade_out = np.linspace(1, 0, fade_out_samples)
        audio[-fade_out_samples:] *= fade_out

    return audio


def mix_with_fades(tracks_and_volumes, output_path='final_mix.wav',
                   sr=22050, fade_in=0.5, fade_out=1.5):
    """
    Mix tracks with automatic fade-in/fade-out for professional feel.

    Same as simple_mix but adds fades to the final output.
    """
    if not AUDIO_AVAILABLE:
        raise ImportError('librosa and soundfile required')

    # First mix without fades
    temp_path = output_path.replace('.wav', '_prefade.wav')
    simple_mix(tracks_and_volumes, temp_path, sr)

    # Apply fades
    audio, _ = librosa.load(temp_path, sr=sr)
    audio = apply_fade(audio, sr, fade_in, fade_out)

    sf.write(output_path, audio, sr)
    print(f'   🎵 Faded mix: {output_path}')

    # Clean up temp
    os.remove(temp_path)
    return output_path


# ─── Quick Preview ───────────────────────────────────────────────────────────

def create_ab_comparison(track_a_path, track_b_path,
                         output_path='ab_comparison.wav',
                         gap_sec=1.0, sr=22050):
    """
    Create an A/B comparison audio file: plays track A, then silence, then track B.
    Useful for comparing original vs Raag-locked or before/after voice conversion.

    Args:
        track_a_path: First track (labeled 'A' / 'Before')
        track_b_path: Second track (labeled 'B' / 'After')
        output_path:  Where to save comparison
        gap_sec:      Silence gap between tracks
        sr:           Sample rate
    """
    if not AUDIO_AVAILABLE:
        raise ImportError('librosa and soundfile required')

    a, _ = librosa.load(track_a_path, sr=sr)
    b, _ = librosa.load(track_b_path, sr=sr)

    gap = np.zeros(int(gap_sec * sr))

    comparison = np.concatenate([a, gap, b])

    # Normalize
    max_val = np.max(np.abs(comparison))
    if max_val > 0:
        comparison = comparison / max_val * 0.9

    sf.write(output_path, comparison, sr)
    a_dur = len(a) / sr
    b_dur = len(b) / sr
    print(f'🔀 A/B comparison: {output_path}')
    print(f'   Track A: 0-{a_dur:.1f}s | Gap | '
          f'Track B: {a_dur + gap_sec:.1f}-{a_dur + gap_sec + b_dur:.1f}s')
    return output_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='VIRASAT.AI — Multi-Track Audio Mixer'
    )
    parser.add_argument('tracks', nargs='+',
                        help='Track files in format: path:volume '
                             '(e.g. vocal.wav:1.0 drums.wav:0.7)')
    parser.add_argument('--output', default='final_mix.wav')
    parser.add_argument('--sr', type=int, default=22050)
    parser.add_argument('--fade', action='store_true',
                        help='Add fade-in/fade-out')
    args = parser.parse_args()

    # Parse track:volume pairs
    tracks_volumes = []
    for tv in args.tracks:
        if ':' in tv:
            path, vol = tv.rsplit(':', 1)
            tracks_volumes.append((path, float(vol)))
        else:
            tracks_volumes.append((tv, 1.0))

    if args.fade:
        mix_with_fades(tracks_volumes, args.output, args.sr)
    else:
        simple_mix(tracks_volumes, args.output, args.sr)
