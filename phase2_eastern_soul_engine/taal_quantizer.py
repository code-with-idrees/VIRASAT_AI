#!/usr/bin/env python3
"""
taal_quantizer.py — Taal-Correct Drum Pattern Generator
=========================================================
Generates rhythmically authentic drum patterns based on Eastern Taal cycles.
Unlike Western 4/4 time, Pakistani/Indian Taals have specific accent patterns
that this module respects mathematically.

Supports: Keherwa (8-beat), Teentaal (16-beat), Dadra (6-beat)

Uses General MIDI drum mapping with Tabla proxies (since standard MIDI
doesn't have dedicated Tabla patches). For production, replace the MIDI
output with real Tabla samples via a sampler.

Usage:
    from taal_quantizer import generate_taal_midi
    midi_path = generate_taal_midi('Keherwa', duration_seconds=60, bpm=90)
"""

import os
import sys

import numpy as np

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False

sys.path.insert(0, os.path.dirname(__file__))
from raag_database import TAAL_DATABASE


# ─── MIDI Drum Map ───────────────────────────────────────────────────────────
# General MIDI percussion — using closest matches for Tabla sounds.
# In production, route these to a Tabla VST/sampler.

DRUM_MAP = {
    'kick':       35,   # Acoustic Bass Drum
    'snare':      38,   # Acoustic Snare
    'hihat_c':    42,   # Closed Hi-Hat
    'hihat_o':    46,   # Open Hi-Hat
    'clap':       39,   # Hand Clap
    'tabla_na':   64,   # Low Conga — proxy for Tabla 'Na' (open right hand)
    'tabla_ge':   63,   # Open Conga — proxy for Tabla 'Ge' (bass left hand)
    'tabla_dha':  62,   # Mute Conga — proxy for 'Dha' (both hands, full resonance)
    'tabla_tin':  65,   # High Timbale — proxy for 'Tin' (rim tap)
    'tabla_te':   66,   # Low Timbale — proxy for 'Te' (quick strike)
    'dholak_hi':  60,   # High Bongo — proxy for Dholak treble
    'dholak_lo':  61,   # Low Bongo — proxy for Dholak bass
}


# ─── Taal Beat Patterns ──────────────────────────────────────────────────────
# Each pattern defines hits as (drum_type, velocity, beat_offset_in_cycle).
# Velocity reflects accent strength: Sam=115, other accents=85-95, fills=60-75.

TAAL_PATTERNS = {
    'Keherwa': {
        'beats': 8,
        'hits': [
            # Main pattern (Dha Ge Na Ti | Na Ke Dhin Na)
            ('tabla_dha', 110, 0),    # Sam (beat 1) — strongest accent
            ('tabla_ge',  70, 1),     # Light fill
            ('tabla_na',  90, 2),     # Accent
            ('tabla_tin', 70, 3),     # Fill
            ('tabla_na',  85, 4),     # Khali — secondary accent
            ('tabla_ge',  65, 5),     # Light fill
            ('tabla_dha', 80, 6),     # Accent
            ('tabla_na',  70, 7),     # Resolution
            # Modern hi-hat layer (optional, for fusion feel)
            ('hihat_c',  45, 0.5),
            ('hihat_c',  40, 1.5),
            ('hihat_c',  45, 2.5),
            ('hihat_c',  40, 3.5),
            ('hihat_c',  45, 4.5),
            ('hihat_c',  40, 5.5),
            ('hihat_c',  45, 6.5),
            ('hihat_c',  40, 7.5),
        ],
    },
    'Teentaal': {
        'beats': 16,
        'hits': [
            # Vibhag 1 (Sam) — Dha Dhin Dhin Dha
            ('tabla_dha', 115, 0),    # SAM — strongest beat
            ('tabla_dha', 75, 1),
            ('tabla_dha', 80, 2),
            ('tabla_dha', 70, 3),
            # Vibhag 2 — Dha Dhin Dhin Dha
            ('tabla_dha', 95, 4),
            ('tabla_ge',  65, 5),
            ('tabla_na',  80, 6),
            ('tabla_na',  65, 7),
            # Vibhag 3 (Khali) — Na Tin Tin Na
            ('tabla_na',  60, 8),     # KHALI — lighter
            ('tabla_tin', 55, 9),
            ('tabla_na',  75, 10),
            ('tabla_na',  60, 11),
            # Vibhag 4 — Dha Dhin Dhin Dha
            ('tabla_dha', 90, 12),
            ('tabla_ge',  65, 13),
            ('tabla_na',  80, 14),
            ('tabla_na',  65, 15),
        ],
    },
    'Dadra': {
        'beats': 6,
        'hits': [
            # Vibhag 1 — Dha Dhin Na
            ('tabla_dha', 110, 0),    # Sam
            ('tabla_dha', 75, 1),
            ('tabla_na',  65, 2),
            # Vibhag 2 — Dha Tin Na
            ('tabla_dha', 80, 3),     # Khali
            ('tabla_tin', 70, 4),
            ('tabla_na',  60, 5),
        ],
    },
    'Jhaptaal': {
        'beats': 10,
        'hits': [
            # 2+3+2+3 structure
            ('tabla_dha', 115, 0),    # Sam
            ('tabla_na',  75, 1),
            ('tabla_dha', 90, 2),
            ('tabla_dha', 70, 3),
            ('tabla_na',  65, 4),
            ('tabla_tin', 60, 5),     # Khali
            ('tabla_na',  75, 6),
            ('tabla_dha', 85, 7),
            ('tabla_dha', 70, 8),
            ('tabla_na',  65, 9),
        ],
    },
    'Ektaal': {
        'beats': 12,
        'hits': [
            # 2+2+2+2+2+2 structure — very meditative
            ('tabla_dha', 110, 0),    # Sam
            ('tabla_ge',  60, 1),
            ('tabla_na',  80, 2),
            ('tabla_ge',  55, 3),
            ('tabla_tin', 55, 4),     # Khali
            ('tabla_na',  55, 5),
            ('tabla_dha', 75, 6),
            ('tabla_ge',  55, 7),
            ('tabla_na',  70, 8),
            ('tabla_ge',  55, 9),
            ('tabla_dha', 80, 10),
            ('tabla_na',  60, 11),
        ],
    },
    'Rupak': {
        'beats': 7,
        'hits': [
            # 3+2+2 structure — Sam is Khali (unique to Rupak)
            ('tabla_tin', 90, 0),     # Sam (also Khali — light!)
            ('tabla_tin', 65, 1),
            ('tabla_na',  60, 2),
            ('tabla_dha', 95, 3),     # Accent
            ('tabla_na',  70, 4),
            ('tabla_dha', 85, 5),     # Accent
            ('tabla_na',  65, 6),
        ],
    },
}


# ─── Pattern Generator ───────────────────────────────────────────────────────

def generate_taal_midi(taal_name, duration_seconds, bpm, output_path=None,
                       humanize=True, include_hihat=True):
    """
    Generate a Taal-correct MIDI drum pattern.

    Args:
        taal_name:         Name from TAAL_PATTERNS (e.g. 'Keherwa')
        duration_seconds:  How long the pattern should play
        bpm:               Beats per minute
        output_path:       Where to save MIDI (auto-named if None)
        humanize:          Add timing variations (±10ms) for natural feel
        include_hihat:     Include modern hi-hat layer (if defined)

    Returns:
        str: Path to the generated MIDI file

    Raises:
        ValueError: If taal_name not found
        ImportError: If pretty_midi not installed
    """
    if not PRETTY_MIDI_AVAILABLE:
        raise ImportError('pretty_midi required. Install: pip install pretty_midi')

    pattern = TAAL_PATTERNS.get(taal_name)
    if not pattern:
        available = ', '.join(TAAL_PATTERNS.keys())
        raise ValueError(f'Taal not found: {taal_name}. Available: {available}')

    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    drum_track = pretty_midi.Instrument(
        program=0, is_drum=True, name=f'Tabla_{taal_name}'
    )

    beats_in_cycle = pattern['beats']
    beat_duration = 60.0 / bpm
    cycle_duration = beat_duration * beats_in_cycle

    np.random.seed(42)  # Reproducible humanization

    current_time = 0.0
    total_hits = 0

    while current_time < duration_seconds:
        for (drum_type, velocity, beat_offset) in pattern['hits']:
            # Skip hi-hat if not requested
            if not include_hihat and 'hihat' in drum_type:
                continue

            note_time = current_time + beat_offset * beat_duration

            if note_time >= duration_seconds:
                break

            # Humanization: add ±10ms random timing variation
            if humanize:
                jitter = np.random.uniform(-0.01, 0.01)
                note_time = max(0, note_time + jitter)

                # Also vary velocity slightly (±5)
                velocity_var = velocity + np.random.randint(-5, 6)
                velocity_var = max(1, min(127, velocity_var))
            else:
                velocity_var = velocity

            note = pretty_midi.Note(
                velocity=velocity_var,
                pitch=DRUM_MAP[drum_type],
                start=note_time,
                end=note_time + 0.05,  # 50ms note duration (percussion)
            )
            drum_track.notes.append(note)
            total_hits += 1

        current_time += cycle_duration

    midi.instruments.append(drum_track)

    # Auto-name output
    if output_path is None:
        output_path = f'taal_{taal_name}_{bpm}bpm.mid'

    midi.write(output_path)

    n_cycles = int(duration_seconds / cycle_duration)
    print(f'🥁 Taal MIDI generated: {taal_name}')
    print(f'   BPM: {bpm} | Duration: {duration_seconds}s | Cycles: {n_cycles}')
    print(f'   Total hits: {total_hits} | Humanize: {humanize}')
    print(f'   Saved: {output_path}')

    return output_path


def get_taal_info(taal_name):
    """
    Get information about a Taal, merging pattern data with database info.
    """
    info = {}

    if taal_name in TAAL_DATABASE:
        info.update(TAAL_DATABASE[taal_name])

    if taal_name in TAAL_PATTERNS:
        info['n_hits_per_cycle'] = len(TAAL_PATTERNS[taal_name]['hits'])
        info['has_pattern'] = True
    else:
        info['has_pattern'] = False

    return info


def list_available_taals():
    """List all Taals that have defined drum patterns."""
    result = []
    for name in TAAL_PATTERNS:
        info = get_taal_info(name)
        result.append({
            'name': name,
            'beats': info.get('beats', TAAL_PATTERNS[name]['beats']),
            'structure': info.get('structure', ''),
            'feel': info.get('feel', ''),
            'hits_per_cycle': info.get('n_hits_per_cycle', 0),
        })
    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='VIRASAT.AI — Taal Pattern Generator'
    )
    parser.add_argument('--taal', default='Keherwa',
                        choices=list(TAAL_PATTERNS.keys()),
                        help='Taal name')
    parser.add_argument('--bpm', type=int, default=90, help='Tempo in BPM')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration in seconds')
    parser.add_argument('--output', default=None)
    parser.add_argument('--no-humanize', action='store_true',
                        help='Disable timing humanization')
    parser.add_argument('--no-hihat', action='store_true',
                        help='Exclude modern hi-hat layer')
    parser.add_argument('--list', action='store_true',
                        help='List available Taals')
    args = parser.parse_args()

    if args.list:
        print('🥁 Available Taal Patterns:')
        for t in list_available_taals():
            print(f'   {t["name"]:12s} | {t["beats"]:2d} beats | '
                  f'{t["structure"]:12s} | {t["hits_per_cycle"]} hits/cycle')
    else:
        generate_taal_midi(
            args.taal, args.duration, args.bpm,
            output_path=args.output,
            humanize=not args.no_humanize,
            include_hihat=not args.no_hihat,
        )
