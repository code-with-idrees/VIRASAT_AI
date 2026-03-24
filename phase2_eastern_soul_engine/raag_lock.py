#!/usr/bin/env python3
"""
raag_lock.py — Raag-Lock Constraint Filter (THE CORE MODULE)
==============================================================
Forces any AI-generated MIDI to play only notes inside a chosen Raag's
allowed note set. This is the mathematical backbone of Virasat.AI's
cultural authenticity — no other AI music tool does this.

Theory:
    A Raag defines which notes (pitch classes) are allowed. Every MIDI note
    has a pitch number (0-127). The chromatic class is: pitch % 12.
    If that class is NOT in the Raag's note set, we correct it.

Three correction strategies:
    1. 'nearest'  — Redirect to closest allowed pitch (RECOMMENDED)
    2. 'delete'   — Remove the note entirely (creates gaps)
    3. 'octave'   — Move to nearest allowed note (same as nearest, legacy)

Usage:
    from raag_lock import apply_raag_lock
    result = apply_raag_lock('melody.mid', 'Bhairavi', strategy='nearest')
    print(result['stats'])
"""

import os
import sys

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False

# Add parent for imports
sys.path.insert(0, os.path.dirname(__file__))
from raag_database import RAAG_DATABASE


# ─── Core Algorithm ──────────────────────────────────────────────────────────

def get_nearest_raag_note(midi_note, allowed_chromatic_classes):
    """
    Given a MIDI note number and a set of allowed chromatic classes,
    return the nearest allowed MIDI note.

    Uses 'nearest-note substitution' — never just deletes.
    Prefers same octave; checks adjacent octaves for boundary cases.

    Args:
        midi_note:                 MIDI note number (0-127)
        allowed_chromatic_classes: Set or list of allowed pitch classes (0-11)

    Returns:
        int: Corrected MIDI note number (0-127)

    Example:
        >>> get_nearest_raag_note(61, {0, 1, 3, 5, 7, 8, 10})  # Bhairavi
        60  # C# (1) is allowed, but C (0) at 60 is closer to 61
    """
    note_class = midi_note % 12
    allowed = set(allowed_chromatic_classes)

    # Already valid?
    if note_class in allowed:
        return midi_note

    octave = midi_note // 12
    best_note = None
    best_distance = 128  # Maximum possible distance

    for allowed_class in allowed:
        # Try same octave
        candidate = octave * 12 + allowed_class
        distance = abs(midi_note - candidate)
        if distance < best_distance and 0 <= candidate <= 127:
            best_distance = distance
            best_note = candidate

        # Try adjacent octaves (for notes near octave boundaries)
        for oct_shift in [-1, 1]:
            candidate = (octave + oct_shift) * 12 + allowed_class
            if 0 <= candidate <= 127:
                distance = abs(midi_note - candidate)
                if distance < best_distance:
                    best_distance = distance
                    best_note = candidate

    return best_note if best_note is not None else midi_note


def apply_raag_lock(input_midi_path, raag_name, output_midi_path=None,
                    strategy='nearest', excluded_instruments=None):
    """
    Main Raag-Lock function. Processes a MIDI file and returns a version
    where all melodic notes conform to the chosen Raag.

    Percussion tracks (is_drum=True) and excluded instruments are skipped.

    Args:
        input_midi_path:      Path to input .mid file
        raag_name:            Name of Raag from RAAG_DATABASE
        output_midi_path:     Where to save output (auto-named if None)
        strategy:             'nearest' (recommended), 'delete', or 'octave'
        excluded_instruments: List of instrument names to skip
                              (e.g. ['Drums'] — always exclude percussion)

    Returns:
        dict with:
            output_path:  Path to the filtered MIDI file
            stats:        Dict with total_notes, modified_notes, deleted_notes
            raag:         Raag name used
            strategy:     Strategy used

    Raises:
        ValueError: If raag_name not found
        ImportError: If pretty_midi not installed
    """
    if not PRETTY_MIDI_AVAILABLE:
        raise ImportError('pretty_midi required. Install: pip install pretty_midi')

    raag = RAAG_DATABASE.get(raag_name)
    if not raag:
        available = ', '.join(RAAG_DATABASE.keys())
        raise ValueError(f'Raag not found: {raag_name}. Available: {available}')

    allowed = set(raag['notes'])
    excluded_instruments = excluded_instruments or [
        'Drums', 'Percussion', 'Tabla', 'Dholak',
    ]

    midi = pretty_midi.PrettyMIDI(input_midi_path)

    stats = {
        'total_notes': 0,
        'modified_notes': 0,
        'deleted_notes': 0,
        'already_valid': 0,
    }

    for instrument in midi.instruments:
        # Skip percussion tracks
        if instrument.is_drum:
            continue
        if any(ex.lower() in instrument.name.lower()
               for ex in excluded_instruments):
            continue

        valid_notes = []
        for note in instrument.notes:
            stats['total_notes'] += 1
            note_class = note.pitch % 12

            if note_class in allowed:
                valid_notes.append(note)
                stats['already_valid'] += 1

            elif strategy == 'delete':
                stats['deleted_notes'] += 1
                # Don't append — note is removed

            elif strategy in ('nearest', 'octave'):
                corrected = get_nearest_raag_note(note.pitch, allowed)
                note.pitch = corrected
                valid_notes.append(note)
                stats['modified_notes'] += 1

            else:
                raise ValueError(
                    f'Unknown strategy: {strategy}. '
                    f'Use: nearest, delete, octave'
                )

        instrument.notes = valid_notes

    # Auto-name output
    if output_midi_path is None:
        base, ext = os.path.splitext(input_midi_path)
        output_midi_path = f'{base}_raaglock_{raag_name}{ext}'

    midi.write(output_midi_path)

    # Report
    total = max(stats['total_notes'], 1)
    mod_pct = stats['modified_notes'] / total * 100
    del_pct = stats['deleted_notes'] / total * 100
    valid_pct = stats['already_valid'] / total * 100

    print(f'🔒 Raag-Lock Applied: {raag_name} (strategy={strategy})')
    print(f'   Total notes:     {stats["total_notes"]}')
    print(f'   Already valid:   {stats["already_valid"]} ({valid_pct:.1f}%)')
    print(f'   Modified:        {stats["modified_notes"]} ({mod_pct:.1f}%)')
    print(f'   Deleted:         {stats["deleted_notes"]} ({del_pct:.1f}%)')

    if mod_pct > 40:
        print(f'   ⚠ WARNING: >{mod_pct:.0f}% notes modified — '
              f'generated track may be too Western. Consider re-prompting.')
    elif mod_pct > 20:
        print(f'   ℹ NOTICE: {mod_pct:.0f}% notes modified — '
              f'acceptable but re-prompting may improve quality.')

    return {
        'output_path': output_midi_path,
        'stats': stats,
        'raag': raag_name,
        'strategy': strategy,
    }


# ─── Batch Processing ────────────────────────────────────────────────────────

def batch_raag_lock(midi_dir, raag_name, strategy='nearest',
                    output_dir=None):
    """
    Apply Raag-Lock to all MIDI files in a directory.

    Args:
        midi_dir:    Directory containing .mid files
        raag_name:   Target Raag
        strategy:    Correction strategy
        output_dir:  Output directory (auto-created if None)

    Returns:
        list of result dicts from apply_raag_lock
    """
    if output_dir is None:
        output_dir = os.path.join(midi_dir, f'raag_locked_{raag_name}')
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for fname in sorted(os.listdir(midi_dir)):
        if fname.endswith('.mid') or fname.endswith('.midi'):
            input_path = os.path.join(midi_dir, fname)
            output_path = os.path.join(output_dir, fname)
            try:
                result = apply_raag_lock(
                    input_path, raag_name,
                    output_midi_path=output_path,
                    strategy=strategy,
                )
                results.append(result)
            except Exception as e:
                print(f'   ❌ Error processing {fname}: {e}')
                results.append({'error': str(e), 'file': fname})

    print(f'\n📊 Batch complete: {len(results)} files processed')
    return results


# ─── Analysis Helper ─────────────────────────────────────────────────────────

def analyze_raag_compliance(midi_path, raag_name):
    """
    Analyze how well a MIDI file already conforms to a Raag
    WITHOUT modifying it. Useful for quality checking.

    Returns:
        dict with compliance_rate, out_of_raag_notes, etc.
    """
    if not PRETTY_MIDI_AVAILABLE:
        raise ImportError('pretty_midi required')

    raag = RAAG_DATABASE.get(raag_name)
    if not raag:
        raise ValueError(f'Raag not found: {raag_name}')

    allowed = set(raag['notes'])
    midi = pretty_midi.PrettyMIDI(midi_path)

    total = 0
    in_raag = 0
    out_notes = []

    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            total += 1
            pc = note.pitch % 12
            if pc in allowed:
                in_raag += 1
            else:
                out_notes.append({
                    'pitch': note.pitch,
                    'pitch_class': pc,
                    'time': round(note.start, 2),
                })

    compliance = in_raag / max(total, 1) * 100

    return {
        'raag': raag_name,
        'total_notes': total,
        'in_raag': in_raag,
        'out_of_raag': total - in_raag,
        'compliance_rate': round(compliance, 1),
        'out_of_raag_notes': out_notes[:20],  # First 20 for brevity
        'is_compliant': compliance >= 90,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='VIRASAT.AI — Raag-Lock Filter'
    )
    parser.add_argument('input', help='Input MIDI file or directory')
    parser.add_argument('--raag', required=True, help='Target Raag name')
    parser.add_argument('--strategy', default='nearest',
                        choices=['nearest', 'delete', 'octave'])
    parser.add_argument('--output', default=None)
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze only — don\'t modify')
    args = parser.parse_args()

    if args.analyze:
        result = analyze_raag_compliance(args.input, args.raag)
        print(f'\n📊 Raag Compliance Analysis: {args.raag}')
        print(f'   Compliance: {result["compliance_rate"]}%')
        print(f'   In-Raag: {result["in_raag"]}/{result["total_notes"]}')
        if result['out_of_raag_notes']:
            print(f'   First out-of-Raag notes:')
            for n in result['out_of_raag_notes'][:5]:
                print(f'     Pitch {n["pitch"]} (class {n["pitch_class"]}) '
                      f'at {n["time"]}s')
    elif os.path.isdir(args.input):
        batch_raag_lock(args.input, args.raag, args.strategy, args.output)
    else:
        apply_raag_lock(args.input, args.raag,
                        output_midi_path=args.output,
                        strategy=args.strategy)
