#!/usr/bin/env python3
"""
test_raag_lock.py — Unit Tests for Raag-Lock Filter
======================================================
Tests the core Raag-Lock algorithm: nearest-note correction,
full MIDI filtering, strategy comparison, and compliance analysis.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False


# ─── Helper: Create Synthetic MIDI ───────────────────────────────────────────

def create_test_midi(notes, output_path='/tmp/test_raag_lock.mid'):
    """
    Create a simple MIDI file with given note pitches for testing.

    Args:
        notes: List of MIDI pitch numbers (0-127)
        output_path: Where to save the MIDI

    Returns:
        str: Path to the created MIDI file
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name='TestMelody')

    for i, pitch in enumerate(notes):
        note = pretty_midi.Note(
            velocity=80,
            pitch=pitch,
            start=i * 0.5,
            end=(i + 1) * 0.5 - 0.05,
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(output_path)
    return output_path


# ─── Tests: get_nearest_raag_note ────────────────────────────────────────────

class TestGetNearestRaagNote:
    """Test the pitch correction algorithm."""

    def test_already_valid_note(self):
        from phase2_eastern_soul_engine.raag_lock import get_nearest_raag_note
        # C (pitch class 0) is in Bhairavi [0, 1, 3, 5, 7, 8, 10]
        result = get_nearest_raag_note(60, {0, 1, 3, 5, 7, 8, 10})
        assert result == 60  # No change needed

    def test_correct_to_nearest_lower(self):
        from phase2_eastern_soul_engine.raag_lock import get_nearest_raag_note
        # D (pitch 62, class 2) is NOT in Bhairavi. Nearest: Db (61, class 1)
        result = get_nearest_raag_note(62, {0, 1, 3, 5, 7, 8, 10})
        assert result == 61 or result == 63  # Either Db or Eb is acceptable

    def test_correct_to_nearest_higher(self):
        from phase2_eastern_soul_engine.raag_lock import get_nearest_raag_note
        # F# (pitch 66, class 6) is NOT in Bhairavi. Nearest: F (65, class 5) or G (67, class 7)
        result = get_nearest_raag_note(66, {0, 1, 3, 5, 7, 8, 10})
        assert result in (65, 67)  # F or G

    def test_all_bhairavi_notes_unchanged(self):
        from phase2_eastern_soul_engine.raag_lock import get_nearest_raag_note
        bhairavi = {0, 1, 3, 5, 7, 8, 10}
        for octave in range(3, 8):  # C3 to C7
            for pc in bhairavi:
                midi_note = octave * 12 + pc
                if 0 <= midi_note <= 127:
                    result = get_nearest_raag_note(midi_note, bhairavi)
                    assert result == midi_note, (
                        f'Note {midi_note} (class {pc}) should be unchanged'
                    )

    def test_all_yaman_notes_unchanged(self):
        from phase2_eastern_soul_engine.raag_lock import get_nearest_raag_note
        yaman = {0, 2, 4, 6, 7, 9, 11}
        for octave in range(3, 8):
            for pc in yaman:
                midi_note = octave * 12 + pc
                if 0 <= midi_note <= 127:
                    result = get_nearest_raag_note(midi_note, yaman)
                    assert result == midi_note

    def test_output_in_valid_midi_range(self):
        from phase2_eastern_soul_engine.raag_lock import get_nearest_raag_note
        for midi_note in range(0, 128):
            result = get_nearest_raag_note(midi_note, {0, 4, 7})
            assert 0 <= result <= 127, f'Note {midi_note} corrected to {result} (out of range)'

    def test_result_is_in_allowed_set(self):
        from phase2_eastern_soul_engine.raag_lock import get_nearest_raag_note
        allowed = {0, 2, 4, 5, 7, 9, 11}  # Major scale
        for midi_note in range(36, 96):
            result = get_nearest_raag_note(midi_note, allowed)
            assert result % 12 in allowed, (
                f'Note {midi_note} corrected to {result} '
                f'(class {result % 12}) not in allowed set'
            )


# ─── Tests: apply_raag_lock ─────────────────────────────────────────────────

@pytest.mark.skipif(not PRETTY_MIDI_AVAILABLE, reason='pretty_midi not installed')
class TestApplyRaagLock:
    """Test the full MIDI filtering function."""

    def test_all_valid_notes_unchanged(self):
        from phase2_eastern_soul_engine.raag_lock import apply_raag_lock
        # Create MIDI with only Bhairavi notes: C, Db, Eb, F, G, Ab, Bb
        bhairavi_pitches = [60, 61, 63, 65, 67, 68, 70]
        midi_path = create_test_midi(bhairavi_pitches)
        result = apply_raag_lock(midi_path, 'Bhairavi', strategy='nearest')
        assert result['stats']['modified_notes'] == 0
        assert result['stats']['already_valid'] == 7

    def test_out_of_raag_notes_modified(self):
        from phase2_eastern_soul_engine.raag_lock import apply_raag_lock
        # Create MIDI with some non-Bhairavi notes (D=62, E=64, F#=66, B=71)
        mixed_pitches = [60, 62, 64, 66, 67, 71]
        midi_path = create_test_midi(mixed_pitches)
        result = apply_raag_lock(midi_path, 'Bhairavi', strategy='nearest')
        # 60 (C) and 67 (G) are valid; 62, 64, 66, 71 should be modified
        assert result['stats']['modified_notes'] == 4
        assert result['stats']['already_valid'] == 2

    def test_delete_strategy(self):
        from phase2_eastern_soul_engine.raag_lock import apply_raag_lock
        mixed_pitches = [60, 62, 64, 66, 67, 71]
        midi_path = create_test_midi(mixed_pitches)
        result = apply_raag_lock(midi_path, 'Bhairavi', strategy='delete')
        assert result['stats']['deleted_notes'] == 4
        assert result['stats']['modified_notes'] == 0

    def test_output_file_created(self):
        from phase2_eastern_soul_engine.raag_lock import apply_raag_lock
        midi_path = create_test_midi([60, 62, 64])
        output = '/tmp/test_raag_lock_output.mid'
        result = apply_raag_lock(midi_path, 'Yaman',
                                  output_midi_path=output)
        assert os.path.exists(output)
        assert result['output_path'] == output

    def test_invalid_raag_raises_error(self):
        from phase2_eastern_soul_engine.raag_lock import apply_raag_lock
        midi_path = create_test_midi([60])
        with pytest.raises(ValueError, match='Raag not found'):
            apply_raag_lock(midi_path, 'NonExistentRaag')

    def test_percussion_track_skipped(self):
        from phase2_eastern_soul_engine.raag_lock import apply_raag_lock
        # Create MIDI with a drum track
        midi = pretty_midi.PrettyMIDI()
        drums = pretty_midi.Instrument(program=0, is_drum=True, name='Drums')
        drums.notes.append(pretty_midi.Note(
            velocity=100, pitch=36, start=0, end=0.5,
        ))
        midi.instruments.append(drums)
        path = '/tmp/test_drums.mid'
        midi.write(path)
        result = apply_raag_lock(path, 'Bhairavi')
        assert result['stats']['total_notes'] == 0  # Drums are skipped


# ─── Tests: analyze_raag_compliance ──────────────────────────────────────────

@pytest.mark.skipif(not PRETTY_MIDI_AVAILABLE, reason='pretty_midi not installed')
class TestAnalyzeRaagCompliance:
    """Test the compliance analysis function."""

    def test_perfect_compliance(self):
        from phase2_eastern_soul_engine.raag_lock import analyze_raag_compliance
        bhairavi_pitches = [60, 61, 63, 65, 67, 68, 70]
        midi_path = create_test_midi(bhairavi_pitches)
        result = analyze_raag_compliance(midi_path, 'Bhairavi')
        assert result['compliance_rate'] == 100.0
        assert result['is_compliant'] is True

    def test_low_compliance(self):
        from phase2_eastern_soul_engine.raag_lock import analyze_raag_compliance
        # All chromatic notes — many will be out of Raag
        chromatic = list(range(60, 72))  # C4 to B4
        midi_path = create_test_midi(chromatic)
        result = analyze_raag_compliance(midi_path, 'Bhairavi')
        # Bhairavi has 7 notes out of 12, so compliance ~ 58%
        assert result['compliance_rate'] < 70
        assert result['is_compliant'] is False


# ─── Cleanup ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up generated test MIDI files after each test."""
    yield
    for f in ['/tmp/test_raag_lock.mid', '/tmp/test_raag_lock_output.mid',
              '/tmp/test_drums.mid']:
        if os.path.exists(f):
            os.remove(f)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
