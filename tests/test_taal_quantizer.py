#!/usr/bin/env python3
"""
test_taal_quantizer.py — Tests for Taal Pattern Generator
============================================================
Tests MIDI drum pattern generation, humanization, and Taal correctness.
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


# ─── Tests: Taal Pattern Generation ─────────────────────────────────────────

@pytest.mark.skipif(not PRETTY_MIDI_AVAILABLE, reason='pretty_midi not installed')
class TestTaalGeneration:
    """Test MIDI pattern generation for each Taal."""

    def test_keherwa_generates_midi(self):
        from phase2_eastern_soul_engine.taal_quantizer import generate_taal_midi
        path = generate_taal_midi('Keherwa', 10, 90,
                                   output_path='/tmp/test_keherwa.mid')
        assert os.path.exists(path)
        midi = pretty_midi.PrettyMIDI(path)
        assert len(midi.instruments) == 1
        assert midi.instruments[0].is_drum is True

    def test_teentaal_generates_midi(self):
        from phase2_eastern_soul_engine.taal_quantizer import generate_taal_midi
        path = generate_taal_midi('Teentaal', 10, 80,
                                   output_path='/tmp/test_teentaal.mid')
        assert os.path.exists(path)
        midi = pretty_midi.PrettyMIDI(path)
        assert len(midi.instruments[0].notes) > 0

    def test_dadra_generates_midi(self):
        from phase2_eastern_soul_engine.taal_quantizer import generate_taal_midi
        path = generate_taal_midi('Dadra', 10, 70,
                                   output_path='/tmp/test_dadra.mid')
        assert os.path.exists(path)

    def test_duration_respected(self):
        from phase2_eastern_soul_engine.taal_quantizer import generate_taal_midi
        path = generate_taal_midi('Keherwa', 30, 90,
                                   output_path='/tmp/test_duration.mid')
        midi = pretty_midi.PrettyMIDI(path)
        last_note_end = max(n.end for n in midi.instruments[0].notes)
        assert last_note_end <= 31  # Allow 1s margin

    def test_humanize_adds_variation(self):
        from phase2_eastern_soul_engine.taal_quantizer import generate_taal_midi
        path_robot = generate_taal_midi('Keherwa', 5, 90,
                                         output_path='/tmp/test_robot.mid',
                                         humanize=False)
        path_human = generate_taal_midi('Keherwa', 5, 90,
                                         output_path='/tmp/test_human.mid',
                                         humanize=True)
        midi_r = pretty_midi.PrettyMIDI(path_robot)
        midi_h = pretty_midi.PrettyMIDI(path_human)

        starts_r = sorted(n.start for n in midi_r.instruments[0].notes)
        starts_h = sorted(n.start for n in midi_h.instruments[0].notes)

        # Humanized should have slightly different timings
        # (same seed makes this deterministic, but values differ from non-humanized)
        if len(starts_r) > 2 and len(starts_h) > 2:
            diffs = [abs(a - b) for a, b in zip(starts_r[:10], starts_h[:10])]
            assert any(d > 0.001 for d in diffs), 'Humanize should shift timings'

    def test_invalid_taal_raises_error(self):
        from phase2_eastern_soul_engine.taal_quantizer import generate_taal_midi
        with pytest.raises(ValueError, match='Taal not found'):
            generate_taal_midi('NonExistentTaal', 10, 90)

    def test_no_hihat_option(self):
        from phase2_eastern_soul_engine.taal_quantizer import generate_taal_midi, DRUM_MAP
        path = generate_taal_midi('Keherwa', 10, 90,
                                   output_path='/tmp/test_nohh.mid',
                                   include_hihat=False)
        midi = pretty_midi.PrettyMIDI(path)
        hihat_pitches = {DRUM_MAP['hihat_c'], DRUM_MAP['hihat_o']}
        for note in midi.instruments[0].notes:
            assert note.pitch not in hihat_pitches, \
                'Hi-hat notes should be excluded'


@pytest.mark.skipif(not PRETTY_MIDI_AVAILABLE, reason='pretty_midi not installed')
class TestTaalInfo:
    """Test Taal information helpers."""

    def test_list_available_taals(self):
        from phase2_eastern_soul_engine.taal_quantizer import list_available_taals
        taals = list_available_taals()
        assert len(taals) >= 3
        names = [t['name'] for t in taals]
        assert 'Keherwa' in names
        assert 'Teentaal' in names
        assert 'Dadra' in names

    def test_get_taal_info(self):
        from phase2_eastern_soul_engine.taal_quantizer import get_taal_info
        info = get_taal_info('Keherwa')
        assert info['beats'] == 8
        assert info['has_pattern'] is True


# ─── Cleanup ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def cleanup():
    yield
    for f in ['/tmp/test_keherwa.mid', '/tmp/test_teentaal.mid',
              '/tmp/test_dadra.mid', '/tmp/test_duration.mid',
              '/tmp/test_robot.mid', '/tmp/test_human.mid',
              '/tmp/test_nohh.mid']:
        if os.path.exists(f):
            os.remove(f)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
