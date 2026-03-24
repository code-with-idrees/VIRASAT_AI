#!/usr/bin/env python3
"""
test_beat_sync.py — Tests for DTW Beat Synchronizer
test_taal_quantizer.py — Tests for Taal Pattern Generator
============================================================
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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


# ─── Helper: Create Synthetic Audio ──────────────────────────────────────────

def create_test_audio(duration=5, bpm=120, sr=22050, path='/tmp/test_beat.wav'):
    """Create a simple click track at given BPM for testing."""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    audio = np.zeros_like(t)

    beat_interval = 60.0 / bpm
    click_duration = 0.01  # 10ms click

    for beat_time in np.arange(0, duration, beat_interval):
        start_sample = int(beat_time * sr)
        end_sample = min(start_sample + int(click_duration * sr), len(audio))
        if start_sample < len(audio):
            audio[start_sample:end_sample] = 0.8

    sf.write(path, audio, sr)
    return path


# ─── Tests: Beat Detection ──────────────────────────────────────────────────

@pytest.mark.skipif(not LIBROSA_AVAILABLE or not SOUNDFILE_AVAILABLE,
                    reason='librosa/soundfile not installed')
class TestBeatDetection:
    """Test beat extraction from audio."""

    def test_extract_beats_returns_array(self):
        from phase2_eastern_soul_engine.beat_sync import extract_beat_times
        audio_path = create_test_audio(duration=5, bpm=120)
        beats, tempo = extract_beat_times(audio_path)
        assert isinstance(beats, np.ndarray)
        assert len(beats) > 0

    def test_tempo_reasonable_range(self):
        from phase2_eastern_soul_engine.beat_sync import extract_beat_times
        audio_path = create_test_audio(duration=5, bpm=120)
        beats, tempo = extract_beat_times(audio_path)
        # Tempo should be in reasonable range (librosa may estimate double/half tempo)
        assert 50 < tempo < 300, f'Tempo {tempo} seems unreasonable'

    def test_beats_are_sorted(self):
        from phase2_eastern_soul_engine.beat_sync import extract_beat_times
        audio_path = create_test_audio(duration=5, bpm=100)
        beats, _ = extract_beat_times(audio_path)
        for i in range(1, len(beats)):
            assert beats[i] > beats[i-1], 'Beat times should be monotonically increasing'


# ─── Cleanup ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def cleanup():
    yield
    for f in ['/tmp/test_beat.wav', '/tmp/test_beat2.wav',
              '/tmp/synced_test.wav']:
        if os.path.exists(f):
            os.remove(f)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
