#!/usr/bin/env python3
"""
test_bleed_detection.py — Tests for Bleed Detection Engine
============================================================
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSpectralEnergy:
    """Test spectral energy computation."""

    def test_energy_in_band(self):
        from phase1_extraction_lab.scripts.bleed_detector import compute_spectral_energy
        # Create mock spectrogram
        freqs = np.linspace(0, 22050, 1025)
        S_power = np.ones((1025, 100))
        energy = compute_spectral_energy(S_power, freqs, 200, 700)
        assert energy > 0

    def test_zero_energy_outside_band(self):
        from phase1_extraction_lab.scripts.bleed_detector import compute_spectral_energy
        freqs = np.linspace(0, 22050, 1025)
        S_power = np.zeros((1025, 100))
        energy = compute_spectral_energy(S_power, freqs, 200, 700)
        assert energy < 1  # Should be near-zero (+ epsilon)


class TestOverallBleedScore:
    """Test overall bleed score computation."""

    def test_all_clean(self):
        from phase1_extraction_lab.scripts.bleed_detector import compute_overall_bleed_score
        scores = {
            "sitar": {"severity": "CLEAN", "energy_ratio_dB": -25, "threshold_db": -15},
            "tabla": {"severity": "CLEAN", "energy_ratio_dB": -20, "threshold_db": -12},
        }
        overall = compute_overall_bleed_score(scores)
        assert overall < 30

    def test_severe_bleed(self):
        from phase1_extraction_lab.scripts.bleed_detector import compute_overall_bleed_score
        scores = {
            "harmonium": {"severity": "SEVERE", "energy_ratio_dB": -5, "threshold_db": -15},
        }
        overall = compute_overall_bleed_score(scores)
        assert overall >= 50

    def test_empty_scores(self):
        from phase1_extraction_lab.scripts.bleed_detector import compute_overall_bleed_score
        assert compute_overall_bleed_score({}) == 0


class TestInstrumentProfiles:
    """Test instrument profile loading."""

    def test_load_profiles(self):
        from phase1_extraction_lab.scripts.bleed_detector import load_instrument_profiles
        profiles = load_instrument_profiles()
        assert len(profiles) > 0
        assert "sitar" in profiles or "harmonium" in profiles

    def test_profile_has_required_fields(self):
        from phase1_extraction_lab.scripts.bleed_detector import load_instrument_profiles
        profiles = load_instrument_profiles()
        for name, profile in profiles.items():
            assert "low_hz" in profile
            assert "high_hz" in profile
            assert "threshold_db" in profile
            assert profile["low_hz"] < profile["high_hz"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
