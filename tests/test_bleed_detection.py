#!/usr/bin/env python3
"""
test_bleed_detection.py — Tests for Bleed Detection Engine v2
================================================================
Tests the spectral energy computation, overall bleed score, instrument
profile loading, and the new vocal-harmonic-exclusion + spectral-flatness
gating that prevents vocal frequencies from being counted as bleed.
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

    def test_all_clean_scores_near_zero(self):
        """Clean stems must score < 10 (was < 30 before the math upgrade)."""
        from phase1_extraction_lab.scripts.bleed_detector import compute_overall_bleed_score
        scores = {
            "sitar": {"severity": "CLEAN", "energy_ratio_dB": -25, "threshold_db": -15, "is_bleeding": False},
            "tabla": {"severity": "CLEAN", "energy_ratio_dB": -20, "threshold_db": -12, "is_bleeding": False},
        }
        overall = compute_overall_bleed_score(scores)
        assert overall < 10, f"Clean stems must score < 10, got {overall}"

    def test_clean_stems_score_zero(self):
        """All-clean stems with energy well below threshold → score 0."""
        from phase1_extraction_lab.scripts.bleed_detector import compute_overall_bleed_score
        scores = {
            "sitar": {"severity": "CLEAN", "energy_ratio_dB": -40, "threshold_db": -15, "is_bleeding": False},
            "harmonium": {"severity": "CLEAN", "energy_ratio_dB": -35, "threshold_db": -15, "is_bleeding": False},
            "tabla": {"severity": "CLEAN", "energy_ratio_dB": -30, "threshold_db": -12, "is_bleeding": False},
            "sarangi": {"severity": "CLEAN", "energy_ratio_dB": -45, "threshold_db": -15, "is_bleeding": False},
        }
        overall = compute_overall_bleed_score(scores)
        assert overall == 0, f"All-clean should be 0, got {overall}"

    def test_severe_bleed_scores_high(self):
        """Actual instrument bleed should still be detected and scored high."""
        from phase1_extraction_lab.scripts.bleed_detector import compute_overall_bleed_score
        scores = {
            "harmonium": {"severity": "SEVERE", "energy_ratio_dB": -5, "threshold_db": -15, "is_bleeding": True},
        }
        overall = compute_overall_bleed_score(scores)
        assert overall >= 50, f"Severe bleed must score >= 50, got {overall}"

    def test_empty_scores(self):
        from phase1_extraction_lab.scripts.bleed_detector import compute_overall_bleed_score
        assert compute_overall_bleed_score({}) == 0

    def test_overlapping_frequency_not_penalized(self):
        """Energy in the vocal-instrument overlap band from a harmonic source
        should NOT trigger high bleed scores."""
        from phase1_extraction_lab.scripts.bleed_detector import compute_overall_bleed_score
        # Simulate: medium energy ratio but CLEAN severity (due to flatness gate)
        scores = {
            "sitar": {"severity": "CLEAN", "energy_ratio_dB": -18, "threshold_db": -15, "is_bleeding": False},
            "harmonium": {"severity": "CLEAN", "energy_ratio_dB": -19, "threshold_db": -15, "is_bleeding": False},
            "sarangi": {"severity": "CLEAN", "energy_ratio_dB": -17, "threshold_db": -15, "is_bleeding": False},
        }
        overall = compute_overall_bleed_score(scores)
        assert overall < 5, f"Harmonic overlap should score < 5, got {overall}"


class TestHarmonicMask:
    """Test the vocal harmonic exclusion mask."""

    def test_mask_covers_fundamental(self):
        from phase1_extraction_lab.scripts.bleed_detector import _build_harmonic_mask
        freqs = np.linspace(0, 22050, 1025)
        f0 = 200.0  # 200 Hz fundamental
        mask = _build_harmonic_mask(freqs, f0, n_harmonics=5, width_hz=40)
        # Should be True around 200, 400, 600, 800, 1000 Hz
        for harmonic_hz in [200, 400, 600, 800, 1000]:
            freq_idx = np.argmin(np.abs(freqs - harmonic_hz))
            assert mask[freq_idx], f"Mask should cover {harmonic_hz} Hz"

    def test_mask_empty_for_zero_f0(self):
        from phase1_extraction_lab.scripts.bleed_detector import _build_harmonic_mask
        freqs = np.linspace(0, 22050, 1025)
        mask = _build_harmonic_mask(freqs, 0.0)
        assert not np.any(mask), "Mask should be empty when f0=0"


class TestHarmonicCoverage:
    """Test harmonic coverage computation."""

    def test_high_coverage_at_harmonics(self):
        """Energy concentrated at harmonic frequencies should give high coverage."""
        from phase1_extraction_lab.scripts.bleed_detector import _compute_harmonic_coverage, _build_harmonic_mask
        freqs = np.linspace(0, 22050, 1025)
        f0 = 200.0
        harmonic_mask = _build_harmonic_mask(freqs, f0, n_harmonics=10, width_hz=40)
        # Create power spectrogram with energy only at harmonics
        S_power = np.zeros((1025, 100)) + 1e-12
        for n in range(1, 11):
            idx = np.argmin(np.abs(freqs - n * f0))
            S_power[idx, :] = 100.0
        coverage = _compute_harmonic_coverage(S_power, freqs, harmonic_mask, 200, 1200)
        assert coverage > 0.8, f"Harmonic-only energy should have high coverage, got {coverage}"

    def test_low_coverage_noise(self):
        """Uniform energy (noise) should have low harmonic coverage."""
        from phase1_extraction_lab.scripts.bleed_detector import _compute_harmonic_coverage, _build_harmonic_mask
        freqs = np.linspace(0, 22050, 1025)
        f0 = 200.0
        harmonic_mask = _build_harmonic_mask(freqs, f0, n_harmonics=10, width_hz=40)
        S_power = np.ones((1025, 100))  # Uniform noise
        coverage = _compute_harmonic_coverage(S_power, freqs, harmonic_mask, 200, 1200)
        assert coverage < 0.5, f"Uniform noise should have low coverage, got {coverage}"


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
