#!/usr/bin/env python3
"""
test_math_utils.py — Tests for DSP Math Utilities
===================================================
Validates frequency conversions, PCP, cosine similarity, and normalization.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestFrequencyConversions:
    """Test Hz ↔ pitch class conversions."""

    def test_tonic_is_sa(self):
        from utils.math_utils import hz_to_pitch_class
        assert hz_to_pitch_class(261.63, 261.63) == 0  # Sa = 0

    def test_octave_maps_to_same(self):
        from utils.math_utils import hz_to_pitch_class
        # Double frequency = same pitch class
        assert hz_to_pitch_class(523.26, 261.63) == 0  # Sa one octave up

    def test_pa_is_7(self):
        from utils.math_utils import hz_to_pitch_class
        # Pa (perfect fifth) = 7 semitones above Sa
        pa_hz = 261.63 * (2 ** (7/12))
        assert hz_to_pitch_class(pa_hz, 261.63) == 7

    def test_invalid_frequency(self):
        from utils.math_utils import hz_to_pitch_class
        assert hz_to_pitch_class(0, 261.63) == -1
        assert hz_to_pitch_class(-100, 261.63) == -1

    def test_pitch_class_roundtrip(self):
        from utils.math_utils import hz_to_pitch_class, pitch_class_to_hz
        for pc in range(12):
            hz = pitch_class_to_hz(pc)
            recovered = hz_to_pitch_class(hz)
            assert recovered == pc


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_identical_vectors(self):
        from utils.math_utils import cosine_similarity
        a = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
        assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        from utils.math_utils import cosine_similarity
        a = np.array([1, 0, 0, 0])
        b = np.array([0, 1, 0, 0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector(self):
        from utils.math_utils import cosine_similarity
        a = np.array([1, 2, 3])
        b = np.zeros(3)
        assert cosine_similarity(a, b) == 0.0

    def test_similar_pcps(self):
        """Similar Raag profiles should have high similarity."""
        from utils.math_utils import cosine_similarity
        # Raag Yaman = Sa Re Ga Ma# Pa Dha Ni
        yaman = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1])
        # Slightly different (one note off)
        similar = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        sim = cosine_similarity(yaman, similar)
        assert sim > 0.8  # Should be high


class TestNormalization:
    """Test normalization functions."""

    def test_min_max_basic(self):
        from utils.math_utils import normalize_min_max
        assert normalize_min_max(5, 0, 10, 100) == 50.0

    def test_min_max_same_range(self):
        from utils.math_utils import normalize_min_max
        assert normalize_min_max(5, 5, 5, 100) == 0.0

    def test_db_roundtrip(self):
        from utils.math_utils import db_to_linear, linear_to_db
        original_db = 20.0
        linear = db_to_linear(original_db)
        recovered = linear_to_db(linear)
        assert recovered == pytest.approx(original_db, abs=0.01)


class TestKLDivergence:
    """Test KL divergence."""

    def test_identical_distributions(self):
        from utils.math_utils import kl_divergence
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_different_distributions(self):
        from utils.math_utils import kl_divergence
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        assert kl_divergence(p, q) > 0  # Should be positive


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
