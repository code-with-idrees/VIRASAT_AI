#!/usr/bin/env python3
"""
test_quality_metrics.py — Tests for Quality Metrics Calculator
===============================================================
Validates SDR/SIR/SAR computation and Virasat Score formula.
"""

import sys
import os
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestNormalization:
    """Test the normalize function."""

    def test_normalize_midpoint(self):
        from phase1_extraction_lab.scripts.quality_metrics import normalize
        assert normalize(5, 0, 10) == 50.0

    def test_normalize_min(self):
        from phase1_extraction_lab.scripts.quality_metrics import normalize
        assert normalize(0, 0, 10) == 0.0

    def test_normalize_max(self):
        from phase1_extraction_lab.scripts.quality_metrics import normalize
        assert normalize(10, 0, 10) == 100.0

    def test_normalize_below_min(self):
        from phase1_extraction_lab.scripts.quality_metrics import normalize
        assert normalize(-5, 0, 10) == 0.0  # Clipped to 0

    def test_normalize_above_max(self):
        from phase1_extraction_lab.scripts.quality_metrics import normalize
        assert normalize(15, 0, 10) == 100.0  # Clipped to 100


class TestVirasatScore:
    """Test the Virasat Score computation."""

    def test_perfect_score(self):
        from phase1_extraction_lab.scripts.quality_metrics import compute_virasat_score
        # Max values for all metrics
        score = compute_virasat_score(sdr=20, sir=30, sar=25, snr=60)
        assert score == 100.0

    def test_zero_score(self):
        from phase1_extraction_lab.scripts.quality_metrics import compute_virasat_score
        score = compute_virasat_score(sdr=0, sir=0, sar=0, snr=0)
        assert score == 0.0

    def test_weight_distribution(self):
        """SIR has highest weight (0.40), so high SIR should dominate."""
        from phase1_extraction_lab.scripts.quality_metrics import compute_virasat_score
        # Only SIR is high
        score_sir = compute_virasat_score(sdr=0, sir=30, sar=0, snr=0)
        # Only SDR is high
        score_sdr = compute_virasat_score(sdr=20, sir=0, sar=0, snr=0)
        # SIR should contribute more
        assert score_sir > score_sdr

    def test_win_condition_threshold(self):
        """Win condition requires Virasat Score > 70."""
        from phase1_extraction_lab.scripts.quality_metrics import compute_virasat_score
        # Realistic high-quality extraction values that yield > 70
        score = compute_virasat_score(sdr=15, sir=25, sar=20, snr=50)
        assert score > 70


class TestClassification:
    """Test metric classification."""

    def test_classify_excellent_sdr(self):
        from phase1_extraction_lab.scripts.quality_metrics import classify_metric
        assert classify_metric("sdr", 15) == "Excellent"

    def test_classify_poor_sir(self):
        from phase1_extraction_lab.scripts.quality_metrics import classify_metric
        assert classify_metric("sir", 5) == "Poor"

    def test_classify_virasat_gold(self):
        from phase1_extraction_lab.scripts.quality_metrics import classify_virasat_score
        result = classify_virasat_score(95)
        assert "Gold" in result

    def test_classify_virasat_retry(self):
        from phase1_extraction_lab.scripts.quality_metrics import classify_virasat_score
        result = classify_virasat_score(30)
        assert "re-separation" in result


class TestSNR:
    """Test SNR computation."""

    def test_snr_clean_signal(self):
        from phase1_extraction_lab.scripts.quality_metrics import compute_snr
        # Pure sine wave with half silence so the noise estimator can find a noise floor
        t = np.linspace(0, 1, 44100)
        signal = np.sin(2 * np.pi * 440 * t) * (t < 0.5)
        snr = compute_snr(signal)
        assert snr > 20  # Should be high for clean signal

    def test_snr_noisy_signal(self):
        from phase1_extraction_lab.scripts.quality_metrics import compute_snr
        # Mostly noise
        noise = np.random.randn(44100) * 0.5
        snr = compute_snr(noise)
        assert snr < 20  # Should be low for noise


class TestBSSMetrics:
    """Test BSS evaluation (SDR/SIR/SAR)."""

    def test_identical_signals(self):
        """Identical reference and estimate should give high SDR."""
        from phase1_extraction_lab.scripts.quality_metrics import compute_bss_metrics
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        result = compute_bss_metrics(signal, signal)
        assert result["sdr_db"] > 20  # Should be very high

    def test_different_signals(self):
        """Completely different signals should give low SDR."""
        from phase1_extraction_lab.scripts.quality_metrics import compute_bss_metrics
        t = np.linspace(0, 1, 44100)
        ref = np.sin(2 * np.pi * 440 * t)
        est = np.sin(2 * np.pi * 880 * t)  # Different frequency
        result = compute_bss_metrics(ref, est)
        assert result["sdr_db"] < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
