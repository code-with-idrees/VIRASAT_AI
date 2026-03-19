# 🧪 Phase 1 — The Extraction Lab (Days 1-5)

> Stem separation, quality analysis, and noise reduction for heritage Pakistani audio.

## 🎯 Objective

Extract **clean, isolated vocal WAV files** from heritage Pakistani music recordings using Demucs v4, with mathematical verification of quality.

## 📋 Days 1-2 Quick Start

### Step 1: Environment Setup (Google Colab)

```python
# Cell 1 — Install dependencies
!pip install demucs librosa soundfile mir_eval matplotlib yt-dlp rich
```

### Step 2: Download Test Songs

```bash
python scripts/download_songs.py --search "Ranjish Hi Sahi Mehdi Hassan" --artist "Mehdi Hassan"
python scripts/download_songs.py --search "Dil Dil Pakistan Vital Signs" --artist "Vital Signs"
```

### Step 3: Run Stem Separation

```bash
# Fine-tuned model (best for old recordings)
python scripts/stem_separator.py --input data/raw/ --model htdemucs_ft --output data/stems/

# Comparison mode (both models)
python scripts/stem_separator.py --input data/raw/ --models htdemucs htdemucs_ft
```

### Step 4: Quality Analysis

```bash
# Bleed detection
python scripts/bleed_detector.py --input data/stems/ --report

# Quality metrics (SDR/SIR/SAR/Virasat Score)
python scripts/quality_metrics.py --estimated data/stems/htdemucs_ft/song_name/vocals.wav

# Noise estimation (for old recordings)
python scripts/noise_estimator.py --input data/raw/
```

### Step 5: Raag Detection

```bash
python scripts/raag_classifier.py --input data/stems/htdemucs_ft/song_name/vocals.wav --auto-tonic --top 3
```

## 📊 Win Conditions

| # | Condition | Target |
|---|-----------|--------|
| 1 | Clean vocals from ≥2 songs | No audible bleed |
| 2 | SIR > 15 dB | `quality_metrics.py` |
| 3 | SAR > 10 dB | `quality_metrics.py` |
| 4 | Both models tested | Comparison report |
| 5 | Virasat Score > 70 | `quality_metrics.py` |

## 📁 Directory Structure

```
phase1_extraction_lab/
├── scripts/           # Python scripts
│   ├── stem_separator.py
│   ├── bleed_detector.py
│   ├── quality_metrics.py
│   ├── noise_estimator.py
│   ├── audio_enhancer.py
│   ├── raag_classifier.py
│   ├── taal_detector.py
│   └── download_songs.py
├── notebooks/         # Colab notebooks
├── data/
│   ├── raw/           # Downloaded songs
│   ├── stems/         # Demucs output
│   ├── enhanced/      # Noise-reduced output
│   └── reports/       # Analysis reports
└── config/
    ├── raag_maps.json
    ├── instrument_profiles.json
    ├── test_songs.json
    └── quality_thresholds.json
```
