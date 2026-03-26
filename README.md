# 🎵 Virasat.AI — Heritage Music Revival Engine

> *"Virasat"* (وراثت) means **Heritage** in Urdu. This AI-powered platform revives and modernizes Pakistani heritage music from the 1930s to 2000s.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Demucs](https://img.shields.io/badge/Demucs-v4_(HT)-green.svg)](https://github.com/facebookresearch/demucs)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🏗️ Architecture — The Two-Wing Engine

| Wing | Purpose | Technology |
|------|---------|------------|
| **Wing 1 — Heritage Revival** | Restores and modernizes 1930s–2000s recordings | Demucs v4, spectral analysis, noise reduction |
| **Wing 2 — Generative Soul** | Creates brand-new Eastern songs using AI | Raag-Lock Technology, voice synthesis |

### 🔐 Secret Weapon: Raag-Lock Technology

A mathematical constraint system that prevents generated instruments from playing frequencies outside the original Raag's allowed note set. Uses Pitch Class Profiles + cosine similarity for Raag detection, then applies frequency-domain filtering to lock all output to permitted notes.

---

## 🎤 Heritage Music Catalog

### Era 1: The Golden Film Era (1930s–1960s)
- **Noor Jehan** — *Malika-e-Tarannum*, 10,000+ songs
- **Ahmed Rushdi** — Pioneer of Filmi Pop ("Ko Ko Korina")
- **Runa Laila** — Queen of Pop (Pakistan & Bangladesh)
- **Munir Hussain** — PTV's golden voice

### Era 2: Classical & Ghazal Golden Age (1960s–1990s)
- **Mehdi Hassan** — *Shahenshah-e-Ghazal*
- **Ghulam Ali** — Patiala Gharana master
- **Nusrat Fateh Ali Khan** — *Shahanshah-e-Qawwali*, 30M+ albums
- **Farida Khanum** — Queen of Ghazal
- **Abida Parveen** — Queen of Sufi Music
- **Ustad Amanat Ali Khan** — Patiala Gharana legend

### Era 3: Pop Revolution (1980s–1990s)
- **Vital Signs** — "Dil Dil Pakistan"
- **Nazia & Zoheb Hassan** — Disco Pop, 65M+ records
- **Junoon** — Sufi Rock pioneers, 30M albums
- **Alamgir** — Pioneer of Urdu Pop
- **Awaz** — 90s pop/rock
- **Abrar-ul-Haq** — King of Bhangra Pop

### Era 4: Digital & Coke Studio (2000s–Present)
- **Strings** — Pop Rock calibration baseline
- **Noori** — Alt Rock
- **Atif Aslam** — Most commercially successful modern singer
- **Sajjad Ali** — Pop/Classical fusion

---

## 📊 Mathematical Foundation

All processing is backed by rigorous DSP mathematics:

| Component | Math | Purpose |
|-----------|------|---------|
| Signal Analysis | STFT, MFCC | Convert audio to analyzable representations |
| Quality Metrics | SDR, SIR, SAR | Objective separation quality measurement |
| Bleed Detection | Spectral energy ratios | Detect instrument leakage per frequency band |
| Raag Detection | Pitch Class Profile + cosine similarity | Identify the Raag of a recording |
| Raag-Lock | Frequency-domain constraint filtering | Lock generated audio to permitted Raag notes |
| Noise Reduction | Spectral subtraction, Wiener filter | Clean heritage recordings before processing |

### Custom Metric: Virasat Score

```
Virasat Score = 0.40×SIR + 0.30×SDR + 0.20×SAR + 0.10×SNR (normalized to 0-100)
```

---

## 🗂️ Project Structure

```
VIRASAT_AI/
├── phase1_extraction_lab/     # Days 1-5: Stem separation & quality analysis
├── phase2_eastern_soul_engine/# Days 6-12: Raag-Lock & voice synthesis
├── phase3_video_pipeline/     # Days 13-18: AI video generation
├── phase4_dashboard/          # Days 19-25: Streamlit UI
├── phase5_launch/             # Days 26-30: Pitch & outreach
├── phase6_full_product/       # Days 31-45: Full platform
├── utils/                     # Shared math & audio utilities
├── tests/                     # Test suite
└── docs/                      # Reference documents
```

---

## 🚀 Quick Start

### Option A: Google Colab (Recommended — GPU)

1. Open `phase1_extraction_lab/notebooks/01_demucs_stem_separation.ipynb` in Google Colab
2. Set runtime to **T4 GPU**
3. Run all cells

### Option B: Local Setup (CPU)

```bash
# Clone and enter project
cd VIRASAT_AI

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run stem separation
python3 phase1_extraction_lab/scripts/stem_separator.py \
  --input phase1_extraction_lab/data/raw/ \
  --output phase1_extraction_lab/data/stems/ \
  --model htdemucs_ft
```

---

## 📅 35-Day Build Timeline

| Phase | Days | Name | Status |
|-------|------|------|--------|
| 1 | 1-5 | **Extraction Lab** — Stem separation & quality | ✅ Complete |
| 2 | 6-12 | **Eastern Soul Engine** — Raag-Lock & synthesis | ✅ Complete |
| 3 | 13-18 | **Video Pipeline** — AI video generation | ⏳ Planned |
| 4 | 19-25 | **Dashboard** — Streamlit UI & heritage slider | ⏳ Planned |
| 5 | 26-30 | **Launch** — Pitch deck & Coke Studio outreach | ⏳ Planned |
| 6 | 31-45 | **Full Product** — API & deployment | ⏳ Planned |

---

## 📜 License

For Demonstration Purposes Only. Heritage audio content used under fair use for research and development.

---

*Built with ❤️ for Pakistani music heritage.*
