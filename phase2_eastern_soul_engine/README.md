# 🎵 Phase 2: Eastern Soul Engine

> *Where heritage meets 2026 production*

The Eastern Soul Engine transforms VIRASAT.AI from a restoration tool into a **creative engine** that generates authentic Eastern music using AI while respecting Raag theory.

---

## 🏗️ Architecture

```
Old Vocal (Phase 1) ──┐
                      ├──→ Raag-Lock Filter ──→ Ghost Collaboration ──→ New Song
AI-Generated Track ───┘
```

### Module Map

| Module | Purpose | Day |
|--------|---------|-----|
| `raag_database.py` | Raag & Taal reference data (5 Raags, 6 Taals) | 6 |
| `prompt_generator.py` | AI music prompt builder (5 style presets) | 6–7 |
| `sonauto_client.py` | Sonauto API client + local fallback synth | 7 |
| `audio_to_midi.py` | WAV → MIDI (Basic Pitch / librosa fallback) | 8 |
| `raag_lock.py` | **THE CORE** — Raag-Lock constraint filter | 8–9 |
| `midi_to_audio.py` | MIDI → WAV (FluidSynth / pretty_midi) | 9 |
| `beat_sync.py` | DTW beat synchronizer for tempo-drifting vocals | 10–11 |
| `taal_quantizer.py` | Taal-correct drum pattern generator | 11 |
| `rvc_training.py` | RVC voice model training wrapper | 12 |
| `rvc_inference.py` | Voice conversion + Ghost Collaboration pipeline | 12 |
| `simple_mixer.py` | Multi-track audio mixer with fades | 12 |
| `pipeline.py` | Master 7-step pipeline orchestrator | 12 |

---

## 🚀 Quick Start

### Quick Test (No GPU / API needed)
```bash
cd VIRASAT_AI/phase2_eastern_soul_engine
python pipeline.py --test
```

### Full Pipeline
```bash
python pipeline.py \
    --raag Bhairavi \
    --taal Keherwa \
    --bpm 90 \
    --style coke_studio \
    --vocal ../virasat_vocals/Ghulam_Ali_Chupke_Chupke_Raat_Din/vocals.wav \
    --model models/ghulam_ali_v1.pth
```

### Individual Modules
```bash
# Generate prompts
python prompt_generator.py --raag Yaman --taal Teentaal --style cinematic

# Raag-Lock a MIDI file
python raag_lock.py melody.mid --raag Bhairavi --strategy nearest

# Generate Taal drums
python taal_quantizer.py --taal Keherwa --bpm 90 --duration 60

# DTW sync
python beat_sync.py --vocal vocal.wav --beat drums.wav --preview

# Mix tracks
python simple_mixer.py vocal.wav:1.0 drums.wav:0.7 sitar.wav:0.5 --fade
```

---

## 🔒 Raag-Lock Technology

The secret weapon: a mathematical constraint filter that prevents AI-generated instruments from playing notes outside the target Raag.

| Strategy | What it Does | When to Use |
|----------|-------------|-------------|
| `nearest` | Moves note to closest allowed pitch | **Default** — sounds most natural |
| `delete` | Removes out-of-Raag notes | Sparse arrangements |
| `octave` | Same as nearest (legacy) | Compatibility |

If modification rate exceeds 40%, the generated track is too Western — re-prompt.

---

## 🥁 Supported Taals

| Taal | Beats | Structure | Feel |
|------|-------|-----------|------|
| Teentaal | 16 | 4+4+4+4 | Steady, classical |
| Keherwa | 8 | 4+4 | Folk, accessible |
| Dadra | 6 | 3+3 | 6/8, romantic |
| Jhaptaal | 10 | 2+3+2+3 | Asymmetric, complex |
| Ektaal | 12 | 2+2+2+2+2+2 | Meditative, slow |
| Rupak | 7 | 3+2+2 | Light, graceful |

---

## 📦 Dependencies

```
pip install librosa soundfile pretty_midi basic-pitch
pip install midi2audio numpy scipy requests
```

For RVC voice models (GPU required):
```
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git
```

---

## 📂 Output Structure

```
outputs/
├── pipeline_run/
│   ├── Bhairavi_coke_studio_*.wav    # Generated backing
│   ├── *_converted.mid                # MIDI conversion
│   ├── *_raaglock_Bhairavi.mid       # Raag-locked MIDI
│   ├── *_rendered.wav                 # Rendered audio
│   ├── drums.mid                      # Taal pattern
│   ├── drums_synced.wav               # DTW-synced drums
│   ├── ghost_vocal.wav                # Voice-converted melody
│   └── FINAL_Bhairavi_Keherwa_90bpm.wav  # Final mix
```

---

*Phase 2 of the 45-Day VIRASAT.AI Build Plan*
