"""
Phase 2: Eastern Soul Engine
=============================
Generates authentic Eastern backing tracks, synchronizes beats to old recordings,
and creates 'Ghost Collaboration' voice models from heritage vocal stems.

Modules:
    raag_database      — Raag & Taal reference data
    prompt_generator    — AI music tool prompt builder
    sonauto_client      — Sonauto/Udio API client
    audio_to_midi       — WAV → MIDI conversion (Basic Pitch)
    raag_lock           — Raag-Lock constraint filter
    midi_to_audio       — MIDI → WAV rendering (FluidSynth)
    beat_sync           — DTW beat synchronizer
    taal_quantizer      — Taal-correct drum pattern generator
    rvc_training        — RVC voice model training
    rvc_inference       — Voice conversion inference
    simple_mixer        — Multi-track audio mixer
    pipeline            — Master pipeline orchestrator
"""

__version__ = "2.0.0"
__phase__ = "Eastern Soul Engine"
