#!/usr/bin/env python3
"""
raag_database.py — Raag & Taal Reference Database
====================================================
Central data store for all Raag note sets, mood descriptors, instrument pairings,
and Taal (rhythm cycle) definitions used across Phase 2.

This module is the mathematical backbone of the Raag-Lock filter: every AI-generated
note is checked against the `notes` list defined here.

Chromatic Pitch Classes:
    C=0, C#=1, D=2, D#=3, E=4, F=5, F#=6, G=7, G#=8, A=9, A#=10, B=11

Usage:
    from raag_database import RAAG_DATABASE, TAAL_DATABASE, get_raag_info
    raag = RAAG_DATABASE['Bhairavi']
    print(raag['notes'])  # [0, 1, 3, 5, 7, 8, 10]
"""

import json
import os

# ─── Raag Database ────────────────────────────────────────────────────────────

RAAG_DATABASE = {
    'Bhairavi': {
        'notes': [0, 1, 3, 5, 7, 8, 10],  # C Db Eb F G Ab Bb
        'mood': 'melancholic, devotional, longing, farewell',
        'time': 'morning or end of concert',
        'instruments': 'sarangi, bansuri, tabla, harmonium',
        'tempo_range': (40, 80),
        'taal': 'Teentaal or Dadra',
        'sonauto_keywords': (
            'sad indian classical, melancholic sitar melody, '
            'morning raga, devotional tabla rhythm, sarangi violin, '
            'minor scale indian'
        ),
        'similar_western': 'Phrygian mode, Spanish Flamenco minor',
        'thaat': 'Bhairavi',
        'vadi': 'Ma',
        'samvadi': 'Sa',
    },
    'Yaman': {
        'notes': [0, 2, 4, 6, 7, 9, 11],  # C D E F# G A B (Lydian)
        'mood': 'romantic, peaceful, evening beauty, hope',
        'time': 'early evening (6pm-9pm)',
        'instruments': 'sitar, sarod, tabla, tanpura',
        'tempo_range': (60, 100),
        'taal': 'Teentaal',
        'sonauto_keywords': (
            'romantic indian evening, sitar melody lydian, '
            'hopeful raga, indian classical evening, peaceful tabla groove'
        ),
        'similar_western': 'Lydian mode, major scale with #4',
        'thaat': 'Kalyan',
        'vadi': 'Ga',
        'samvadi': 'Ni',
    },
    'Darbari': {
        'notes': [0, 2, 3, 5, 7, 8, 10],  # C D Eb F G Ab Bb
        'mood': 'serious, dignified, royal, deep, late night',
        'time': 'late night (after midnight)',
        'instruments': 'been, sarod, tabla, tanpura',
        'tempo_range': (30, 60),
        'taal': 'Ektaal',
        'sonauto_keywords': (
            'dark royal indian classical, deep night raga, '
            'slow dramatic tabla, dignified sarod, cinematic indian epic, '
            'dorian flat 6 ethnic'
        ),
        'similar_western': 'Aeolian b6 (natural minor with b6)',
        'thaat': 'Asavari',
        'vadi': 'Re',
        'samvadi': 'Pa',
    },
    'Kafi': {
        'notes': [0, 2, 3, 5, 7, 9, 10],  # C D Eb F G A Bb
        'mood': 'folk, romantic, Sufi, devotional, earthy',
        'time': 'midnight to dawn',
        'instruments': 'harmonium, dholak, sarangi, tabla',
        'tempo_range': (80, 130),
        'taal': 'Keherwa',
        'sonauto_keywords': (
            'sufi folk punjabi, romantic dholak rhythm, '
            'earthy harmonium, pakistani folk, qawwali beat, dorian mode ethnic'
        ),
        'similar_western': 'Dorian mode',
        'thaat': 'Kafi',
        'vadi': 'Pa',
        'samvadi': 'Sa',
    },
    'Bhairav': {
        'notes': [0, 1, 4, 5, 7, 8, 11],  # C Db E F G Ab B
        'mood': 'spiritual, majestic, divine, awakening',
        'time': 'dawn/early morning',
        'instruments': 'shehnai, bansuri, tabla, veena',
        'tempo_range': (40, 70),
        'taal': 'Jhaptaal or Teentaal',
        'sonauto_keywords': (
            'divine morning indian, majestic shehnai, '
            'spiritual dawn raga, flute meditative, slow awakening drums'
        ),
        'similar_western': 'Double harmonic / Byzantine scale',
        'thaat': 'Bhairav',
        'vadi': 'Dha',
        'samvadi': 'Re',
    },
}


# ─── Taal (Rhythm Cycle) Database ────────────────────────────────────────────

TAAL_DATABASE = {
    'Teentaal': {
        'beats': 16,
        'structure': '4+4+4+4',
        'feel': 'steady, classical',
        'vibhags': [1, 5, 9, 13],  # Beat positions where sections start
        'sam': 1,                   # The strongest beat (downbeat)
        'khali': 9,                 # The "empty" beat
    },
    'Keherwa': {
        'beats': 8,
        'structure': '4+4',
        'feel': 'folk, accessible',
        'vibhags': [1, 5],
        'sam': 1,
        'khali': 5,
    },
    'Dadra': {
        'beats': 6,
        'structure': '3+3',
        'feel': '6/8 waltz-like, romantic',
        'vibhags': [1, 4],
        'sam': 1,
        'khali': 4,
    },
    'Jhaptaal': {
        'beats': 10,
        'structure': '2+3+2+3',
        'feel': 'asymmetric, complex',
        'vibhags': [1, 3, 6, 8],
        'sam': 1,
        'khali': 6,
    },
    'Ektaal': {
        'beats': 12,
        'structure': '2+2+2+2+2+2',
        'feel': 'meditative, slow',
        'vibhags': [1, 3, 5, 7, 9, 11],
        'sam': 1,
        'khali': 5,
    },
    'Rupak': {
        'beats': 7,
        'structure': '3+2+2',
        'feel': 'light, graceful, 7/8',
        'vibhags': [1, 4, 6],
        'sam': 1,
        'khali': 1,  # Rupak: Sam IS the Khali (unique)
    },
}


# ─── Note Name Helpers ────────────────────────────────────────────────────────

CHROMATIC_NAMES = [
    'C', 'C#', 'D', 'D#', 'E', 'F',
    'F#', 'G', 'G#', 'A', 'A#', 'B'
]

SARGAM_NAMES = [
    'Sa', 'Re♭', 'Re', 'Ga♭', 'Ga', 'Ma',
    'Ma#', 'Pa', 'Dha♭', 'Dha', 'Ni♭', 'Ni'
]


def chromatic_to_name(pitch_class, use_sargam=False):
    """Convert chromatic pitch class (0-11) to note name."""
    names = SARGAM_NAMES if use_sargam else CHROMATIC_NAMES
    return names[pitch_class % 12]


def get_raag_note_names(raag_name, use_sargam=True):
    """Get human-readable note names for a Raag."""
    raag = RAAG_DATABASE.get(raag_name)
    if not raag:
        raise ValueError(f'Raag not found: {raag_name}')
    return [chromatic_to_name(n, use_sargam) for n in raag['notes']]


# ─── Interop with Phase 1 ────────────────────────────────────────────────────

def get_raag_info(raag_name):
    """
    Retrieve Raag info, merging Phase 2 database with Phase 1 config if available.

    Returns combined dict with all available fields from both phases.
    """
    info = {}

    # Try Phase 1 config first
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'phase1_extraction_lab',
        'config', 'raag_maps.json'
    )
    if os.path.exists(config_path):
        with open(config_path) as f:
            phase1_data = json.load(f)
        raag_key = raag_name.lower().replace(' ', '_')
        if raag_key in phase1_data.get('raags', {}):
            info.update(phase1_data['raags'][raag_key])

    # Overlay Phase 2 data (takes priority for shared keys)
    if raag_name in RAAG_DATABASE:
        info.update(RAAG_DATABASE[raag_name])

    if not info:
        raise ValueError(f'Raag not found in either database: {raag_name}')

    return info


def list_all_raags():
    """List all available Raag names from Phase 2 database."""
    return list(RAAG_DATABASE.keys())


def list_all_taals():
    """List all available Taal names."""
    return list(TAAL_DATABASE.keys())


# ─── CLI Preview ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('🎵 VIRASAT.AI — Phase 2 Raag Database')
    print('=' * 55)
    for name, raag in RAAG_DATABASE.items():
        notes = get_raag_note_names(name, use_sargam=True)
        print(f'\n  Raag {name}')
        print(f'    Notes:  {", ".join(notes)}')
        print(f'    Mood:   {raag["mood"]}')
        print(f'    Time:   {raag["time"]}')
        print(f'    Tempo:  {raag["tempo_range"][0]}-{raag["tempo_range"][1]} BPM')
        print(f'    Western: {raag["similar_western"]}')

    print(f'\n\n🥁 Taal Database')
    print('=' * 55)
    for name, taal in TAAL_DATABASE.items():
        print(f'  {name:12s} | {taal["beats"]:2d} beats | {taal["structure"]:12s} | {taal["feel"]}')
