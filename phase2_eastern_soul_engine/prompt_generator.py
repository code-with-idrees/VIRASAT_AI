#!/usr/bin/env python3
"""
prompt_generator.py — Raag-Aware AI Music Prompt Builder
==========================================================
Generates optimized text prompts for Sonauto / Udio that produce
culturally authentic Eastern backing tracks.

Key insight: AI music models don't understand 'Raag Bhairavi' — but they DO
understand 'sad sarangi melody, morning devotional tabla rhythm'. This module
bridges that vocabulary gap.

Usage:
    from prompt_generator import generate_eastern_prompt
    config = generate_eastern_prompt('Bhairavi', 'Keherwa', style='coke_studio')
    print(config['prompt'])

    # Or from CLI:
    python prompt_generator.py --raag Bhairavi --taal Keherwa --style coke_studio
"""

import argparse
import json
import os
import sys

# Add parent directory for imports when running standalone
sys.path.insert(0, os.path.dirname(__file__))

from raag_database import RAAG_DATABASE, TAAL_DATABASE


# ─── Style Presets ────────────────────────────────────────────────────────────

STYLE_MAP = {
    'coke_studio': (
        'Coke Studio Pakistan production, modern studio live session, '
        'warm mixing, contemporary South Asian pop, radio-ready production'
    ),
    'lo_fi': (
        'lo-fi chill beats, vinyl crackle, bedroom production, '
        'nostalgic warm sound, tape saturation, mellow vibes'
    ),
    'cinematic': (
        'Hans Zimmer cinematic epic, orchestral South Asian, '
        'emotional film score, building tension, sweeping strings'
    ),
    'sufi_rock': (
        'Junoon-style sufi rock, electric guitar meets tabla, '
        'psychedelic Eastern fusion, powerful vocals, arena rock energy'
    ),
    'trap': (
        'trap hi-hats 808 bass, Eastern samples chopped, dark modern trap, '
        'South Asian drill music, heavy bass, rattling hi-hats'
    ),
}


# ─── Prompt Generator ────────────────────────────────────────────────────────

def generate_eastern_prompt(raag_name, taal_name, style='coke_studio',
                            era='2026', bpm=None):
    """
    Generate an optimized text prompt for Sonauto/Udio that produces
    culturally authentic Eastern backing tracks.

    Args:
        raag_name:  Name of Raag from RAAG_DATABASE (e.g. 'Bhairavi')
        taal_name:  Name of Taal from TAAL_DATABASE (e.g. 'Keherwa')
        style:      One of: coke_studio, lo_fi, cinematic, sufi_rock, trap
        era:        Production era for sound quality reference
        bpm:        Override BPM (auto-calculated from Raag tempo range if None)

    Returns:
        dict with keys:
            prompt   — The full text prompt string
            raag     — Raag name
            taal     — Taal name
            bpm      — Target BPM
            notes    — Allowed chromatic pitch classes
            style    — Style preset used
            metadata — Additional context for downstream modules

    Raises:
        ValueError: If raag_name or taal_name not found in databases
    """
    raag = RAAG_DATABASE.get(raag_name)
    taal = TAAL_DATABASE.get(taal_name)

    if not raag:
        available = ', '.join(RAAG_DATABASE.keys())
        raise ValueError(f'Unknown Raag: {raag_name}. Available: {available}')
    if not taal:
        available = ', '.join(TAAL_DATABASE.keys())
        raise ValueError(f'Unknown Taal: {taal_name}. Available: {available}')

    # Calculate target BPM from Raag tempo range if not specified
    target_bpm = bpm if bpm else (raag['tempo_range'][0] + raag['tempo_range'][1]) // 2

    # Get style description
    style_desc = STYLE_MAP.get(style, STYLE_MAP['coke_studio'])

    # Build the prompt — describe the SOUND, not the theory
    prompt = (
        f'Instrumental backing track, {target_bpm} BPM, '
        f'{taal["beats"]}-beat cycle.\n'
        f'Mood: {raag["mood"]}.\n'
        f'Instruments: {raag["instruments"]}.\n'
        f'Style: {style_desc}.\n'
        f'Keywords: {raag["sonauto_keywords"]}.\n'
        f'NO VOCALS. NO WESTERN DRUMS. Tabla only for percussion.\n'
        f'Melodic feel similar to {raag["similar_western"]} '
        f'but with Indian ornaments and gamakas.\n'
        f'High quality studio recording, {era} production standard.\n'
        f'Duration: 120 seconds. Starts soft, builds to climax at 60 seconds.'
    )

    return {
        'prompt': prompt,
        'raag': raag_name,
        'taal': taal_name,
        'bpm': target_bpm,
        'notes': raag['notes'],
        'style': style,
        'metadata': {
            'mood': raag['mood'],
            'instruments': raag['instruments'],
            'tempo_range': raag['tempo_range'],
            'similar_western': raag['similar_western'],
            'taal_beats': taal['beats'],
            'taal_structure': taal['structure'],
        }
    }


def generate_batch_prompts(raag_names=None, taal_names=None,
                           styles=None):
    """
    Generate prompts for multiple Raag × Taal × Style combinations.

    Args:
        raag_names: List of Raag names (default: all)
        taal_names: List of Taal names (default: use each Raag's default)
        styles:     List of style presets (default: all)

    Returns:
        list of prompt config dicts
    """
    if raag_names is None:
        raag_names = list(RAAG_DATABASE.keys())
    if styles is None:
        styles = list(STYLE_MAP.keys())

    prompts = []

    for raag_name in raag_names:
        raag = RAAG_DATABASE[raag_name]

        # Use specified taals or fall back to Raag's default
        if taal_names:
            taals_to_use = taal_names
        else:
            # Extract first taal from the Raag's default
            default_taal = raag['taal'].split(' or ')[0].strip()
            taals_to_use = [default_taal]

        for taal_name in taals_to_use:
            for style in styles:
                try:
                    config = generate_eastern_prompt(raag_name, taal_name, style)
                    prompts.append(config)
                except ValueError as e:
                    print(f'  ⚠ Skipped: {e}')

    return prompts


def save_prompt_library(prompts, output_path='prompt_library.json'):
    """Save a list of prompt configs to JSON file."""
    # Convert tuples to lists for JSON serialization
    serializable = []
    for p in prompts:
        entry = dict(p)
        if 'metadata' in entry and 'tempo_range' in entry['metadata']:
            entry['metadata']['tempo_range'] = list(entry['metadata']['tempo_range'])
        serializable.append(entry)

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f'💾 Saved {len(serializable)} prompts to {output_path}')
    return output_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='VIRASAT.AI — Eastern Music Prompt Generator'
    )
    parser.add_argument('--raag', default='Bhairavi',
                        help='Raag name (default: Bhairavi)')
    parser.add_argument('--taal', default='Keherwa',
                        help='Taal name (default: Keherwa)')
    parser.add_argument('--style', default='coke_studio',
                        choices=list(STYLE_MAP.keys()),
                        help='Style preset (default: coke_studio)')
    parser.add_argument('--bpm', type=int, default=None,
                        help='Override BPM (auto if not set)')
    parser.add_argument('--batch', action='store_true',
                        help='Generate all Raag × Style combinations')
    parser.add_argument('--save', default=None,
                        help='Save to JSON file')

    args = parser.parse_args()

    if args.batch:
        print('🎵 Generating batch prompt library...\n')
        prompts = generate_batch_prompts()
        print(f'\n✅ Generated {len(prompts)} prompts')
        if args.save:
            save_prompt_library(prompts, args.save)
    else:
        config = generate_eastern_prompt(args.raag, args.taal,
                                         style=args.style, bpm=args.bpm)
        print(f'🎵 Prompt for Raag {args.raag} + {args.taal} ({args.style}):')
        print(f'   BPM: {config["bpm"]}')
        print(f'   Notes: {config["notes"]}')
        print(f'\n{config["prompt"]}')

        if args.save:
            save_prompt_library([config], args.save)


if __name__ == '__main__':
    main()
