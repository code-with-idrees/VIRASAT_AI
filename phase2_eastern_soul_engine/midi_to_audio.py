#!/usr/bin/env python3
"""
midi_to_audio.py — MIDI to WAV Renderer
==========================================
Renders MIDI files to WAV audio using FluidSynth.
Requires FluidSynth installed on the system and a SoundFont (.sf2) file.

Setup:
    # Linux/Colab:  apt-get install -y fluidsynth
    # macOS:        brew install fluid-synth
    # Python:       pip install midi2audio

Usage:
    from midi_to_audio import midi_to_wav
    wav_path = midi_to_wav('melody_raaglock_Bhairavi.mid')
"""

import os
import sys
import shutil

try:
    from midi2audio import FluidSynth
    MIDI2AUDIO_AVAILABLE = True
except ImportError:
    MIDI2AUDIO_AVAILABLE = False

try:
    import pretty_midi
    import numpy as np
    import soundfile as sf
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False


# ─── Default SoundFont Paths ─────────────────────────────────────────────────

SOUNDFONT_SEARCH_PATHS = [
    '/usr/share/sounds/sf2/FluidR3_GM.sf2',
    '/usr/share/sounds/sf2/FluidR3_GS.sf2',
    '/usr/share/soundfonts/FluidR3_GM.sf2',
    '/usr/share/soundfonts/default.sf2',
    '/usr/local/share/fluidsynth/GeneralUser_GS.sf2',
    os.path.expanduser('~/soundfonts/FluidR3_GM.sf2'),
]


def find_soundfont():
    """Search for an available SoundFont file on the system."""
    for path in SOUNDFONT_SEARCH_PATHS:
        if os.path.exists(path):
            return path
    return None


# ─── MIDI to WAV ─────────────────────────────────────────────────────────────

def midi_to_wav(midi_path, soundfont_path=None, output_path=None,
                sample_rate=44100):
    """
    Render a MIDI file to WAV audio using FluidSynth.

    Args:
        midi_path:      Path to input .mid file
        soundfont_path: Path to .sf2 SoundFont file (auto-detected if None)
        output_path:    Where to save WAV (auto-named if None)
        sample_rate:    Output sample rate (default: 44100)

    Returns:
        str: Path to the rendered WAV file
    """
    if not os.path.exists(midi_path):
        raise FileNotFoundError(f'MIDI file not found: {midi_path}')

    if output_path is None:
        output_path = midi_path.rsplit('.', 1)[0] + '_rendered.wav'

    # Try FluidSynth first
    if MIDI2AUDIO_AVAILABLE:
        if soundfont_path is None:
            soundfont_path = find_soundfont()

        if soundfont_path and os.path.exists(soundfont_path):
            return _render_with_fluidsynth(
                midi_path, soundfont_path, output_path, sample_rate
            )
        else:
            print('⚠ No SoundFont found. Using sine-wave fallback.')
            print('  Install FluidSynth: apt-get install -y fluidsynth')

    # Fallback: render using pretty_midi's built-in synthesizer
    if PRETTY_MIDI_AVAILABLE:
        return _render_with_pretty_midi(midi_path, output_path, sample_rate)

    raise ImportError(
        'Neither midi2audio nor pretty_midi available. '
        'Install: pip install midi2audio pretty_midi'
    )


def _render_with_fluidsynth(midi_path, soundfont_path, output_path,
                             sample_rate):
    """Render MIDI using FluidSynth (best quality)."""
    print(f'🎹 Rendering MIDI with FluidSynth...')
    print(f'   SoundFont: {os.path.basename(soundfont_path)}')

    fs = FluidSynth(sound_font=soundfont_path, sample_rate=sample_rate)
    fs.midi_to_audio(midi_path, output_path)

    print(f'   ✅ Rendered: {output_path}')
    return output_path


def _render_with_pretty_midi(midi_path, output_path, sample_rate):
    """
    Fallback: render MIDI using pretty_midi's fluidsynth method,
    or synthesize with sine waves if FluidSynth binary is missing.
    """
    print(f'🔧 Rendering MIDI with pretty_midi fallback...')

    midi = pretty_midi.PrettyMIDI(midi_path)

    # Try pretty_midi's built-in synthesize (uses sine waves)
    audio = midi.synthesize(fs=sample_rate)

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.85

    sf.write(output_path, audio, sample_rate)
    print(f'   ✅ Rendered (sine-wave synth): {output_path}')
    return output_path


# ─── Batch Rendering ─────────────────────────────────────────────────────────

def batch_render(midi_dir, output_dir=None, soundfont_path=None):
    """
    Render all MIDI files in a directory to WAV.

    Args:
        midi_dir:       Directory containing .mid files
        output_dir:     Output directory (same as midi_dir if None)
        soundfont_path: SoundFont to use

    Returns:
        list of rendered WAV paths
    """
    if output_dir is None:
        output_dir = midi_dir
    os.makedirs(output_dir, exist_ok=True)

    rendered = []
    for fname in sorted(os.listdir(midi_dir)):
        if fname.endswith('.mid') or fname.endswith('.midi'):
            midi_path = os.path.join(midi_dir, fname)
            wav_name = fname.rsplit('.', 1)[0] + '_rendered.wav'
            output_path = os.path.join(output_dir, wav_name)
            try:
                result = midi_to_wav(midi_path, soundfont_path, output_path)
                rendered.append(result)
            except Exception as e:
                print(f'   ❌ Error rendering {fname}: {e}')

    print(f'\n📊 Batch render complete: {len(rendered)} files')
    return rendered


# ─── Utility: Full Pipeline Shortcut ─────────────────────────────────────────

def full_raag_lock_pipeline(raw_wav_path, raag_name, strategy='nearest'):
    """
    Convenience function: WAV → MIDI → Raag-Lock → Rendered WAV.

    Args:
        raw_wav_path: Path to raw WAV (e.g. Sonauto output)
        raag_name:    Target Raag for filtering
        strategy:     Raag-Lock strategy ('nearest' recommended)

    Returns:
        str: Path to final Raag-locked WAV
    """
    from audio_to_midi import wav_to_midi
    from raag_lock import apply_raag_lock

    print(f'\n🔄 Full Raag-Lock Pipeline: {raag_name}')
    print(f'   Input: {raw_wav_path}')

    # Step 1: WAV → MIDI
    midi_path = wav_to_midi(raw_wav_path)

    # Step 2: Apply Raag-Lock
    locked = apply_raag_lock(midi_path, raag_name, strategy=strategy)

    # Step 3: Render back to WAV
    final_wav = midi_to_wav(locked['output_path'])

    print(f'   ✅ Final output: {final_wav}')
    return final_wav


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='VIRASAT.AI — MIDI to WAV Renderer'
    )
    parser.add_argument('input', help='Input MIDI file or directory')
    parser.add_argument('--output', default=None)
    parser.add_argument('--soundfont', default=None, help='SoundFont .sf2 path')
    parser.add_argument('--sr', type=int, default=44100, help='Sample rate')
    args = parser.parse_args()

    if os.path.isdir(args.input):
        batch_render(args.input, args.output, args.soundfont)
    else:
        midi_to_wav(args.input, args.soundfont, args.output, args.sr)
