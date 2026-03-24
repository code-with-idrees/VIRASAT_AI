#!/usr/bin/env python3
"""
rvc_inference.py — Voice Conversion Inference & Ghost Collaboration
=====================================================================
Uses a trained RVC model to convert any audio into the target voice's style.
This is the 'Ghost Collaboration' engine — it makes a new melody sound like
it's being sung by a heritage legend.

Prerequisites:
    - Trained RVC .pth model (from rvc_training.py)
    - RVC-WebUI installed (for actual inference)
    - Input audio: hummed melody, synthesizer line, or MIDI-rendered melody

Usage:
    from rvc_inference import rvc_inference, create_new_classic
    output = rvc_inference('melody.wav', 'models/ghulam_ali_v1.pth')
"""

import os
import sys
import subprocess

try:
    import librosa
    import soundfile as sf
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


# ─── Configuration ────────────────────────────────────────────────────────────

RVC_DIR = os.environ.get(
    'RVC_DIR',
    os.path.expanduser('~/Retrieval-based-Voice-Conversion-WebUI')
)


# ─── Voice Conversion ────────────────────────────────────────────────────────

def rvc_inference(input_audio_path, model_path,
                  index_path=None, output_path=None,
                  f0_method='rmvpe', pitch_shift=0):
    """
    Convert an input audio file to the trained voice model's style.

    The input can be:
        - Your own hummed melody
        - A synthesizer/MIDI-rendered melody
        - Any vocal recording (pitch + rhythm preserved, timbre changed)

    Args:
        input_audio_path: WAV file with the melody to convert
        model_path:       Path to trained RVC .pth model
        index_path:       Optional .index file for better quality
        output_path:      Where to save output (auto-named if None)
        f0_method:        Pitch detection method:
                          'rmvpe' (best), 'crepe', 'dio', 'harvest'
        pitch_shift:      Adjust key in semitones (+2 = higher, -2 = lower)

    Returns:
        str: Path to the voice-converted WAV
    """
    if not os.path.exists(input_audio_path):
        raise FileNotFoundError(f'Input audio not found: {input_audio_path}')

    if output_path is None:
        base = input_audio_path.rsplit('.', 1)[0]
        output_path = f'{base}_vc_output.wav'

    # Check if model_path is a placeholder
    if model_path and model_path.endswith('.json'):
        print('⚠ Placeholder model detected. Using passthrough mode.')
        return _passthrough_with_effect(input_audio_path, output_path)

    # Try RVC inference
    infer_script = os.path.join(RVC_DIR, 'tools', 'infer_cli.py')
    if os.path.exists(infer_script) and model_path and os.path.exists(model_path):
        return _rvc_infer(
            infer_script, input_audio_path, model_path,
            index_path, output_path, f0_method, pitch_shift
        )
    else:
        print('⚠ RVC not installed or model not found. Using audio effect fallback.')
        return _passthrough_with_effect(input_audio_path, output_path)


def _rvc_infer(infer_script, input_path, model_path,
               index_path, output_path, f0_method, pitch_shift):
    """Run actual RVC inference via CLI."""
    print(f'🎤 Running RVC voice conversion...')
    print(f'   Input:  {os.path.basename(input_path)}')
    print(f'   Model:  {os.path.basename(model_path)}')
    print(f'   Method: {f0_method} | Pitch shift: {pitch_shift}')

    cmd = [
        sys.executable, infer_script,
        '--input_path', input_path,
        '--opt_path', output_path,
        '--model_path', model_path,
        '--index_path', index_path or '',
        '--f0method', f0_method,
        '--f0up_key', str(pitch_shift),
        '--index_rate', '0.88',
        '--protect', '0.33',
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=RVC_DIR,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max
        )

        if result.returncode == 0:
            print(f'   ✅ Voice conversion complete: {output_path}')
            return output_path
        else:
            print(f'   ❌ RVC inference failed: {result.stderr[:300]}')
            return _passthrough_with_effect(input_path, output_path)

    except subprocess.TimeoutExpired:
        print('   ⏱ RVC inference timeout.')
        return _passthrough_with_effect(input_path, output_path)
    except FileNotFoundError:
        print('   ❌ RVC CLI not found.')
        return _passthrough_with_effect(input_path, output_path)


def _passthrough_with_effect(input_path, output_path):
    """
    Fallback: apply simple audio effects to simulate voice conversion.
    NOT actual voice conversion — just for testing the pipeline flow.
    Adds subtle vibrato and warmth to make it slightly different.
    """
    if not AUDIO_AVAILABLE:
        # If no audio libs, just copy the file
        import shutil
        shutil.copy2(input_path, output_path)
        print(f'   📋 Passthrough (no processing): {output_path}')
        return output_path

    y, sr = librosa.load(input_path, sr=22050)

    # Add subtle vibrato effect
    t = np.arange(len(y)) / sr
    vibrato_freq = 5.0    # Hz
    vibrato_depth = 0.002  # seconds
    vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_freq * t)

    # Time-shift using interpolation to create vibrato
    indices = np.arange(len(y)) + (vibrato * sr).astype(int)
    indices = np.clip(indices.astype(int), 0, len(y) - 1)
    y_vibrato = y[indices]

    # Add warmth (subtle low-pass characteristic)
    # Simple moving average as a basic low-pass
    kernel_size = 3
    kernel = np.ones(kernel_size) / kernel_size
    y_warm = np.convolve(y_vibrato, kernel, mode='same')

    # Mix: 70% vibrato + 30% warm
    y_out = 0.7 * y_vibrato + 0.3 * y_warm

    # Normalize
    max_val = np.max(np.abs(y_out))
    if max_val > 0:
        y_out = y_out / max_val * 0.85

    sf.write(output_path, y_out, sr)
    print(f'   🔧 Fallback effect applied: {output_path}')
    print('   (Use RVC on GPU for real voice conversion)')
    return output_path


# ─── Ghost Collaboration: Full Pipeline ──────────────────────────────────────

def create_new_classic(raag_name, taal_name, bpm, duration=60,
                       voice_model_path=None,
                       melody_source=None,
                       output_dir='outputs/ghost_collab/'):
    """
    Full Ghost Collaboration pipeline:
    Generate melody → Raag-Lock → Voice Conversion → Output

    Creates a brand new song that never existed before,
    performed in the voice character of a heritage legend.

    Args:
        raag_name:        Target Raag
        taal_name:        Target Taal
        bpm:              Tempo
        duration:         Song duration in seconds
        voice_model_path: Path to RVC .pth model
        melody_source:    Path to seed melody WAV (generated if None)
        output_dir:       Where to save outputs

    Returns:
        str: Path to the ghost vocal WAV
    """
    sys.path.insert(0, os.path.dirname(__file__))
    from prompt_generator import generate_eastern_prompt
    from sonauto_client import generate_track_local_fallback
    from raag_lock import apply_raag_lock
    from midi_to_audio import midi_to_wav
    from audio_to_midi import wav_to_midi

    os.makedirs(output_dir, exist_ok=True)

    print(f'\n👻 Ghost Collaboration: {raag_name} + {taal_name}')
    print(f'   Creating a song that has never existed before...\n')

    # Step 1: Get or generate a seed melody
    if melody_source and os.path.exists(melody_source):
        print(f'[1/4] Using provided melody: {melody_source}')
        melody_wav = melody_source
    else:
        print('[1/4] Generating seed melody...')
        config = generate_eastern_prompt(raag_name, taal_name, bpm=bpm)
        melody_wav = generate_track_local_fallback(config, output_dir)

    # Step 2: Convert to MIDI and apply Raag-Lock
    print('[2/4] Applying Raag-Lock filter...')
    melody_midi = wav_to_midi(melody_wav)
    locked = apply_raag_lock(melody_midi, raag_name, strategy='nearest')

    # Step 3: Render locked MIDI back to audio
    print('[3/4] Rendering Raag-locked melody...')
    locked_wav = midi_to_wav(locked['output_path'])

    # Step 4: Voice conversion
    print('[4/4] Applying voice conversion...')
    ghost_output = os.path.join(output_dir, f'ghost_vocal_{raag_name}.wav')

    if voice_model_path:
        ghost_vocal = rvc_inference(
            locked_wav, voice_model_path,
            output_path=ghost_output,
        )
    else:
        # No model — use effect fallback
        ghost_vocal = _passthrough_with_effect(locked_wav, ghost_output)

    print(f'\n👻 Ghost Collaboration complete!')
    print(f'   Output: {ghost_vocal}')
    return ghost_vocal


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='VIRASAT.AI — RVC Voice Conversion & Ghost Collaboration'
    )
    subparsers = parser.add_subparsers(dest='command')

    # Convert command
    convert = subparsers.add_parser('convert',
                                     help='Convert audio to target voice')
    convert.add_argument('--input', required=True, help='Input WAV')
    convert.add_argument('--model', required=True, help='RVC .pth model')
    convert.add_argument('--output', default=None)
    convert.add_argument('--pitch', type=int, default=0,
                         help='Pitch shift in semitones')
    convert.add_argument('--method', default='rmvpe',
                         choices=['rmvpe', 'crepe', 'dio', 'harvest'])

    # Ghost command
    ghost = subparsers.add_parser('ghost',
                                   help='Create new Ghost Collaboration')
    ghost.add_argument('--raag', default='Bhairavi')
    ghost.add_argument('--taal', default='Keherwa')
    ghost.add_argument('--bpm', type=int, default=90)
    ghost.add_argument('--model', default=None, help='Voice model path')
    ghost.add_argument('--melody', default=None, help='Seed melody WAV')

    args = parser.parse_args()

    if args.command == 'convert':
        rvc_inference(args.input, args.model,
                      output_path=args.output,
                      pitch_shift=args.pitch,
                      f0_method=args.method)
    elif args.command == 'ghost':
        create_new_classic(args.raag, args.taal, args.bpm,
                          voice_model_path=args.model,
                          melody_source=args.melody)
    else:
        parser.print_help()
