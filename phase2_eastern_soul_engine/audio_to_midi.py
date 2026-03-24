#!/usr/bin/env python3
"""
audio_to_midi.py — WAV to MIDI Conversion
============================================
Converts audio WAV files to MIDI using Spotify's Basic Pitch model.
Basic Pitch works best on melodic instruments (sitar, sarangi, bansuri).
Less accurate on complex chords or percussion — skip tabla tracks.

Dependencies:
    pip install basic-pitch

Usage:
    from audio_to_midi import wav_to_midi
    midi_path = wav_to_midi('generated_tracks/Bhairavi_coke_studio.wav')
"""

import os
import sys

try:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    BASIC_PITCH_AVAILABLE = False

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def wav_to_midi(wav_path, output_midi_path=None,
                onset_threshold=0.5, frame_threshold=0.3,
                min_note_length_ms=50):
    """
    Convert a WAV audio file to MIDI using Spotify's Basic Pitch.

    Args:
        wav_path:           Path to input WAV file
        output_midi_path:   Where to save output MIDI (auto-named if None)
        onset_threshold:    Confidence threshold for note onsets (0-1)
        frame_threshold:    Confidence threshold for frame-level detection (0-1)
        min_note_length_ms: Minimum note duration in milliseconds

    Returns:
        str: Path to the generated MIDI file

    Raises:
        ImportError: If basic-pitch is not installed
        FileNotFoundError: If wav_path doesn't exist
    """
    if not BASIC_PITCH_AVAILABLE:
        print('⚠ basic-pitch not installed. Using librosa fallback.')
        return wav_to_midi_fallback(wav_path, output_midi_path)

    if not os.path.exists(wav_path):
        raise FileNotFoundError(f'Audio file not found: {wav_path}')

    if output_midi_path is None:
        output_midi_path = wav_path.rsplit('.', 1)[0] + '_converted.mid'

    print(f'🎵 Converting WAV to MIDI: {os.path.basename(wav_path)}')
    print(f'   Onset threshold: {onset_threshold}')
    print(f'   Frame threshold: {frame_threshold}')

    # Run Basic Pitch inference
    model_output, midi_data, note_events = predict(
        wav_path,
        ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_length_ms,
    )

    # Save MIDI
    midi_data.write(output_midi_path)

    # Report stats
    n_notes = sum(len(inst.notes) for inst in midi_data.instruments)
    duration = midi_data.get_end_time()
    print(f'   ✅ MIDI saved: {output_midi_path}')
    print(f'   Notes detected: {n_notes}')
    print(f'   Duration: {duration:.1f}s')

    if n_notes == 0:
        print('   ⚠ WARNING: No notes detected! Audio may be too percussive.')
        print('   Try lowering onset_threshold/frame_threshold or use melodic stems only.')

    return output_midi_path


def wav_to_midi_fallback(wav_path, output_midi_path=None):
    """
    Fallback MIDI conversion using librosa pitch detection.
    Less accurate than Basic Pitch but works without the dependency.

    Uses pitch tracking to detect the dominant pitch at each frame,
    then quantizes to MIDI notes.
    """
    if not LIBROSA_AVAILABLE or not PRETTY_MIDI_AVAILABLE:
        raise ImportError(
            'Fallback requires librosa and pretty_midi. '
            'Install: pip install librosa pretty_midi'
        )

    if output_midi_path is None:
        output_midi_path = wav_path.rsplit('.', 1)[0] + '_converted.mid'

    print(f'🔧 Fallback MIDI conversion: {os.path.basename(wav_path)}')

    # Load audio
    y, sr = librosa.load(wav_path, sr=22050, mono=True)

    # Track pitch
    pitches, magnitudes = librosa.piptrack(
        y=y, sr=sr, n_fft=2048, hop_length=512
    )

    # Extract dominant pitch at each frame
    hop_length = 512
    frame_times = librosa.frames_to_time(
        np.arange(pitches.shape[1]), sr=sr, hop_length=hop_length
    )

    # Create MIDI
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name='Melody')

    current_pitch = None
    note_start = None
    min_duration = 0.05  # 50ms minimum note

    for frame_idx in range(pitches.shape[1]):
        # Find the loudest pitch in this frame
        mag_col = magnitudes[:, frame_idx]
        if np.max(mag_col) < 0.01:  # Silence threshold
            if current_pitch is not None:
                # End current note
                note_end = frame_times[frame_idx]
                if note_end - note_start >= min_duration:
                    midi_note = int(round(
                        12 * np.log2(current_pitch / 440.0) + 69
                    ))
                    if 0 <= midi_note <= 127:
                        note = pretty_midi.Note(
                            velocity=80,
                            pitch=midi_note,
                            start=note_start,
                            end=note_end,
                        )
                        instrument.notes.append(note)
                current_pitch = None
                note_start = None
            continue

        peak_idx = mag_col.argmax()
        freq = pitches[peak_idx, frame_idx]

        if freq < 50 or freq > 4000:  # Skip extreme frequencies
            continue

        midi_note = int(round(12 * np.log2(freq / 440.0) + 69))

        if current_pitch is None:
            # Start new note
            current_pitch = freq
            note_start = frame_times[frame_idx]
        elif abs(midi_note - int(round(
                12 * np.log2(current_pitch / 440.0) + 69))) > 0:
            # Pitch changed — end old note, start new
            note_end = frame_times[frame_idx]
            if note_end - note_start >= min_duration:
                old_midi = int(round(
                    12 * np.log2(current_pitch / 440.0) + 69
                ))
                if 0 <= old_midi <= 127:
                    note = pretty_midi.Note(
                        velocity=80,
                        pitch=old_midi,
                        start=note_start,
                        end=note_end,
                    )
                    instrument.notes.append(note)
            current_pitch = freq
            note_start = frame_times[frame_idx]

    # End any remaining note
    if current_pitch is not None and note_start is not None:
        note_end = frame_times[-1]
        if note_end - note_start >= min_duration:
            midi_note = int(round(
                12 * np.log2(current_pitch / 440.0) + 69
            ))
            if 0 <= midi_note <= 127:
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=midi_note,
                    start=note_start,
                    end=note_end,
                )
                instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(output_midi_path)

    n_notes = len(instrument.notes)
    print(f'   ✅ Fallback MIDI saved: {output_midi_path}')
    print(f'   Notes detected: {n_notes}')

    if n_notes == 0:
        print('   ⚠ WARNING: No notes detected in fallback mode.')

    return output_midi_path


def get_midi_stats(midi_path):
    """
    Get statistics about a MIDI file for debugging.

    Returns:
        dict with note count, duration, pitch range, etc.
    """
    if not PRETTY_MIDI_AVAILABLE:
        raise ImportError('pretty_midi required. Install: pip install pretty_midi')

    midi = pretty_midi.PrettyMIDI(midi_path)
    all_notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            all_notes.extend(inst.notes)

    if not all_notes:
        return {'note_count': 0, 'duration': 0, 'pitch_range': (0, 0)}

    pitches = [n.pitch for n in all_notes]
    return {
        'note_count': len(all_notes),
        'duration': midi.get_end_time(),
        'pitch_range': (min(pitches), max(pitches)),
        'pitch_classes': sorted(set(p % 12 for p in pitches)),
        'instruments': [
            inst.name for inst in midi.instruments if not inst.is_drum
        ],
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='VIRASAT.AI — WAV to MIDI Converter'
    )
    parser.add_argument('input', help='Input WAV file path')
    parser.add_argument('--output', default=None, help='Output MIDI file path')
    parser.add_argument('--onset', type=float, default=0.5,
                        help='Onset threshold (0-1)')
    parser.add_argument('--frame', type=float, default=0.3,
                        help='Frame threshold (0-1)')
    args = parser.parse_args()

    midi_path = wav_to_midi(args.input, args.output,
                            onset_threshold=args.onset,
                            frame_threshold=args.frame)
    stats = get_midi_stats(midi_path)
    print(f'\n📊 MIDI Stats:')
    for k, v in stats.items():
        print(f'   {k}: {v}')
