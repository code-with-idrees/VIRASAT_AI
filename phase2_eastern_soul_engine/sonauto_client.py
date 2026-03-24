#!/usr/bin/env python3
"""
sonauto_client.py — Sonauto / Udio AI Music Generation Client
================================================================
Submits generation requests to Sonauto's API and downloads the resulting
audio tracks. Includes a local fallback synthesizer for offline testing.

API Setup:
    1. Sign up at https://sonauto.ai (free tier: ~20 credits/month)
    2. Get your API key from the dashboard
    3. Set environment variable: export SONAUTO_KEY='your_key_here'

Usage:
    from sonauto_client import generate_track_sonauto
    from prompt_generator import generate_eastern_prompt

    config = generate_eastern_prompt('Bhairavi', 'Keherwa')
    track = generate_track_sonauto(config, 'generated_tracks/')
"""

import os
import sys
import time
import json
import hashlib

import numpy as np

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


# ─── Configuration ────────────────────────────────────────────────────────────

SONAUTO_API_KEY = os.environ.get('SONAUTO_KEY', '')
SONAUTO_BASE_URL = 'https://sonauto.ai/api/v1'

# Polling configuration
MAX_POLL_ATTEMPTS = 60   # 5 minutes at 5-second intervals
POLL_INTERVAL_SEC = 5


# ─── Sonauto API Client ──────────────────────────────────────────────────────

def generate_track_sonauto(prompt_config, output_dir='generated_tracks/'):
    """
    Call Sonauto API to generate a music track from a prompt config.

    Args:
        prompt_config: Dict from generate_eastern_prompt() containing
                       'prompt', 'raag', 'taal', 'bpm', 'style'
        output_dir:    Directory to save the downloaded WAV

    Returns:
        str: Path to the downloaded WAV file, or None on failure
    """
    if not REQUESTS_AVAILABLE:
        print('⚠ requests library not available. Using local fallback.')
        return generate_track_local_fallback(prompt_config, output_dir)

    api_key = SONAUTO_API_KEY
    if not api_key:
        print('⚠ SONAUTO_KEY not set. Using local fallback synthesizer.')
        return generate_track_local_fallback(prompt_config, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    # Step 1: Submit generation request
    print(f'🎵 Submitting generation request to Sonauto...')
    print(f'   Raag: {prompt_config["raag"]} | '
          f'Taal: {prompt_config["taal"]} | '
          f'Style: {prompt_config["style"]}')

    try:
        response = requests.post(
            f'{SONAUTO_BASE_URL}/generate',
            headers=headers,
            json={
                'prompt': prompt_config['prompt'],
                'duration': 120,
                'make_instrumental': True,
                'model': 'chirp-v3',
            },
            timeout=30,
        )
    except requests.RequestException as e:
        print(f'❌ API request failed: {e}')
        print('   Falling back to local synthesizer...')
        return generate_track_local_fallback(prompt_config, output_dir)

    if response.status_code != 200:
        print(f'❌ API error {response.status_code}: {response.text}')
        print('   Falling back to local synthesizer...')
        return generate_track_local_fallback(prompt_config, output_dir)

    job_id = response.json().get('id', 'unknown')
    print(f'   Job ID: {job_id}')

    # Step 2: Poll for completion
    for attempt in range(MAX_POLL_ATTEMPTS):
        time.sleep(POLL_INTERVAL_SEC)

        try:
            status_resp = requests.get(
                f'{SONAUTO_BASE_URL}/generate/{job_id}',
                headers=headers,
                timeout=15,
            )
            status = status_resp.json()
        except requests.RequestException as e:
            print(f'   [{attempt * POLL_INTERVAL_SEC}s] Poll error: {e}')
            continue

        job_status = status.get('status', 'pending')

        if job_status == 'complete':
            audio_url = status.get('audio_url')
            print(f'   ✅ Generation complete!')
            break
        elif job_status == 'failed':
            print(f'   ❌ Generation failed: {status.get("error", "unknown")}')
            return None
        else:
            print(f'   [{attempt * POLL_INTERVAL_SEC}s] Status: {job_status}')
    else:
        print('   ⏱ Timeout waiting for generation.')
        return None

    # Step 3: Download the audio
    filename = (
        f'{output_dir}/{prompt_config["raag"]}_'
        f'{prompt_config["style"]}_{job_id[:8]}.wav'
    )

    try:
        audio_data = requests.get(audio_url, timeout=60).content
        with open(filename, 'wb') as f:
            f.write(audio_data)
        print(f'   💾 Downloaded: {filename}')
        return filename
    except requests.RequestException as e:
        print(f'   ❌ Download failed: {e}')
        return None


# ─── Local Fallback Synthesizer ───────────────────────────────────────────────
# For testing when no API key is available. Generates simple sine-wave tones
# using the Raag's note set so that downstream modules can be tested.

def generate_track_local_fallback(prompt_config, output_dir='generated_tracks/'):
    """
    Generate a simple synthetic test track using the Raag's note set.
    This is NOT a real music track — it's a test signal for validating
    the Raag-Lock and beat sync pipelines.

    Creates a sequence of sine-wave tones at Raag-allowed pitches.
    """
    if not SOUNDFILE_AVAILABLE:
        print('❌ soundfile not installed. Cannot generate fallback track.')
        return None

    os.makedirs(output_dir, exist_ok=True)

    sr = 22050  # Sample rate
    duration = 30  # seconds (shorter for testing)
    notes = prompt_config['notes']
    bpm = prompt_config['bpm']

    beat_duration = 60.0 / bpm  # seconds per beat
    base_freq = 261.63  # Middle C (C4)

    # Build a simple melody from the Raag notes
    audio = np.zeros(int(duration * sr))
    t_per_note = beat_duration  # One note per beat

    current_sample = 0
    note_idx = 0

    while current_sample < len(audio):
        # Cycle through Raag notes
        pitch_class = notes[note_idx % len(notes)]
        freq = base_freq * (2 ** (pitch_class / 12.0))

        # Determine number of samples for this note
        n_samples = int(t_per_note * sr)
        if current_sample + n_samples > len(audio):
            n_samples = len(audio) - current_sample

        # Generate sine wave with envelope (attack-release)
        t = np.linspace(0, t_per_note, n_samples, endpoint=False)
        envelope = np.ones_like(t)
        attack = int(0.05 * sr)  # 50ms attack
        release = int(0.1 * sr)   # 100ms release
        if len(envelope) > attack:
            envelope[:attack] = np.linspace(0, 1, attack)
        if len(envelope) > release:
            envelope[-release:] = np.linspace(1, 0, release)

        tone = 0.3 * np.sin(2 * np.pi * freq * t) * envelope
        audio[current_sample:current_sample + n_samples] += tone

        current_sample += n_samples
        note_idx += 1

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.8

    # Generate unique filename
    hash_id = hashlib.md5(
        prompt_config['prompt'].encode()
    ).hexdigest()[:8]
    filename = (
        f'{output_dir}/{prompt_config["raag"]}_'
        f'{prompt_config["style"]}_{hash_id}_fallback.wav'
    )
    sf.write(filename, audio, sr)

    print(f'🔧 Local fallback track generated: {filename}')
    print(f'   (Synthetic {duration}s test signal — not real music)')
    return filename


# ─── Track Rating Helper ─────────────────────────────────────────────────────

def rate_track(track_path, rating, notes=''):
    """
    Save a rating for a generated track. Used during prompt iteration
    to track which prompts produce the best results.

    Args:
        track_path: Path to the WAV file
        rating:     1-5 star rating
        notes:      Free-text notes about what worked/didn't
    """
    ratings_file = os.path.join(os.path.dirname(track_path), 'ratings.json')

    ratings = {}
    if os.path.exists(ratings_file):
        with open(ratings_file) as f:
            ratings = json.load(f)

    ratings[os.path.basename(track_path)] = {
        'rating': rating,
        'notes': notes,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(ratings_file, 'w') as f:
        json.dump(ratings, f, indent=2)

    stars = '⭐' * rating + '☆' * (5 - rating)
    print(f'{stars} Rated: {os.path.basename(track_path)}')


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(__file__))
    from prompt_generator import generate_eastern_prompt

    import argparse
    parser = argparse.ArgumentParser(
        description='VIRASAT.AI — Sonauto Track Generator'
    )
    parser.add_argument('--raag', default='Bhairavi')
    parser.add_argument('--taal', default='Keherwa')
    parser.add_argument('--style', default='coke_studio')
    parser.add_argument('--output', default='generated_tracks/')
    parser.add_argument('--fallback', action='store_true',
                        help='Force local fallback (skip API)')
    args = parser.parse_args()

    config = generate_eastern_prompt(args.raag, args.taal, style=args.style)

    if args.fallback:
        track = generate_track_local_fallback(config, args.output)
    else:
        track = generate_track_sonauto(config, args.output)

    if track:
        print(f'\n✅ Track ready: {track}')
