#!/usr/bin/env python3
"""
rvc_training.py — RVC Voice Model Training
=============================================
Prepares training data and trains an RVC (Retrieval-based Voice Conversion)
model from a clean vocal stem. The trained model captures a singer's vocal
timbre, texture, resonance, and vibrato style.

This is the setup side of the 'Ghost Collaboration' feature. Once trained,
the model can be used via rvc_inference.py to make any audio sound like
the target voice.

Prerequisites:
    - RVC-WebUI cloned: git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git
    - GPU with CUDA (Google Colab T4 recommended)
    - Clean vocal stem from Phase 1 (Demucs output, Adobe-enhanced)

Usage:
    from rvc_training import prepare_training_data, train_rvc_model
    model_dir = prepare_training_data('vocal_clean.wav', 'ghulam_ali_v1')
    train_rvc_model('ghulam_ali_v1', model_dir, epochs=200)
"""

import os
import sys
import subprocess
import shutil

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
DEFAULT_DATASET_DIR = 'datasets'
DEFAULT_MODEL_DIR = 'models'

# Training parameters
DEFAULT_SAMPLE_RATE = 40000   # RVC preferred rate
CHUNK_DURATION_SEC = 10       # Split audio into 10-second chunks
MIN_CHUNK_RATIO = 0.5         # Skip chunks shorter than 50% of target


# ─── Training Data Preparation ───────────────────────────────────────────────

def prepare_training_data(vocal_wav_path, model_name,
                          dataset_dir=DEFAULT_DATASET_DIR,
                          target_sr=DEFAULT_SAMPLE_RATE):
    """
    Prepare audio for RVC training. Splits long audio into chunks
    for training stability.

    Args:
        vocal_wav_path: Path to clean vocal WAV (Phase 1 output)
        model_name:     Name for the voice model (e.g. 'ghulam_ali_v1')
        dataset_dir:    Base directory for training data
        target_sr:      Target sample rate (RVC uses 40kHz)

    Returns:
        str: Path to the prepared dataset directory
    """
    if not AUDIO_AVAILABLE:
        raise ImportError(
            'librosa and soundfile required. '
            'Install: pip install librosa soundfile'
        )

    if not os.path.exists(vocal_wav_path):
        raise FileNotFoundError(f'Vocal file not found: {vocal_wav_path}')

    model_dir = os.path.join(dataset_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    print(f'🎤 Preparing training data: {model_name}')
    print(f'   Source: {vocal_wav_path}')

    # Load and resample to target rate
    y, sr = librosa.load(vocal_wav_path, sr=target_sr, mono=True)
    total_duration = len(y) / target_sr

    print(f'   Duration: {total_duration:.1f}s | Sample rate: {target_sr}Hz')

    # Split into chunks
    chunk_size = target_sr * CHUNK_DURATION_SEC
    min_chunk_size = int(chunk_size * MIN_CHUNK_RATIO)

    chunks = []
    for i in range(0, len(y), chunk_size):
        chunk = y[i:i + chunk_size]
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)

    # Save chunks
    for idx, chunk in enumerate(chunks):
        chunk_path = os.path.join(model_dir, f'chunk_{idx:03d}.wav')
        sf.write(chunk_path, chunk, target_sr)

    print(f'   ✅ Prepared {len(chunks)} training chunks in {model_dir}')
    print(f'   Each chunk: {CHUNK_DURATION_SEC}s @ {target_sr}Hz')

    if total_duration < 10:
        print('   ⚠ WARNING: Very short audio (<10s). Model quality may be low.')
        print('   Recommended: 1-5 minutes of clean vocal.')
    elif total_duration < 30:
        print('   ℹ Short audio (10-30s). Consider using more source material.')

    return model_dir


# ─── RVC Model Training ──────────────────────────────────────────────────────

def train_rvc_model(model_name, dataset_dir=None,
                    epochs=100, batch_size=4,
                    save_every=10, rvc_dir=None):
    """
    Train an RVC voice model.

    This wraps the RVC-WebUI training script. Requires the RVC repo
    to be cloned and dependencies installed.

    Args:
        model_name:  Name for the model (must match prepare_training_data)
        dataset_dir: Path to training data directory
        epochs:      Number of training epochs (100=basic, 300+=high quality)
        batch_size:  Training batch size (4 for T4 GPU, 8 for A100)
        save_every:  Save checkpoint every N epochs
        rvc_dir:     Path to RVC-WebUI installation

    Returns:
        str: Path to the trained .pth model file

    Note:
        Training time estimates (T4 GPU):
            100 epochs  → ~10 min
            200 epochs  → ~20 min
            500 epochs  → ~50 min
    """
    if rvc_dir is None:
        rvc_dir = RVC_DIR

    if not os.path.exists(rvc_dir):
        print(f'❌ RVC directory not found: {rvc_dir}')
        print('   Clone it first:')
        print('   git clone https://github.com/RVC-Project/'
              'Retrieval-based-Voice-Conversion-WebUI.git')
        return _create_placeholder_model(model_name)

    train_script = os.path.join(rvc_dir, 'train.py')
    if not os.path.exists(train_script):
        print(f'❌ train.py not found in {rvc_dir}')
        return _create_placeholder_model(model_name)

    print(f'🏋️ Training RVC model: {model_name}')
    print(f'   Epochs: {epochs} | Batch size: {batch_size}')
    print(f'   Save every: {save_every} epochs')

    cmd = [
        sys.executable, train_script,
        '--exp_name', model_name,
        '--save_every_epoch', str(save_every),
        '--total_epoch', str(epochs),
        '--batch_size', str(batch_size),
        '--if_save_latest', '1',
        '--if_cache_gpu_data', '1',
        '--if_save_every_weights', '0',
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=rvc_dir,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )

        if result.returncode == 0:
            model_path = os.path.join(
                rvc_dir, 'weights', f'{model_name}.pth'
            )
            print(f'   ✅ Training complete: {model_path}')
            return model_path
        else:
            print(f'   ❌ Training failed: {result.stderr[:500]}')
            return _create_placeholder_model(model_name)

    except subprocess.TimeoutExpired:
        print('   ⏱ Training timeout (1 hour). Check progress manually.')
        return None
    except FileNotFoundError:
        print('   ❌ Python or train.py not found.')
        return _create_placeholder_model(model_name)


def _create_placeholder_model(model_name):
    """
    Create a placeholder model file for testing when RVC is not installed.
    The placeholder contains metadata but cannot perform actual voice conversion.
    """
    import json

    model_dir = DEFAULT_MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)

    placeholder_path = os.path.join(model_dir, f'{model_name}_placeholder.json')

    metadata = {
        'model_name': model_name,
        'type': 'placeholder',
        'note': 'This is a placeholder. Train with RVC on GPU for real model.',
        'instructions': [
            '1. Open Google Colab with GPU runtime',
            '2. Clone RVC-WebUI repository',
            '3. Install dependencies',
            '4. Run train_rvc_model() with your vocal stem',
        ],
    }

    with open(placeholder_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'   📋 Placeholder model saved: {placeholder_path}')
    print('   (Use Google Colab with GPU to train a real model)')
    return placeholder_path


# ─── Utilities ───────────────────────────────────────────────────────────────

def check_training_requirements():
    """
    Check if all requirements for RVC training are met.
    Returns dict with status for each requirement.
    """
    import importlib

    requirements = {
        'python_version': sys.version,
        'cuda_available': False,
        'gpu_name': 'None',
        'librosa': False,
        'soundfile': False,
        'torch': False,
        'rvc_installed': os.path.exists(RVC_DIR),
    }

    # Check CUDA
    try:
        import torch
        requirements['torch'] = True
        requirements['cuda_available'] = torch.cuda.is_available()
        if requirements['cuda_available']:
            requirements['gpu_name'] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    # Check audio libs
    try:
        import librosa
        requirements['librosa'] = True
    except ImportError:
        pass

    try:
        import soundfile
        requirements['soundfile'] = True
    except ImportError:
        pass

    return requirements


def estimate_training_time(audio_duration_sec, epochs, gpu='T4'):
    """
    Estimate training time based on audio length and epochs.

    Returns:
        str: Human-readable time estimate
    """
    # Rough estimates based on RVC benchmarks
    base_time_per_epoch = {
        'T4': 6,     # seconds per epoch
        'A100': 2,
        'V100': 4,
        'CPU': 60,
    }

    time_per_epoch = base_time_per_epoch.get(gpu, 10)
    # Audio length scales training roughly linearly
    scale = max(1, audio_duration_sec / 60.0)
    total_seconds = time_per_epoch * epochs * scale

    if total_seconds < 60:
        return f'{total_seconds:.0f} seconds'
    elif total_seconds < 3600:
        return f'{total_seconds / 60:.0f} minutes'
    else:
        return f'{total_seconds / 3600:.1f} hours'


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='VIRASAT.AI — RVC Voice Model Training'
    )
    parser.add_argument('--input', required=True,
                        help='Clean vocal WAV from Phase 1')
    parser.add_argument('--name', required=True,
                        help='Model name (e.g. ghulam_ali_v1)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--check', action='store_true',
                        help='Check requirements only')
    args = parser.parse_args()

    if args.check:
        reqs = check_training_requirements()
        print('🔍 Training Requirements Check:')
        for k, v in reqs.items():
            status = '✅' if v else '❌'
            if isinstance(v, str):
                status = 'ℹ'
            print(f'   {status} {k}: {v}')
    else:
        model_dir = prepare_training_data(args.input, args.name)
        model_path = train_rvc_model(
            args.name, model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print(f'\n🎤 Model ready: {model_path}')
