#!/usr/bin/env python3
"""
visualization.py — Audio Visualization Utilities
=================================================
Spectrogram plotting, comparison charts, and PCP display.
"""

from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def plot_spectrogram(audio_path, output_path=None, title=None, sr=44100):
    """Generate spectrogram visualization."""
    if not MATPLOTLIB_AVAILABLE or not LIBROSA_AVAILABLE:
        print("⚠️  matplotlib and librosa required")
        return None

    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 5))
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    ax.set_title(title or f"Spectrogram: {Path(audio_path).stem}", fontsize=13)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        plt.show()
        plt.close(fig)
    return None


def plot_comparison(audio_path_1, audio_path_2, label_1="Model A", label_2="Model B",
                    output_path=None, sr=44100):
    """Side-by-side spectrogram comparison of two stems."""
    if not MATPLOTLIB_AVAILABLE or not LIBROSA_AVAILABLE:
        return None

    y1, _ = librosa.load(str(audio_path_1), sr=sr, mono=True)
    y2, _ = librosa.load(str(audio_path_2), sr=sr, mono=True)

    S1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max)
    S2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    librosa.display.specshow(S1, sr=sr, x_axis='time', y_axis='hz', ax=ax1, cmap='magma')
    ax1.set_title(f"{label_1}: {Path(audio_path_1).stem}", fontsize=12)

    librosa.display.specshow(S2, sr=sr, x_axis='time', y_axis='hz', ax=ax2, cmap='magma')
    ax2.set_title(f"{label_2}: {Path(audio_path_2).stem}", fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
    return output_path


def plot_pcp(pcp, title="Pitch Class Profile", output_path=None):
    """Plot Pitch Class Profile as bar chart."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    note_names = ["Sa", "Re♭", "Re", "Ga♭", "Ga", "Ma", "Ma#",
                  "Pa", "Dha♭", "Dha", "Ni♭", "Ni"]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#ff6b6b' if v == max(pcp) else '#4ecdc4' for v in pcp]
    ax.bar(range(12), pcp, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(12))
    ax.set_xticklabels(note_names, fontsize=11)
    ax.set_ylabel("Relative Energy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
    return output_path


def plot_metrics_comparison(metrics_dict, output_path=None):
    """Bar chart comparing quality metrics across models/songs."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    labels = list(metrics_dict.keys())
    metric_names = ['SDR', 'SIR', 'SAR']

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metric_names):
        key = metric.lower() + '_db'
        values = [m.get(key, 0) for m in metrics_dict.values()]
        ax.bar(x + i * width, values, width, label=metric, alpha=0.85)

    ax.set_xlabel('Song / Model', fontsize=12)
    ax.set_ylabel('dB', fontsize=12)
    ax.set_title('VIRASAT AI — Quality Metrics Comparison', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
    return output_path
