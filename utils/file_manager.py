#!/usr/bin/env python3
"""
file_manager.py — Path Management & File Organization
======================================================
Manages file paths, output organization, and directory creation
for the VIRASAT AI pipeline.
"""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime


# ─── Project Paths ────────────────────────────────────────────

def get_project_root():
    """Get the VIRASAT_AI project root directory."""
    # Walk up from this file to find project root
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "README.md").exists() and (current / "phase1_extraction_lab").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent


def get_phase1_dir():
    return get_project_root() / "phase1_extraction_lab"


def get_data_dir():
    return get_phase1_dir() / "data"


def get_raw_dir():
    return get_data_dir() / "raw"


def get_stems_dir():
    return get_data_dir() / "stems"


def get_enhanced_dir():
    return get_data_dir() / "enhanced"


def get_reports_dir():
    return get_data_dir() / "reports"


def get_config_dir():
    return get_phase1_dir() / "config"


# ─── Directory Management ────────────────────────────────────

def ensure_dirs():
    """Create all required directories if they don't exist."""
    dirs = [
        get_raw_dir(),
        get_stems_dir(),
        get_enhanced_dir(),
        get_reports_dir(),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def create_session_dir(prefix="session"):
    """Create a timestamped session directory for organized output."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = get_reports_dir() / f"{prefix}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


# ─── Config Loading ──────────────────────────────────────────

def load_config(name):
    """Load a JSON config file from the config directory."""
    config_path = get_config_dir() / name
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def save_report(data, filename, session_dir=None):
    """Save a JSON report to the reports directory."""
    if session_dir is None:
        session_dir = get_reports_dir()
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    output_path = session_dir / filename
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return output_path


def get_file_size_mb(path):
    """Get file size in MB."""
    return round(os.path.getsize(path) / (1024 * 1024), 2)


def list_audio_files(directory):
    """List all audio files in directory with sizes."""
    directory = Path(directory)
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = []
    for f in sorted(directory.rglob("*")):
        if f.suffix.lower() in extensions:
            files.append({
                "path": str(f),
                "name": f.name,
                "size_mb": get_file_size_mb(f),
                "extension": f.suffix,
            })
    return files
