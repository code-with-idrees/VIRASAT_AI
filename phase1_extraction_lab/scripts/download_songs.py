#!/usr/bin/env python3
"""
download_songs.py — Song Downloader for VIRASAT AI
===================================================
Uses yt-dlp to download audio from YouTube for testing.
Organizes downloads by artist/era with metadata extraction.

Usage:
  python download_songs.py --url "https://youtube.com/watch?v=..." --artist "Noor Jehan"
  python download_songs.py --search "Ranjish Hi Sahi Mehdi Hassan" --artist "Mehdi Hassan"
  python download_songs.py --config config/test_songs.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ─── Configuration ────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
CONFIG_DIR = Path(__file__).parent.parent / "config"

AUDIO_FORMAT = "wav"  # WAV for maximum quality (Demucs input)
SAMPLE_RATE = 44100   # 44.1 kHz standard


# ─── Core Functions ──────────────────────────────────────────

def check_ytdlp():
    """Check if yt-dlp is installed."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_audio(url, output_dir=None, filename=None, artist=None):
    """
    Download audio from YouTube URL using yt-dlp.

    Parameters:
        url:        YouTube URL
        output_dir: Output directory (default: data/raw/)
        filename:   Custom filename (without extension)
        artist:     Artist name for directory organization

    Returns:
        dict: {success, output_path, title, duration}
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir = Path(output_dir)

    # Organize by artist if provided
    if artist:
        output_dir = output_dir / artist.replace(" ", "_")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output template
    if filename:
        output_template = str(output_dir / f"{filename}.%(ext)s")
    else:
        output_template = str(output_dir / "%(title)s.%(ext)s")

    # yt-dlp command (using python -m yt_dlp for better compatibility)
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--extract-audio",
        "--audio-format", AUDIO_FORMAT,
        "--audio-quality", "0",  # Best quality
        "--output", output_template,
        "--no-playlist",  # Single video only
        "--no-warnings",  # Suppress noisy warnings
        "--write-info-json",  # Save metadata
        url
    ]

    print(f"\n⬇️  Downloading: {url}")
    if artist:
        print(f"   Artist: {artist}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode == 0:
            # Find the downloaded file
            wav_files = sorted(output_dir.glob("*.wav"), key=os.path.getmtime, reverse=True)
            if wav_files:
                output_path = wav_files[0]
                print(f"   ✅ Downloaded: {output_path.name}")
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "url": url,
                    "artist": artist,
                }
            else:
                print(f"   ⚠️  Downloaded but file not found in {output_dir}")
                return {"success": False, "error": "File not found after download"}
        else:
            print(f"   ❌ Error: {result.stderr[:200]}")
            return {"success": False, "error": result.stderr[:500]}

    except subprocess.TimeoutExpired:
        print("   ❌ Timeout (>5 min)")
        return {"success": False, "error": "Download timeout"}
    except FileNotFoundError:
        print("   ❌ yt-dlp not installed. Run: pip install yt-dlp")
        return {"success": False, "error": "yt-dlp not installed"}


def search_and_download(query, output_dir=None, artist=None, max_results=1):
    """
    Search YouTube and download the best match.

    Parameters:
        query:       Search query string
        output_dir:  Output directory
        artist:      Artist name
        max_results: Number of results to try
    """
    # Use yt-dlp search
    url = f"ytsearch{max_results}:{query}"
    return download_audio(url, output_dir, artist=artist)


def download_from_config(config_path=None):
    """
    Download all test songs from the test_songs.json config.
    Only downloads songs that have YouTube URLs set.
    """
    if config_path is None:
        config_path = CONFIG_DIR / "test_songs.json"

    config_path = Path(config_path)
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return []

    with open(config_path) as f:
        config = json.load(f)

    results = []
    test_songs = config.get("test_songs", [])

    print(f"\n{'='*60}")
    print(f"🎼 VIRASAT AI — Batch Song Download")
    print(f"{'='*60}")
    print(f"   Songs in config: {len(test_songs)}")

    for song in test_songs:
        url = song.get("youtube_url")
        artist = song.get("artist", "Unknown")
        title = song.get("song_title", "Unknown")

        if url:
            filename = f"{artist.replace(' ', '_')}_{title.replace(' ', '_')}"
            result = download_audio(url, artist=artist, filename=filename)
            result["song_config"] = song
            results.append(result)
        else:
            print(f"\n⏭️  Skipping: {artist} - {title} (no URL set)")
            results.append({
                "success": False,
                "artist": artist,
                "title": title,
                "error": "No YouTube URL configured",
            })

    # Summary
    success_count = sum(1 for r in results if r.get("success"))
    print(f"\n{'='*60}")
    print(f"📊 Download Summary: {success_count}/{len(results)} successful")
    print(f"{'='*60}")

    return results


# ─── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VIRASAT AI — Song Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from URL
  python download_songs.py --url "https://youtube.com/watch?v=..." --artist "Mehdi Hassan"

  # Search and download
  python download_songs.py --search "Ranjish Hi Sahi Mehdi Hassan" --artist "Mehdi Hassan"

  # Batch download from config
  python download_songs.py --config config/test_songs.json
        """,
    )

    parser.add_argument("--url", help="YouTube URL to download")
    parser.add_argument("--search", help="Search query to find and download")
    parser.add_argument("--config", help="Path to test_songs.json for batch download")
    parser.add_argument("--artist", help="Artist name for directory organization")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--filename", help="Custom filename (without extension)")
    parser.add_argument("--check", action="store_true", help="Check if yt-dlp is installed")

    args = parser.parse_args()

    if args.check:
        if check_ytdlp():
            print("✅ yt-dlp is installed")
        else:
            print("❌ yt-dlp not found. Install: pip install yt-dlp")
        return

    if args.config:
        download_from_config(args.config)
    elif args.url:
        download_audio(args.url, args.output, args.filename, args.artist)
    elif args.search:
        search_and_download(args.search, args.output, args.artist)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
