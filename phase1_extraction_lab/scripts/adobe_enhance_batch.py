#!/usr/bin/env python3
"""
adobe_enhance_batch.py — Batch Automation for Adobe Podcast Enhance
===================================================================
Automates the upload, processing, and download cycle for the Adobe Enhance API.
Handles daily upload limits using a retry queue.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


def load_queue(queue_file):
    if os.path.exists(queue_file):
        with open(queue_file) as f:
            return json.load(f)
    return []


def save_queue(q, queue_file):
    with open(queue_file, 'w') as f:
        json.dump(q, f, indent=2)


def process_day(files_to_process, output_dir, token, daily_limit, queue_file):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process up to DAILY_LIMIT files
    processed = 0
    remaining_files = []
    
    for wav_path_str in files_to_process:
        wav_path = Path(wav_path_str)
        if processed >= daily_limit:
            remaining_files.append(wav_path_str)
            continue
            
        print(f"📡 Uploading: {wav_path.name}")
        
        try:
            with open(wav_path, 'rb') as f:
                resp = requests.post(
                    'https://podcast.adobe.com/api/v1/audio',
                    headers={'Authorization': f'Bearer {token}'},
                    files={'file': (wav_path.name, f, 'audio/wav')}
                )
                
            if resp.status_code != 200:
                print(f"❌ Upload failed (HTTP {resp.status_code}): {resp.text}")
                remaining_files.append(wav_path_str)
                continue
                
            job_id = resp.json().get('id')
            if not job_id:
                print(f"❌ No job ID returned by Adobe.")
                remaining_files.append(wav_path_str)
                continue
                
            print(f"   ⏳ Job ID: {job_id} — waiting for processing...")
            
            # Poll for completion
            success = False
            for _ in range(60):  # Max 5 minutes wait
                time.sleep(5)
                status_r = requests.get(
                    f'https://podcast.adobe.com/api/v1/audio/{job_id}',
                    headers={'Authorization': f'Bearer {token}'}
                )
                
                if status_r.json().get('status') == 'done':
                    download_url = status_r.json()['output']['url']
                    audio_data = requests.get(download_url).content
                    out_path = output_dir / f'enhanced_{wav_path.name}'
                    with open(out_path, 'wb') as out:
                        out.write(audio_data)
                    print(f"   ✅ Saved enhanced audio: {out_path}")
                    success = True
                    processed += 1
                    break
                    
            if not success:
                print(f"   ⚠️ Timeout waiting for job completion. Added back to queue.")
                remaining_files.append(wav_path_str)
                
        except Exception as e:
            print(f"❌ Error processing {wav_path.name}: {e}")
            remaining_files.append(wav_path_str)
            
    # Save the remaining files to the queue
    if remaining_files:
        print(f"\nℹ️ Hit daily limit or encountered errors. {len(remaining_files)} files saved to queue for tomorrow.")
    else:
        print(f"\n✅ All files processed successfully. Queue is empty.")
        
    save_queue(remaining_files, queue_file)


def main():
    parser = argparse.ArgumentParser(description="VIRASAT AI — Adobe Enhance Batch Automation")
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Input directory containing stems (will recursively search for vocals.wav) or a single WAV file")
    parser.add_argument("--output", "-o", type=str, required=True, 
                        help="Output directory for enhanced WAV files")
    parser.add_argument("--token", "-t", type=str, 
                        help="Adobe Podcast API Session Token (Bearer token)")
    parser.add_argument("--limit", "-l", type=int, default=5, 
                        help="Daily upload limit (default: 5)")
    parser.add_argument("--queue", "-q", type=str, default="enhance_queue.json", 
                        help="Queue file to save unprocessed files (default: enhance_queue.json)")
    
    args = parser.parse_args()
    
    token = args.token or os.environ.get("ADOBE_SESSION_TOKEN")
    if not token:
        print("❌ Error: Adobe session token is required.")
        print("   Pass it via --token or set the ADOBE_SESSION_TOKEN environment variable.")
        print("   You can extract this from the enhance.adobe.com Network tab in DevTools.")
        sys.exit(1)
        
    # Auto-strip 'Bearer ' if the user copy-pasted it with the prefix
    if token.startswith("Bearer "):
        token = token[7:].strip()
        
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    # Initialize the queue
    queue = load_queue(args.queue)
    
    # If a new path was provided, add its files to the queue
    if input_path.is_file():
        if str(input_path) not in queue:
            queue.append(str(input_path))
    elif input_path.is_dir():
        # Find all vocals.wav
        found_vocals = list(input_path.rglob("vocals.wav"))
        for v in found_vocals:
            if str(v) not in queue:
                queue.append(str(v))
    else:
        print(f"❌ Input path {input_path} does not exist.")
        sys.exit(1)
        
    if not queue:
        print("ℹ️ No files to process.")
        sys.exit(0)
        
    print(f"📦 Starting batch enhancement queue ({len(queue)} files)...")
    process_day(queue, output_dir, token, args.limit, args.queue)


if __name__ == "__main__":
    main()
