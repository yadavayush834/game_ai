#!/usr/bin/env python3
"""
record.py — Ultra-fast screen + input recorder for Game-Bot.

Usage:
    python record.py                  # Start recording a new session
    python record.py --fps 30         # Cap capture rate to 30 FPS (default: 20)
    python record.py --region 0,0,1920,1080  # Capture a specific screen region
    python record.py --test           # Benchmark capture speed, no recording

Press Ctrl+C in the terminal to stop recording.
"""

import os
import sys
import csv
import time
import argparse
import threading
import numpy as np
from datetime import datetime
from PIL import Image

import mss
from pynput.keyboard import Listener as KeyListener
from pynput.mouse import Listener as MouseListener

from core.utils import (
    ACTIONS, ACTION_TO_INDEX,
    pynput_key_to_action, pynput_mouse_to_action,
)

# ── Global state (thread-safe via GIL for simple booleans) ────────────────────
# 8-element array: w, a, s, d, space, shift, click_left, click_right
active_keys = [0] * 8
lock = threading.Lock()


# ── Listeners ─────────────────────────────────────────────────────────────────

def on_key_press(key):
    action = pynput_key_to_action(key)
    if action is not None:
        idx = ACTION_TO_INDEX[action]
        with lock:
            active_keys[idx] = 1


def on_key_release(key):
    action = pynput_key_to_action(key)
    if action is not None:
        idx = ACTION_TO_INDEX[action]
        with lock:
            active_keys[idx] = 0


def on_mouse_click(x, y, button, pressed):
    action = pynput_mouse_to_action(button)
    if action is not None:
        idx = ACTION_TO_INDEX[action]
        with lock:
            active_keys[idx] = 1 if pressed else 0


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(region=None, duration=5):
    """Capture as fast as possible for `duration` seconds and report FPS."""
    print(f"Benchmarking screen capture for {duration}s …")
    sct = mss.mss()
    monitor = sct.monitors[1] if region is None else {
        "left": region[0], "top": region[1],
        "width": region[2] - region[0], "height": region[3] - region[1],
    }

    count = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        sct.grab(monitor)
        count += 1
    elapsed = time.perf_counter() - t0
    fps = count / elapsed
    print(f"Captured {count} frames in {elapsed:.2f}s → {fps:.1f} FPS")
    return fps


# ── Recording ─────────────────────────────────────────────────────────────────

def record(fps=20, region=None):
    """
    Records screenshots + current key state at the target FPS.
    Saves frames as PNGs and writes a manifest.csv linking each frame to its
    multi-label action vector.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join("Data", "Recordings", f"session_{timestamp}")
    frames_dir = os.path.join(session_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    csv_path = os.path.join(session_dir, "manifest.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    header = ["frame"] + [ACTIONS[i] for i in range(8)]
    writer.writerow(header)

    # Start input listeners in background threads
    kl = KeyListener(on_press=on_key_press, on_release=on_key_release)
    ml = MouseListener(on_click=on_mouse_click)
    kl.daemon = True
    ml.daemon = True
    kl.start()
    ml.start()

    sct = mss.mss()
    monitor = sct.monitors[1] if region is None else {
        "left": region[0], "top": region[1],
        "width": region[2] - region[0], "height": region[3] - region[1],
    }

    interval = 1.0 / fps
    frame_count = 0

    print("=" * 60)
    print(f"  GAME-BOT RECORDER  —  session: {timestamp}")
    print(f"  Target FPS: {fps}  |  Region: {'Full Screen' if region is None else region}")
    print(f"  Output: {session_dir}")
    print("=" * 60)
    print("Recording … Press Ctrl+C to stop.\n")

    try:
        while True:
            t0 = time.perf_counter()

            # Capture
            raw = sct.grab(monitor)
            img = Image.frombytes("RGB", (raw.width, raw.height), raw.rgb)
            img = img.resize((150, 150), Image.LANCZOS)

            # Snapshot current keys
            with lock:
                snapshot = list(active_keys)

            # Save frame
            fname = f"frame_{frame_count:08d}.png"
            img.save(os.path.join(frames_dir, fname))

            # Write CSV row (relative path so dataset.py can resolve it)
            writer.writerow([os.path.join("frames", fname)] + snapshot)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  Captured {frame_count} frames …")

            # Throttle
            elapsed = time.perf_counter() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        csv_file.close()
        kl.stop()
        ml.stop()
        print(f"\nDone! Saved {frame_count} frames to {session_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Game-Bot Recorder")
    parser.add_argument("--fps", type=int, default=20,
                        help="Target capture frames per second (default: 20)")
    parser.add_argument("--region", type=str, default=None,
                        help="Screen region as left,top,right,bottom (e.g. 0,0,1920,1080)")
    parser.add_argument("--test", action="store_true",
                        help="Run a capture benchmark instead of recording")
    args = parser.parse_args()

    region = None
    if args.region:
        region = tuple(int(x) for x in args.region.split(","))
        assert len(region) == 4, "Region must be 4 integers: left,top,right,bottom"

    if args.test:
        benchmark(region)
    else:
        record(fps=args.fps, region=region)


if __name__ == "__main__":
    main()
