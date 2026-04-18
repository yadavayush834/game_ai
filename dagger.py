#!/usr/bin/env python3
"""
dagger.py — DAgger (Dataset Aggregation) for Game-Bot.

How DAgger works:
  1. The AI plays the game using the current trained model.
  2. You WATCH. When the AI makes a mistake, you press the CORRECT key(s).
  3. Your key presses are treated as "expert corrections" and recorded
     alongside the current screen state.
  4. At the end, these corrections are saved as a new session for retraining.
  5. You retrain:  python train_new.py --resume
  6. Repeat — the AI improves each round.

Usage:
    python dagger.py                          # Run DAgger with current best model
    python dagger.py --threshold 0.4          # Lower threshold so AI acts more
    python dagger.py --fps 15 --delay 5
    python dagger.py --region 0,0,1920,1080
"""

import os
import csv
import sys
import time
import argparse
import threading
from collections import deque
from datetime import datetime

import mss
import torch
from PIL import Image
from torchvision import transforms
from pynput.keyboard import Controller as Keyboard, Listener as KeyListener
from pynput.mouse import Controller as Mouse, Button, Listener as MouseListener

from core.model import GameBotModel, NUM_FRAMES
from core.utils import ACTIONS, ACTION_TO_INDEX, action_to_pynput, pynput_key_to_action, pynput_mouse_to_action

# ── Frame pre-processing ──────────────────────────────────────────────────────
_FRAME_TRANSFORM = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

# ── Shared state ──────────────────────────────────────────────────────────────
lock              = threading.Lock()
human_keys        = [0] * 8   # Keys the HUMAN is currently pressing
human_intervening = [False]   # True when human is providing input


def _build_monitor(sct, region_str):
    if region_str:
        r = tuple(int(x) for x in region_str.split(","))
        return {"left": r[0], "top": r[1], "width": r[2]-r[0], "height": r[3]-r[1]}
    return sct.monitors[1]


def dagger(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    model     = GameBotModel(num_frames=NUM_FRAMES).to(device)
    model_path = args.model
    loaded    = False

    if os.path.isfile(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"Loaded model: {model_path}")
        loaded = True

    if not loaded:
        ckpt = os.path.join("Data", "Checkpoints", "best.pt")
        if os.path.isfile(ckpt):
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model"] if "model" in state else state)
            print(f"Loaded checkpoint: {ckpt}")
        else:
            print("ERROR: No trained model found. Run  python train_new.py  first.")
            sys.exit(1)

    model.eval()

    # ── AI controllers ────────────────────────────────────────────────────
    keyboard  = Keyboard()
    mouse     = Mouse()
    held_keys = set()

    # ── Human input listeners ─────────────────────────────────────────────
    def on_human_key_press(key):
        action = pynput_key_to_action(key)
        if action is not None:
            idx = ACTION_TO_INDEX[action]
            with lock:
                human_keys[idx] = 1
                human_intervening[0] = True

    def on_human_key_release(key):
        action = pynput_key_to_action(key)
        if action is not None:
            idx = ACTION_TO_INDEX[action]
            with lock:
                human_keys[idx] = 0
            # Stop flagging as intervening if all human keys released
            with lock:
                if sum(human_keys) == 0:
                    human_intervening[0] = False

    def on_human_mouse_click(x, y, button, pressed):
        action = pynput_mouse_to_action(button)
        if action is not None:
            idx = ACTION_TO_INDEX[action]
            with lock:
                human_keys[idx] = 1 if pressed else 0
                if pressed:
                    human_intervening[0] = True
                elif sum(human_keys) == 0:
                    human_intervening[0] = False

    kl = KeyListener(on_press=on_human_key_press, on_release=on_human_key_release)
    ml = MouseListener(on_click=on_human_mouse_click)
    kl.daemon = True
    ml.daemon = True
    kl.start()
    ml.start()

    # ── Session output ────────────────────────────────────────────────────
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join("Data", "Recordings", f"dagger_{timestamp}")
    frames_dir  = os.path.join(session_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    csv_path = os.path.join(session_dir, "manifest.csv")
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.writer(csv_file)
    writer.writerow(["frame"] + [ACTIONS[i] for i in range(8)])

    # ── Screen capture + frame buffer ─────────────────────────────────────
    sct       = mss.mss()
    monitor   = _build_monitor(sct, args.region)
    interval  = 1.0 / args.fps
    threshold = args.threshold

    zero_frame = torch.zeros(1, 150, 150)
    frame_buf  = deque([zero_frame] * NUM_FRAMES, maxlen=NUM_FRAMES)

    # ── Countdown ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  GAME-BOT DAGGER  |  threshold={threshold}  fps={args.fps}")
    print(f"  Session: {session_dir}")
    print(f"{'='*60}")
    print("\n  How to use:")
    print("  • Watch the AI play")
    print("  • When AI makes a mistake → press the CORRECT key(s) yourself")
    print("  • Your input is recorded as expert correction data")
    print("  • Press Ctrl+C to stop, then retrain with:  python train_new.py --resume\n")
    print(f"Starting in {args.delay}s — switch to your game window!\n")

    for i in range(args.delay, 0, -1):
        print(f"  {i}…")
        time.sleep(1)
    print("  GO!\n")

    frame_count       = 0
    correction_count  = 0

    try:
        while True:
            t0 = time.perf_counter()

            # ── Capture ───────────────────────────────────────────────
            raw   = sct.grab(monitor)
            img   = Image.frombytes("RGB", (raw.width, raw.height), raw.rgb)
            frame = _FRAME_TRANSFORM(img)
            frame_buf.append(frame)

            # ── AI inference ──────────────────────────────────────────
            stacked = torch.cat(list(frame_buf), dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(stacked)).squeeze(0).cpu().numpy()

            ai_predicted = {ACTIONS[i] for i, p in enumerate(probs) if p >= threshold}

            # ── Determine which labels to save ────────────────────────
            with lock:
                is_human = human_intervening[0]
                h_keys   = list(human_keys)

            if is_human:
                # Human is correcting — save HUMAN's keys as ground truth
                save_labels = h_keys
                correction_count += 1
                active_note = "[CORRECTION]"
            else:
                # AI is on its own — let it act, save AI prediction as label
                save_labels = [1 if ACTIONS[i] in ai_predicted else 0 for i in range(8)]
                active_note = ""

            # ── Execute AI actions (regardless of human override) ─────
            # Note: human keys work naturally because the LISTENER already
            # fires real OS-level key events, which the game receives directly.
            # The AI controller handles the AI's own actions.
            for action in list(held_keys):
                if action not in ai_predicted:
                    obj = action_to_pynput(action)
                    (mouse.release if action.startswith("click_") else keyboard.release)(obj)
                    held_keys.discard(action)

            for action in ai_predicted:
                if action not in held_keys:
                    obj = action_to_pynput(action)
                    (mouse.press if action.startswith("click_") else keyboard.press)(obj)
                    held_keys.add(action)

            # ── Save frame ────────────────────────────────────────────
            # Save original RGB frame (record.py compatible format)
            png_img  = img.resize((150, 150), Image.LANCZOS)
            fname    = f"frame_{frame_count:08d}.png"
            png_img.save(os.path.join(frames_dir, fname))
            writer.writerow([os.path.join("frames", fname)] + save_labels)

            frame_count += 1
            if frame_count % 60 == 0:
                active = ", ".join(ai_predicted) or "none"
                print(f"  Frame {frame_count:6d} | AI: {active:<30} | Corrections: {correction_count} {active_note}")

            # ── Throttle ──────────────────────────────────────────────
            sleep = interval - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        pass
    finally:
        # Release all AI-held keys
        for action in list(held_keys):
            obj = action_to_pynput(action)
            (mouse.release if action.startswith("click_") else keyboard.release)(obj)
        csv_file.close()
        kl.stop()
        ml.stop()

        print(f"\n{'='*60}")
        print(f"  Stopped. {frame_count} frames saved, {correction_count} corrections.")
        print(f"  Session: {session_dir}")
        if correction_count > 0:
            print(f"\n  Now retrain to apply the corrections:")
            print(f"    python train_new.py --resume")
        else:
            print(f"\n  No corrections were made (you didn't press any keys).")
            print(f"  Try pressing the correct keys next time the AI makes a mistake!")
        print(f"{'='*60}")


def main():
    p = argparse.ArgumentParser(description="Game-Bot DAgger Trainer")
    p.add_argument("--model",     default="Data/Model/game_bot.pt")
    p.add_argument("--threshold", type=float, default=0.4,
                   help="AI action probability threshold (lower = more active AI)")
    p.add_argument("--fps",       type=int,   default=15)
    p.add_argument("--region",    default=None)
    p.add_argument("--delay",     type=int,   default=5)
    dagger(p.parse_args())

if __name__ == "__main__":
    main()
