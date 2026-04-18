#!/usr/bin/env python3
"""
play.py — Let the Game-Bot AI play the game (with 4-frame stacking).

Usage:
    python play.py
    python play.py --model Data/Model/game_bot.pt
    python play.py --threshold 0.5 --fps 20 --delay 3
    python play.py --region 0,0,1920,1080
"""

import os
import sys
import time
import argparse
from collections import deque

import mss
import torch
from PIL import Image
from torchvision import transforms
from pynput.keyboard import Controller as Keyboard
from pynput.mouse import Controller as Mouse, Button

from core.model import GameBotModel, NUM_FRAMES
from core.utils import ACTIONS, action_to_pynput

# ── Frame pre-processing (must match core/dataset.py) ────────────────────────
_FRAME_TRANSFORM = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.Grayscale(),          # → 1-channel
    transforms.ToTensor(),           # → (1, 150, 150) float [0,1]
])


def play(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    model = GameBotModel(num_frames=NUM_FRAMES).to(device)

    model_path = args.model
    loaded = False

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

    # ── Controllers ───────────────────────────────────────────────────────
    keyboard   = Keyboard()
    mouse      = Mouse()
    held_keys  = set()   # track what we currently have pressed

    # ── Screen capture ────────────────────────────────────────────────────
    sct = mss.mss()
    if args.region:
        r       = tuple(int(x) for x in args.region.split(","))
        monitor = {"left": r[0], "top": r[1], "width": r[2]-r[0], "height": r[3]-r[1]}
    else:
        monitor = sct.monitors[1]

    # ── Rolling frame buffer ──────────────────────────────────────────────
    # Initialise with NUM_FRAMES zero frames so we can predict from frame 1
    zero_frame = torch.zeros(1, 150, 150)
    frame_buf  = deque([zero_frame] * NUM_FRAMES, maxlen=NUM_FRAMES)

    interval  = 1.0 / args.fps
    threshold = args.threshold

    # ── Countdown ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  GAME-BOT PLAYER  |  threshold={threshold}  fps={args.fps}")
    print(f"{'='*55}")
    print(f"\nStarting in {args.delay}s — switch to your game window!\n")
    for i in range(args.delay, 0, -1):
        print(f"  {i}…")
        time.sleep(1)
    print("  GO!\n")

    frame_count = 0
    try:
        while True:
            t0 = time.perf_counter()

            # ── Capture & preprocess ──────────────────────────────────
            raw   = sct.grab(monitor)
            img   = Image.frombytes("RGB", (raw.width, raw.height), raw.rgb)
            frame = _FRAME_TRANSFORM(img)          # (1, 150, 150)
            frame_buf.append(frame)

            # Stack into (NUM_FRAMES, 150, 150) and add batch dim
            stacked = torch.cat(list(frame_buf), dim=0)           # (4, 150, 150)
            tensor  = stacked.unsqueeze(0).to(device)              # (1, 4, 150, 150)

            # ── Inference ─────────────────────────────────────────────
            with torch.no_grad():
                logits = model(tensor)
                probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            predicted = {ACTIONS[i] for i, p in enumerate(probs) if p >= threshold}

            # ── Release keys no longer predicted ──────────────────────
            for action in list(held_keys):
                if action not in predicted:
                    obj = action_to_pynput(action)
                    (mouse.release if action.startswith("click_") else keyboard.release)(obj)
                    held_keys.discard(action)

            # ── Press newly predicted keys ─────────────────────────────
            for action in predicted:
                if action not in held_keys:
                    obj = action_to_pynput(action)
                    (mouse.press if action.startswith("click_") else keyboard.press)(obj)
                    held_keys.add(action)

            frame_count += 1
            if frame_count % 60 == 0:
                active = ", ".join(predicted) or "none"
                print(f"  Frame {frame_count:6d} | {active}")

            # ── Throttle ──────────────────────────────────────────────
            sleep = interval - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        pass
    finally:
        for action in list(held_keys):
            obj = action_to_pynput(action)
            (mouse.release if action.startswith("click_") else keyboard.release)(obj)
        print(f"\nStopped after {frame_count} frames. All inputs released.")


def main():
    p = argparse.ArgumentParser(description="Game-Bot AI Player")
    p.add_argument("--model",     default="Data/Model/game_bot.pt")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fps",       type=int,   default=20)
    p.add_argument("--region",    default=None)
    p.add_argument("--delay",     type=int,   default=3)
    play(p.parse_args())

if __name__ == "__main__":
    main()
