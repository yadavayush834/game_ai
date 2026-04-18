#!/usr/bin/env python3
"""
play.py — Let the Game-Bot AI play the game.

Usage:
    python play.py                              # Use best model
    python play.py --model Data/Model/game_bot.pt
    python play.py --threshold 0.6              # Higher threshold = fewer actions
    python play.py --fps 20                     # Inference rate cap
    python play.py --region 0,0,1920,1080       # Capture specific region
    python play.py --delay 3                    # Countdown before starting (default 3)
"""

import os
import sys
import time
import argparse
import numpy as np
from PIL import Image

import mss
import torch
from torchvision import transforms
from pynput.keyboard import Controller as Keyboard
from pynput.mouse import Controller as Mouse, Button

from core.model import GameBotModel
from core.utils import ACTIONS, action_to_pynput

# ── Transform (must match training) ──────────────────────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def play(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    model = GameBotModel(num_actions=8).to(device)

    model_path = args.model
    if not os.path.isfile(model_path):
        # Try best checkpoint fallback
        alt = os.path.join("Data", "Checkpoints", "best.pt")
        if os.path.isfile(alt):
            ckpt = torch.load(alt, map_location=device)
            model.load_state_dict(ckpt["model"])
            print(f"Loaded model from checkpoint: {alt}")
        else:
            print(f"ERROR: Model not found at '{model_path}' or '{alt}'.")
            print("Train a model first with: python train_new.py")
            sys.exit(1)
    else:
        state = torch.load(model_path, map_location=device)
        # Handle both raw state_dict and checkpoint dict
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
        print(f"Loaded model from: {model_path}")

    model.eval()

    # ── Input controllers ─────────────────────────────────────────────────
    keyboard = Keyboard()
    mouse = Mouse()

    # Track currently held keys so we can release them
    currently_held = set()

    # ── Screen capture ────────────────────────────────────────────────────
    sct = mss.mss()
    region = None
    if args.region:
        r = tuple(int(x) for x in args.region.split(","))
        region = {"left": r[0], "top": r[1], "width": r[2] - r[0], "height": r[3] - r[1]}
    monitor = region if region else sct.monitors[1]

    interval = 1.0 / args.fps
    threshold = args.threshold

    # ── Countdown ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  GAME-BOT AI PLAYER")
    print(f"  Threshold: {threshold}  |  FPS: {args.fps}")
    print(f"{'='*60}")
    print(f"\nStarting in {args.delay} seconds — switch to your game window!\n")
    for i in range(args.delay, 0, -1):
        print(f"  {i}…")
        time.sleep(1)
    print("  GO!\n")

    frame_count = 0
    try:
        while True:
            t0 = time.perf_counter()

            # Capture screen
            raw = sct.grab(monitor)
            img = Image.frombytes("RGB", (raw.width, raw.height), raw.rgb)

            # Preprocess
            tensor = inference_transform(img).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            # Decide which actions to trigger
            predicted_actions = set()
            for i, prob in enumerate(probs):
                action_name = ACTIONS[i]
                if prob >= threshold:
                    predicted_actions.add(action_name)

            # ── Execute actions ───────────────────────────────────────
            # Release keys no longer predicted
            for action_name in list(currently_held):
                if action_name not in predicted_actions:
                    pynput_obj = action_to_pynput(action_name)
                    if action_name.startswith("click_"):
                        mouse.release(pynput_obj)
                    else:
                        keyboard.release(pynput_obj)
                    currently_held.discard(action_name)

            # Press newly predicted keys
            for action_name in predicted_actions:
                if action_name not in currently_held:
                    pynput_obj = action_to_pynput(action_name)
                    if action_name.startswith("click_"):
                        mouse.press(pynput_obj)
                    else:
                        keyboard.press(pynput_obj)
                    currently_held.add(action_name)

            frame_count += 1
            if frame_count % 60 == 0:
                active = ", ".join(predicted_actions) if predicted_actions else "none"
                print(f"  Frame {frame_count} | Active: {active}")

            # Throttle
            elapsed = time.perf_counter() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        # Release all held keys on exit
        for action_name in list(currently_held):
            pynput_obj = action_to_pynput(action_name)
            if action_name.startswith("click_"):
                mouse.release(pynput_obj)
            else:
                keyboard.release(pynput_obj)
        print(f"\nStopped after {frame_count} frames. All keys released.")


def main():
    parser = argparse.ArgumentParser(description="Game-Bot AI Player")
    parser.add_argument("--model", type=str, default="Data/Model/game_bot.pt",
                        help="Path to trained model weights")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold to trigger an action (default: 0.5)")
    parser.add_argument("--fps", type=int, default=20,
                        help="Inference rate cap (default: 20)")
    parser.add_argument("--region", type=str, default=None,
                        help="Screen region as left,top,right,bottom")
    parser.add_argument("--delay", type=int, default=3,
                        help="Countdown seconds before AI starts (default: 3)")
    args = parser.parse_args()
    play(args)


if __name__ == "__main__":
    main()
