#Game-Bot — AI That Learns to Play Any Game by Watching You

An AI agent that watches you play a game, learns your moves, then plays the game itself.
Built with PyTorch, MobileNetV2, frame stacking, and DAgger iterative correction.

---

## How It Works

```
You play → Bot records screen + keys → Neural network trains → Bot plays
```

1. **Record** — You play a game. The bot captures your screen at up to 60 FPS and logs every key/mouse action you take simultaneously.
2. **Train** — A MobileNetV2 CNN learns to map 4 stacked screen frames → which keys to press.
3. **Play** — The AI watches the screen in real-time and presses keys just like you did.
4. **Correct (DAgger)** — Watch the AI play. Press the right key whenever it makes a mistake. Those corrections become new training data.

---

## Comparison vs Original (ardamavi/Game-Bot)

| Feature | Original (2017) | This Version |
|---|---|---|
| Framework | Keras / TensorFlow 1.x | **PyTorch + MobileNetV2** |
| Model | Custom CNN, `Dense(4)` — broken output | **Multi-label head, 10 independent actions** |
| Screen capture | PIL `ImageGrab` (~10 FPS) | **MSS (~60+ FPS)** |
| Temporal context | None (single frame) | **4-frame stacking (model sees motion)** |
| Tracked actions | All 200+ keyboard keys (impossible to learn) | **10 game-relevant actions: W A S D Up Down Space Shift LClick RClick** |
| Data format | Labels embedded in filenames (breaks on multi-key) | **CSV manifest + PNG frames** |
| Multi-key support | No | **Yes — multiple keys simultaneously** |
| Training | No scheduler, no checkpoint | **AdamW, cosine LR decay, best/latest checkpoints, resume** |
| Class imbalance | Not handled | **`pos_weight` in BCEWithLogitsLoss** |
| AI playback | No key-release logic (keys get stuck) | **Full press/release state tracking** |
| Iterative improvement | None | **DAgger: correct AI mistakes in real-time** |
| CLI | Hardcoded | **Full argparse on all scripts** |
| Environment | Global Python | **Virtual environment (`venv/`)** |

---

## Tracked Actions

The AI predicts 10 independent binary actions per frame:

| Index | Action | Key | Primary Use |
|---|---|---|---|
| 0 | `w` | W | Move forward |
| 1 | `a` | A | Move left |
| 2 | `s` | S | Move backward |
| 3 | `d` | D | Move right |
| 4 | `space` | Space | Jump (many games) |
| 5 | `shift` | Shift | Sprint / Crouch |
| 6 | `click_left` | Left Click | Shoot / Attack |
| 7 | `click_right` | Right Click | Aim / Block |
| 8 | `up` | ↑ Up Arrow | Jump (Chrome Dino) |
| 9 | `down` | ↓ Down Arrow | Duck (Chrome Dino) |

> **Adding your own keys:** Edit `core/utils.py` and add entries to `ACTIONS`, `pynput_key_to_action`, and `action_to_pynput`. Update `NUM_ACTIONS` in `core/model.py` to match.

---

## Project Structure

```
game_ai/
├── core/
│   ├── __init__.py
│   ├── model.py        # MobileNetV2 + frame projection + classification head
│   ├── dataset.py      # Frame-stacked dataset (CSV + PNG), train/val split
│   └── utils.py        # Action mappings, pynput converters
├── record.py           # Screen + input recorder (MSS, up to 60 FPS)
├── train_new.py        # PyTorch training loop (AdamW, cosine LR, checkpointing)
├── play.py             # Real-time AI inference + input simulation
├── dagger.py           # DAgger: AI plays, you correct its mistakes
├── venv/               # Python virtual environment
├── requirements.txt
├── README.md
└── Data/               # Created automatically at runtime
    ├── Recordings/     # Recorded sessions (record.py and dagger.py)
    ├── Checkpoints/    # Training checkpoints (train_new.py)
    └── Model/          # Final exported model (train_new.py)
```

---

## Quick Start

### Prerequisites
- Python 3.8+
- Windows (MSS + pynput screen/input access)
- GPU optional but recommended for training

### 1. Set Up Environment

```powershell
# Clone & enter the project
git clone https://github.com/yadavayush834/game_ai.git
cd game_ai

# Create & activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> For GPU (CUDA) support, replace the torch install with:
> ```
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Beginner Walkthrough — Chrome Dino

**Best first game:** Open Chrome and go to `chrome://dino` (or disconnect your internet). It uses **Up Arrow** to jump and **Down Arrow** to duck — simple to learn.

### Step 1 — Record Your Gameplay

```powershell
.\venv\Scripts\activate
python record.py --fps 10
```

Switch to the Chrome Dino window and play for **3–5 minutes**. The bot silently saves screenshots + your keystrokes. Press `Ctrl+C` in the terminal when done.

> **Important:** Use the keys that are in the tracked actions list above. Chrome Dino uses `Up` to jump — this is now tracked.

### Step 2 — Train the Model

```powershell
python train_new.py --epochs 50 --batch 8
```

You'll see output like:
```
Epoch    1/50 | Train: 0.65 | Val: 0.62 | Acc: 0.74 | LR: 1.00e-03 | 42s
Epoch    2/50 | Train: 0.58 | Val: 0.54 | Acc: 0.81 | LR: 9.90e-04 | 40s
★ New best model saved (val_loss=0.54)
```

Loss going down + accuracy going up = the AI is learning. Training takes ~5–20 minutes on CPU.

### Step 3 — Let the AI Play

```powershell
python play.py --threshold 0.4 --delay 5
```

You have **5 seconds** to click on the Chrome Dino window before the AI starts. Watch it play!

Press `Ctrl+C` to stop.

### Step 4 — Improve With DAgger

When the AI makes obvious mistakes (e.g., not jumping over a cactus), use DAgger:

```powershell
python dagger.py --threshold 0.4 --delay 5
```

- Watch the AI play
- When it's about to crash without jumping → **press Up yourself**
- Your correction is saved automatically
- Press `Ctrl+C` when done, then retrain:

```powershell
python train_new.py --epochs 50 --resume
```

Repeat this loop — the AI improves each round.

---

## Command Reference

```powershell
# Always activate first:
.\venv\Scripts\activate

# Record gameplay (use --fps 10 for CPU, --fps 30 for GPU):
python record.py --fps 10
python record.py --fps 10 --region 0,0,1920,1080  # specific screen area

# Train:
python train_new.py --epochs 50 --batch 8
python train_new.py --epochs 100 --resume          # continue from checkpoint

# AI plays:
python play.py --threshold 0.4 --delay 5
python play.py --threshold 0.3                     # more aggressive AI
python play.py --threshold 0.7                     # more conservative AI

# DAgger correction:
python dagger.py --threshold 0.4 --delay 5

# Benchmark screen capture speed:
python record.py --test
```

---

## Tips for Better Results

| Tip | Detail |
|---|---|
| **Record more** | 10–20 minutes of gameplay beats 2 minutes every time |
| **Multiple sessions** | All sessions are merged automatically during training |
| **Vary your play** | Include easy AND hard moments — obstacles at different distances |
| **Threshold tuning** | If AI does nothing → lower `--threshold`. If pressing randomly → raise it |
| **Capture only the game** | Use `--region` to crop to the game window for cleaner training data |
| **Re-record if using wrong keys** | If you used keys not in the tracked list, delete the session and re-record |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No manifest.csv found` | Run `python record.py` first |
| `Model not found` | Run `python train_new.py` first |
| `ModuleNotFoundError` | You forgot `.\venv\Scripts\activate` |
| AI does nothing at all | Lower threshold: `--threshold 0.2` |
| Keys get stuck after stopping | Script already handles this — all keys released on `Ctrl+C` |
| Training loss not improving | Record more diverse data; try more epochs |
| Old model has wrong output size | Delete `Data/Checkpoints` and `Data/Model`, retrain from scratch |
| Changed keys mid-project | Delete old recordings — old CSVs have fewer columns than the new model expects |

---

## Requirements

```
torch
torchvision
numpy
pillow
mss
pynput
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE)

Original concept by [Arda Mavi](https://github.com/ardamavi/Game-Bot). This is a ground-up rewrite.
