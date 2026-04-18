# Game Bot
### Originally by Arda Mavi — Rewritten 

An AI that **learns to play any game by watching you**.  
Record your gameplay, train a neural network, then sit back and watch the AI play.

---

## How It Works

1. **Record** — You play the game while Game-Bot captures your screen and keyboard/mouse inputs at up to 60 FPS.
2. **Train** — A PyTorch CNN learns the mapping from screen pixels → actions (multi-label: multiple keys at once).
3. **Play** — The AI captures the screen in real-time, runs inference, and sends keyboard/mouse inputs just like you would.

## What's New (vs. Original)

| Feature | Original | This Version |
|---|---|---|
| Framework | Keras / TensorFlow 1.x | **PyTorch** |
| Screen Capture | PIL ImageGrab (~10 FPS) | **MSS (~60+ FPS)** |
| Model Output | Single Dense(4) sigmoid (broken) | **Multi-label 8-action head** |
| Actions | All keyboard keys (200+) | **8 game keys: W A S D Space Shift LClick RClick** |
| Data Format | Labels in filenames | **CSV manifest + PNG frames** |
| Training | No checkpoints, no scheduler | **AdamW, Cosine LR, best/latest checkpoints, resume** |
| Class Imbalance | Not handled | **pos_weight in BCEWithLogitsLoss** |
| AI Playback | No key release logic | **Proper press/release state tracking** |
| CLI | Hardcoded | **Full argparse with --fps, --region, --threshold, etc.** |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** For GPU acceleration, install PyTorch with CUDA support:  
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### 2. Record Training Data

Open your game, then in a terminal:

```bash
python record.py
```

Play the game for a while. Press `Ctrl+C` to stop recording.  
Data is saved to `Data/Recordings/session_<timestamp>/`.

**Options:**
```
--fps 30              Capture rate (default: 20)
--region 0,0,1920,1080  Capture a specific screen region
--test                Benchmark capture speed
```

### 3. Train the Model

```bash
python train_new.py
```

**Options:**
```
--epochs 200          Number of training epochs (default: 100)
--batch 64            Batch size (default: 32)
--lr 0.0005           Learning rate (default: 0.001)
--resume              Resume from latest checkpoint
--data-dir <path>     Custom data directory
```

### 4. Let the AI Play

Open your game, then:

```bash
python play.py
```

You'll get a 3-second countdown to switch to the game window.

**Options:**
```
--model <path>        Path to model weights
--threshold 0.6       Action probability threshold (default: 0.5)
--fps 20              Inference rate (default: 20)
--delay 5             Countdown seconds (default: 3)
--region ...          Screen capture region
```

## Project Structure

```
game_ai/
├── core/
│   ├── __init__.py
│   ├── model.py        # PyTorch CNN architecture
│   ├── dataset.py      # Dataset loader (CSV + PNG frames)
│   └── utils.py        # Action mappings & pynput helpers
├── record.py           # Screen + input recorder
├── train_new.py        # Training loop
├── play.py             # AI gameplay inference
├── requirements.txt
├── Data/
│   ├── Recordings/     # Recorded sessions (created by record.py)
│   ├── Checkpoints/    # Training checkpoints (created by train_new.py)
│   └── Model/          # Final exported model (created by train_new.py)
└── README.md
```

## Tracked Actions

The AI predicts 8 independent binary actions per frame:

| Index | Action | Typical Use |
|---|---|---|
| 0 | `W` | Move forward |
| 1 | `A` | Move left |
| 2 | `S` | Move backward |
| 3 | `D` | Move right |
| 4 | `Space` | Jump |
| 5 | `Shift` | Sprint / Crouch |
| 6 | `Left Click` | Shoot / Attack |
| 7 | `Right Click` | Aim / Block |

## Tips

- **Record multiple sessions** — more data = better AI. Sessions are automatically merged during training.
- **Use `--region`** to crop to just the game window for cleaner training data.
- **Lower `--threshold`** (e.g., 0.3) if the AI seems too passive; raise it (e.g., 0.7) if too trigger-happy.
- **Use TensorBoard** with `tensorboard --logdir Data/Checkpoints/` to track training loss.

## Requirements

- Python 3.8+
- Windows (uses MSS and pynput for screen capture and input simulation)
- GPU recommended for training (CPU works but slower)

## License

Apache License 2.0 — see [LICENSE](LICENSE)
