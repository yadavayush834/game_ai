"""
core/dataset.py — Frame-stacked PyTorch Dataset for Game-Bot.

Each sample returns NUM_FRAMES consecutive grayscale frames stacked as a
single tensor (NUM_FRAMES, H, W) paired with the label of the LAST frame.
Frames at the start of a session are zero-padded.
"""

import os
import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset, random_split
from torchvision import transforms

from core.model import NUM_FRAMES

# ── Per-frame pre-processing ──────────────────────────────────────────────────
# Converted to grayscale; NOT normalised here — we normalise the stacked tensor
_FRAME_TRANSFORM = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.Grayscale(),               # RGB → 1-channel grayscale
    transforms.ToTensor(),                # (1, H, W), values in [0, 1]
])


class GameBotDataset(Dataset):
    """
    Reads a session's manifest.csv.

    CSV format (written by record.py):
        frame,w,a,s,d,space,shift,click_left,click_right
        frames/frame_00000000.png,0,0,0,0,0,0,0,0
        ...

    __getitem__(i) returns:
        frames_tensor : (NUM_FRAMES, H, W) float32   — stacked grayscale frames
        label         : (8,)              float32   — multi-label binary vector
    """

    def __init__(self, csv_path: str, num_frames: int = NUM_FRAMES):
        self.num_frames = num_frames
        self.base_dir = os.path.dirname(csv_path)
        self.frame_paths = []
        self.labels = []

        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)                           # skip header
            for row in reader:
                fp = row[0]
                if not os.path.isabs(fp):
                    fp = os.path.join(self.base_dir, fp)
                self.frame_paths.append(fp)
                self.labels.append([int(x) for x in row[1:]])

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_frame(self, path: str) -> torch.Tensor:
        """Load one PNG and apply the per-frame transform → (1, H, W)."""
        img = Image.open(path).convert('RGB')
        return _FRAME_TRANSFORM(img)               # (1, H, W)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int):
        # Build a window of self.num_frames consecutive frames.
        # Indices before the session start are zero-padded.
        frames = []
        for offset in range(self.num_frames - 1, -1, -1):  # oldest → newest
            src_idx = idx - offset
            if src_idx < 0:
                frames.append(torch.zeros(1, 150, 150))    # zero padding
            else:
                frames.append(self._load_frame(self.frame_paths[src_idx]))

        # Stack along channel dim → (NUM_FRAMES, H, W)
        stacked = torch.cat(frames, dim=0)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return stacked, label


# ── Utility: build train / val splits from all recorded sessions ──────────────

def get_train_val_datasets(data_dir: str = 'Data/Recordings',
                           val_split: float = 0.15,
                           num_frames: int = NUM_FRAMES):
    """
    Walks data_dir recursively and collects every manifest.csv.
    Merges all sessions into one ConcatDataset, then splits train/val.
    """
    all_csvs = []
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if f == 'manifest.csv':
                all_csvs.append(os.path.join(root, f))

    if not all_csvs:
        raise FileNotFoundError(
            f"No manifest.csv files found under '{data_dir}'.\n"
            "Run:  python record.py   to capture training data first."
        )

    datasets = [GameBotDataset(p, num_frames=num_frames) for p in all_csvs]
    merged = ConcatDataset(datasets)

    val_size   = max(1, int(len(merged) * val_split))
    train_size = len(merged) - val_size
    train_ds, val_ds = random_split(
        merged, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Found {len(all_csvs)} session(s) → {len(merged)} total samples")
    print(f"  Train: {train_size}  |  Val: {val_size}  |  Frames stacked: {num_frames}")
    return train_ds, val_ds
