# dataset.py — PyTorch Dataset for Game-Bot
import os
import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class GameBotDataset(Dataset):
    """
    Reads a CSV manifest created by record.py.
    Each row: frame_path, w, a, s, d, space, shift, click_left, click_right
    Labels are multi-label binary vectors (0 or 1 per action).
    """

    def __init__(self, csv_path, transform=None):
        self.samples = []
        self.base_dir = os.path.dirname(csv_path)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((150, 150)),
                transforms.ToTensor(),                       # HWC uint8 → CHW float [0,1]
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform

        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header
            for row in reader:
                frame_path = row[0]
                if not os.path.isabs(frame_path):
                    frame_path = os.path.join(self.base_dir, frame_path)
                labels = [int(x) for x in row[1:]]
                self.samples.append((frame_path, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, labels = self.samples[idx]
        image = Image.open(frame_path).convert('RGB')
        image = self.transform(image)
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels


def get_train_val_datasets(data_dir='Data/Recordings', val_split=0.15):
    """
    Scans *all* session CSVs inside data_dir, merges them into one large
    dataset, then splits into train / val.
    """
    from torch.utils.data import ConcatDataset, random_split

    all_csvs = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f == 'manifest.csv':
                all_csvs.append(os.path.join(root, f))

    if not all_csvs:
        raise FileNotFoundError(
            f"No manifest.csv files found under {data_dir}. "
            "Run record.py first to capture training data."
        )

    datasets = [GameBotDataset(csv_path) for csv_path in all_csvs]
    merged = ConcatDataset(datasets)

    val_size = max(1, int(len(merged) * val_split))
    train_size = len(merged) - val_size
    train_ds, val_ds = random_split(
        merged, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Loaded {len(merged)} samples from {len(all_csvs)} session(s).")
    print(f"  Train: {train_size}  |  Val: {val_size}")
    return train_ds, val_ds
