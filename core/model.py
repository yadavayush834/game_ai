import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

NUM_FRAMES = 4   # how many consecutive frames to stack
NUM_ACTIONS = 10  # w, a, s, d, space, shift, click_left, click_right, up, down


class GameBotModel(nn.Module):
    """
    Frame-stacked MobileNetV2 Game Bot.

    Input:  (B, NUM_FRAMES, H, W)  — stacked grayscale frames
    Output: (B, NUM_ACTIONS)       — raw logits for multi-label classification
    """

    def __init__(self, num_actions=NUM_ACTIONS, num_frames=NUM_FRAMES, freeze_backbone=True):
        super().__init__()

        # ── 1. Project NUM_FRAMES grayscale channels → 3 for MobileNetV2 ──
        self.frame_proj = nn.Sequential(
            nn.Conv2d(num_frames, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        # ── 2. Pretrained MobileNetV2 backbone ────────────────────────────
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features  # outputs (B, 1280, h, w)

        if freeze_backbone:
            # Freeze all layers except the last conv block so we only fine-tune
            trainable_from = 14  # unfreeze blocks 14-18 (last ~30%)
            for i, layer in enumerate(self.features):
                requires_grad = (i >= trainable_from)
                for p in layer.parameters():
                    p.requires_grad = requires_grad

        # ── 3. Classification head ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      # (B, 1280, 1, 1)
            nn.Flatten(),                  # (B, 1280)
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        # x: (B, NUM_FRAMES, H, W)
        x = self.frame_proj(x)   # → (B, 3, H, W)
        x = self.features(x)     # → (B, 1280, h, w)
        x = self.classifier(x)   # → (B, NUM_ACTIONS)  logits
        return x


if __name__ == '__main__':
    model = GameBotModel()
    test = torch.randn(2, NUM_FRAMES, 150, 150)
    out = model(test)
    print(f'Input:  {test.shape}')
    print(f'Output: {out.shape}')
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'Params: {trainable:,} trainable / {total:,} total')
