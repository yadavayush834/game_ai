import torch
import torch.nn as nn
import torch.nn.functional as F

class GameBotModel(nn.Module):
    def __init__(self, num_actions=8):
        super(GameBotModel, self).__init__()
        # Input: 3 x 150 x 150
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2) # Output: 32 x 38 x 38
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # Output: 64 x 19 x 19
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # Output: 64 x 19 x 19
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 64 x 9 x 9
        
        # Calculate flattened size
        # 64 * 9 * 9 = 5184
        self.fc1 = nn.Linear(5184, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Logits for multi-label classification
        # Note: We do not apply sigmoid here.
        # We will use BCEWithLogitsLoss during training, which is numerically more stable.
        # During inference (play.py), we will apply torch.sigmoid(x).
        return x

if __name__ == '__main__':
    # Test model
    model = GameBotModel()
    test_input = torch.randn(1, 3, 150, 150)
    output = model(test_input)
    print("Output shape:", output.shape)
