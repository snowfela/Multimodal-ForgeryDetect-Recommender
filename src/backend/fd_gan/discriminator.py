import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Spatial branch: processes spatial features
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # Example conv layer
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # Flatten the output to feed into a linear layer
            nn.Linear(128 * 64 * 64, 1)  # Example fully connected layer (adjust size accordingly)
        )
        # Frequency branch: processes frequency features
        self.frequency_branch = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # Example conv layer
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # Flatten the output to feed into a linear layer
            nn.Linear(128 * 64 * 64, 1)  # Example fully connected layer (adjust size accordingly)
        )
        self.low_rank_module = nn.Conv2d(64, 64, kernel_size=1)  # Simplified LRM for example
    def forward(self, image):
        # Pass the image through spatial and frequency branches
        spatial_features = self.spatial_branch(image)
        frequency_features = self.frequency_branch(image)

        return spatial_features, frequency_features
