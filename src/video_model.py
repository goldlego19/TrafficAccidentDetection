"""
End-to-End 3D CNN for Accident Detection
Uses R(2+1)D to process spatial and temporal visual features simultaneously.
"""

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class VideoAccidentDetector(nn.Module):
    def __init__(self, pretrained=True, dropout=0.5):
        super(VideoAccidentDetector, self).__init__()
        
        # Load pre-trained kinetics backbone
        weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
        self.backbone = r2plus1d_18(weights=weights)
        
        # Replace the final fully connected layer for binary classification
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Expected input shape: (Batch, Channels, Time, Height, Width)
        return self.backbone(x)

if __name__ == "__main__":
    # Test the model shape
    model = VideoAccidentDetector()
    dummy_input = torch.randn(2, 3, 16, 112, 112) # Batch=2, RGB, 16 frames, 112x112
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be [2, 1]