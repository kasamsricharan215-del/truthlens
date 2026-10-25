import os
from pathlib import Path
import torch
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

def main():
    project_root = Path(__file__).resolve().parents[1]
    save_path = project_root / "models" / "image" / "image_model.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("Building EfficientNet-B0 image model...")
    # Load default pretrained ImageNet weights
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)

    # Initialize the 2-class head
    # Class 0: AI_GENERATED / FAKE, Class 1: REAL
    nn.init.xavier_uniform_(model.classifier[1].weight)
    nn.init.zeros_(model.classifier[1].bias)

    torch.save(model.state_dict(), str(save_path))
    print(f"Successfully saved image model weights to: {save_path}")

if __name__ == "__main__":
    main()
