# STEP 1: Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from config import HF_TOKEN
import os

# STEP 2: Load Dataset
if not HF_TOKEN:
    raise ValueError("❌ No HF_TOKEN found. Please add it in config.py")

dataset = load_dataset("ayshthkr/kolam_dataset", split="train", token=HF_TOKEN)

# STEP 3: Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def preprocess(example):
    images = []
    for img in example["image"]:  # ensure we loop through list
        if isinstance(img, Image.Image):
            pil_img = img
        else:
            pil_img = Image.fromarray(img)  # if numpy array or list
        pil_img = pil_img.convert("RGB")
        images.append(transform(pil_img))
    return {"pixel_values": torch.stack(images)}

dataset = dataset.with_transform(preprocess)

# Create dataloaders
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# STEP 4: Define CNN (Feature Extractor)
class KolamCNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(KolamCNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        batch_size, channels, h, w = x.size()
        x = x.view(batch_size, channels, -1)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        return x

# Initialize model
cnn = KolamCNNFeatureExtractor()

# STEP 5: Try one batch (just to see output shape)
for batch in train_loader:
    images = batch["pixel_values"]
    features = cnn(images)  # CNN output → for ViT later
    print("CNN output shape:", features.shape)

    image0_features = features[0]  # first image
    print(image0_features.shape)
    
    channel0 = image0_features[0]  # first channel
    print(channel0.shape)           # torch.Size([64])
    print(channel0)
    break
