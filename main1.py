import os
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm  # progress bar
import torch

# STEP 1 — Extract ZIP file
zip_path = "kolam_organized.zip"  # your dataset zip
extract_dir = "kolam_dataset"

if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        print(f"✅ Extracted dataset to {extract_dir}")
else:
    print("📦 Dataset already extracted")

# STEP 2 — Define transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# STEP 3 — Load dataset
dataset = datasets.ImageFolder(root=extract_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=False)
print(f"✅ Loaded {len(dataset)} images from {extract_dir}")

# STEP 4 — Define CNN feature extractor
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
        x = x.view(batch_size, channels, -1)  # flatten spatial dimensions
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        return x

# STEP 5 — Initialize CNN
cnn = KolamCNNFeatureExtractor()

# STEP 6 — Extract and store features
all_features = []
all_labels = []

print("🔄 Extracting CNN feature maps...")
for images, labels in tqdm(train_loader):
    with torch.no_grad():
        feats = cnn(images)  # (batch, seq_len, features)
        all_features.append(feats)
        all_labels.append(labels)

# Combine all batches
all_features = torch.cat(all_features, dim=0)
all_labels = torch.cat(all_labels, dim=0)

print(f"\n✅ Final feature tensor shape: {all_features.shape}")
print(f"✅ Labels shape: {all_labels.shape}")

# STEP 7 — Save to file
save_path = "kolam_features.pt"
torch.save({"features": all_features, "labels": all_labels}, save_path)
print(f"💾 Saved features to {save_path}")

data = torch.load("kolam_features.pt")
features = data["features"]  # shape: [939, 1024, 64]
labels = data["labels"]      # shape: [939]

print(features.shape)
print(labels.shape)

img1_features = features[0]  # shape: [1024, 64]
img2_features = features[1]  # shape: [1024, 64]

print("Image 1 features shape:", img1_features.shape)
print("Image 2 features shape:", img2_features.shape)

# Print first token (patch) feature vector of image 1
print(img1_features[0])

# Print the first 5 token vectors
print(img1_features[:100])
