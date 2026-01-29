import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)

class TestEmbedder:
    def __init__(self):
        weights = models.ResNet50_Weights.DEFAULT
        base_model = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        self.pooling = GeM(p=3.0)
        self.model = nn.Sequential(self.backbone, self.pooling, nn.Flatten())
        self.model.eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def encode_centercrop(self, image):
        # Current method
        tfm = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            self.normalize
        ])
        t = tfm(image).unsqueeze(0)
        with torch.no_grad():
            feat = self.model(t)
        return self._post(feat)

    def encode_squash(self, image):
        # Resize whole image to 448x448 (Strech/Squash)
        tfm = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            self.normalize
        ])
        t = tfm(image).unsqueeze(0)
        with torch.no_grad():
            feat = self.model(t)
        return self._post(feat)

    def encode_fivecrop(self, image):
        # Resize to 512, then take 5 crops of 448
        tfm = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.FiveCrop(448),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([self.normalize(crop) for crop in crops]))
        ])
        crops = tfm(image) # (5, 3, 448, 448)
        with torch.no_grad():
            feats = self.model(crops) # (5, 2048)
        
        # Mean pooling
        feat = torch.mean(feats, dim=0).unsqueeze(0)
        return self._post(feat)

    def _post(self, feat):
        emb_np = feat.numpy().flatten()
        norm = np.linalg.norm(emb_np)
        if norm > 0: emb_np = emb_np / norm
        return emb_np

# === RUN TEST ===

QUERY_PATH = "/Users/ajisampurno/.gemini/antigravity/brain/e8a00380-afa9-4f8f-9003-b1b3263e0fb5/uploaded_image_1768988755116.png"
TARGET_PATH = "static/images/MJ221983.00 22198 KHAKI PERADA_2025-09-27.jpg"

embedder = TestEmbedder()
img_q = Image.open(QUERY_PATH).convert("RGB")
img_t = Image.open(TARGET_PATH).convert("RGB")

print("1. CenterCrop (Baseline):")
q = embedder.encode_centercrop(img_q)
t = embedder.encode_centercrop(img_t)
print(f"Score: {np.dot(q, t):.4f}")

print("\n2. Squash (No Crop):")
q = embedder.encode_squash(img_q)
t = embedder.encode_squash(img_t)
print(f"Score: {np.dot(q, t):.4f}")

print("\n3. FiveCrop (Averaging):")
q = embedder.encode_fivecrop(img_q)
t = embedder.encode_fivecrop(img_t)
print(f"Score: {np.dot(q, t):.4f}")
