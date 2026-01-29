import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F

# === REPLICATE EMBEDDER LOGIC WITH CLAHE ===

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

    def apply_clahe(self, image_path):
        # Read as Lab
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final)

    def get_transform(self, size):
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            self.normalize
        ])

    def encode(self, image_path, use_clahe=False):
        if use_clahe:
            image = self.apply_clahe(image_path)
        else:
            image = Image.open(image_path).convert("RGB")
            
        base_size = 512
        scales = [base_size, int(base_size / 1.414), int(base_size * 1.414)]
        features = torch.zeros(1, 2048)
        
        with torch.no_grad():
            for s in scales:
                tfm = self.get_transform(s)
                tensor = tfm(image).unsqueeze(0)
                features += self.model(tensor)
                
        features = features / len(scales)
        emb_np = features.numpy().flatten()
        norm = np.linalg.norm(emb_np)
        if norm > 0: emb_np = emb_np / norm
        return emb_np

# === RUN TEST ===

QUERY_PATH = "/Users/ajisampurno/.gemini/antigravity/brain/e8a00380-afa9-4f8f-9003-b1b3263e0fb5/uploaded_image_1768988755116.png"
# This is the file the user wants to be top 10
TARGET_PATH = "static/images/MJ221983.00 22198 KHAKI PERADA_2025-09-27.jpg"

embedder = TestEmbedder()

print("Calculating BASELINE (No CLAHE)...")
bq = embedder.encode(QUERY_PATH, use_clahe=False)
bt = embedder.encode(TARGET_PATH, use_clahe=False)
base_score = np.dot(bq, bt)
print(f"Baseline Score: {base_score:.4f}")

print("\nCalculating WITH CLAHE...")
cq = embedder.encode(QUERY_PATH, use_clahe=True)
ct = embedder.encode(TARGET_PATH, use_clahe=True)
clahe_score = np.dot(cq, ct)
print(f"CLAHE Score:    {clahe_score:.4f}")

improvement = clahe_score - base_score
print(f"\nImprovement: {improvement:.4f}")

if improvement > 0.02:
    print("SUCCESS: CLAHE significantly improves matching.")
else:
    print("FAILURE: CLAHE had minimal or negative effect.")
