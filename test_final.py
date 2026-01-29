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

    def get_transform_squash(self, size):
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            self.normalize
        ])

    def get_transform_free(self, size):
        return transforms.Compose([
            transforms.Resize(size), # Keep AR
            transforms.ToTensor(),
            self.normalize
        ])

    def encode(self, image, mode='squash'):
        base_size = 512
        scales = [base_size, int(base_size / 1.414), int(base_size * 1.414)]
        features = torch.zeros(1, 2048)
        
        with torch.no_grad():
            for s in scales:
                if mode == 'squash':
                    tfm = self.get_transform_squash(s)
                else:
                    tfm = self.get_transform_free(s)
                    
                tensor = tfm(image).unsqueeze(0)
                features += self.model(tensor)
                
        features = features / len(scales)
        emb_np = features.numpy().flatten()
        norm = np.linalg.norm(emb_np)
        if norm > 0: emb_np = emb_np / norm
        return emb_np

# === RUN TEST ===

QUERY_PATH = "/Users/ajisampurno/.gemini/antigravity/brain/e8a00380-afa9-4f8f-9003-b1b3263e0fb5/uploaded_image_1768988755116.png"
TARGET_PATH = "static/images/MJ221983.00 22198 KHAKI PERADA_2025-09-27.jpg"

embedder = TestEmbedder()
img_q = Image.open(QUERY_PATH).convert("RGB")
img_t = Image.open(TARGET_PATH).convert("RGB")

print("1. Multi-Scale SQUASH (Baseline):")
q = embedder.encode(img_q, mode='squash')
t = embedder.encode(img_t, mode='squash')
print(f"Score: {np.dot(q, t):.4f}")

print("\n2. Multi-Scale FREE AR (Proposed):")
q = embedder.encode(img_q, mode='free')
t = embedder.encode(img_t, mode='free')
print(f"Score: {np.dot(q, t):.4f}")
