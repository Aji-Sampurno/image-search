import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps # Added ImageOps
import numpy as np
import ssl

# Bypass SSL verification for model download
ssl._create_default_https_context = ssl._create_unverified_context

class CNNEmbedder:
    def __init__(self):
        print("Loading DINOv2 (ViT-S/14) model...")
        # Load DINOv2 Small (ViT-S/14) - 384 dim
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.eval()
        
        # DINOv2 Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(518), # Resize to valid patch multiple (14*37=518)
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def encode(self, image_input):
        """
        Generates Multi-View Robust embedding using DINOv2
        Views: Original, Rot 90, 180, 270, Mirror, CenterCrop
        Arg:
            image_input: PIL Image or path to image
        Returns:
            np.array (float32) of shape (384,) normalized L2
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        # Create Views
        views = []
        
        # 1. Original
        views.append(image)
        
        # 2. Rotations
        views.append(image.rotate(90, expand=True))
        views.append(image.rotate(180, expand=True))
        views.append(image.rotate(270, expand=True))
        
        # 3. Mirror
        views.append(ImageOps.mirror(image))
        
        # 4. Zoom (Center Crop 75%)
        w, h = image.size
        c = min(w, h) * 0.75
        left = (w - c)/2
        top = (h - c)/2
        views.append(image.crop((left, top, left+c, top+c)))
        
        # 5. Macro Zoom (Center Crop 50%) - To match close-up fabrics
        c2 = min(w, h) * 0.50
        left2 = (w - c2)/2
        top2 = (h - c2)/2
        views.append(image.crop((left2, top2, left2+c2, top2+c2)))
        
        # Batch Process
        tensors = []
        for v in views:
            tensors.append(self.transform(v))
            
        batch = torch.stack(tensors) # (6, 3, 518, 518)
        
        with torch.no_grad():
            features = self.model(batch) # (6, 384)
            
        # Average Pooling
        avg_feature = torch.mean(features, dim=0) # (384,)
        
        emb_np = avg_feature.numpy()
        
        # L2 Normalize
        norm = np.linalg.norm(emb_np)
        if norm > 0:
            emb_np = emb_np / norm
            
        return emb_np.astype(np.float32)

if __name__ == "__main__":
    embedder = CNNEmbedder()
    dummy = Image.new('RGB', (518, 518), color='red')
    vector = embedder.encode(dummy)
    print(f"Vector shape: {vector.shape}") # Should be (384,)
