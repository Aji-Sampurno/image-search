import torch
import numpy as np
import cv2
from embedder import CNNEmbedder
from PIL import Image, ImageOps
import sys

def apply_clahe(pil_img):
    img = np.array(pil_img)
    # Check if RGB
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final)
    else:
        # Grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return Image.fromarray(clahe.apply(img))

def encode_robust_with_clahe(embedder, img_path):
    image = Image.open(img_path).convert("RGB")
    
    # Apply CLAHE to the base image?
    # Usually we want to augment: Original + CLAHE version?
    # Or just use CLAHE for everything?
    # Let's try adding CLAHE versions to the mix.
    
    inputs = []
    
    # 1. Base Variants
    inputs.append(image)
    inputs.append(apply_clahe(image)) # Add CLAHE version
    
    encoded_features = []
    
    for base in inputs:
        # Rotations
        encoded_features.append(embedder.encode(base))
        encoded_features.append(embedder.encode(base.rotate(90, expand=True)))
        encoded_features.append(embedder.encode(base.rotate(180, expand=True)))
        encoded_features.append(embedder.encode(base.rotate(270, expand=True)))
        
        # Flips
        encoded_features.append(embedder.encode(ImageOps.mirror(base)))
        
        # Crop (Zoom in)
        w, h = base.size
        c = min(w, h) * 0.75
        left = (w - c)/2
        top = (h - c)/2
        crop = base.crop((left, top, left+c, top+c))
        encoded_features.append(embedder.encode(crop))

    final_vec = np.mean(encoded_features, axis=0)
    norm = np.linalg.norm(final_vec)
    return final_vec / norm

def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_clahe.py <img1> <img2>")
        sys.exit(1)
        
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    
    embedder = CNNEmbedder()
    
    print("Computing Robust Embeddings (with CLAHE)...")
    vec1 = encode_robust_with_clahe(embedder, img1_path)
    vec2 = encode_robust_with_clahe(embedder, img2_path)
    
    similarity = np.dot(vec1, vec2) * 100
    
    print(f"\nRobust + CLAHE Similarity Score: {similarity:.4f}%")

if __name__ == "__main__":
    main()
