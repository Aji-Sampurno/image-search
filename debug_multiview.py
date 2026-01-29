import torch
import numpy as np
from embedder import CNNEmbedder
from PIL import Image, ImageOps
import sys

def encode_robust(embedder, img_path):
    image = Image.open(img_path).convert("RGB")
    
    # 1. Base Image
    inputs = [image]
    
    # 2. Rotations
    inputs.append(image.rotate(90, expand=True))
    inputs.append(image.rotate(180, expand=True))
    inputs.append(image.rotate(270, expand=True))
    
    # 3. Flips
    inputs.append(ImageOps.mirror(image))
    inputs.append(ImageOps.flip(image))
    
    # 4. Zoom/Crop (Center Crop 75%)
    w, h = image.size
    crop_size = min(w, h) * 0.75
    left = (w - crop_size)/2
    top = (h - crop_size)/2
    inputs.append(image.crop((left, top, left+crop_size, top+crop_size)))
    
    vectors = []
    for img in inputs:
        # Preprocessing is handled inside embedder.get_transform (but we don't have access to it directly efficiently without mod)
        # We will use the embedder.encode but pass PIL image
        # Note: embedder.encode handles PIL input
        vec = embedder.encode(img)
        vectors.append(vec)
        
    # Average pooling
    final_vec = np.mean(vectors, axis=0)
    
    # L2 Normalize
    norm = np.linalg.norm(final_vec)
    return final_vec / norm

def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_multiview.py <img1> <img2>")
        sys.exit(1)
        
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    
    embedder = CNNEmbedder()
    
    # We apply robust encoding to BOTH? Or just one?
    # Ideally, if we index with robust encoding, we catch everything.
    # But re-indexing takes time. 
    # Let's try apply robust to BOTH to simulate if we updated the index and the query.
    
    print("Computing Robust Embeddings...")
    vec1 = encode_robust(embedder, img1_path)
    vec2 = encode_robust(embedder, img2_path)
    
    similarity = np.dot(vec1, vec2) * 100
    
    print(f"\nRobust Similarity Score: {similarity:.4f}%")

if __name__ == "__main__":
    main()
