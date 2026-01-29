import torch
import numpy as np
from embedder import CNNEmbedder
from PIL import Image
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_pair.py <img1> <img2>")
        sys.exit(1)
        
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    
    embedder = CNNEmbedder()
    
    vec1 = embedder.encode(img1_path)
    vec2 = embedder.encode(img2_path)
    
    # Cosine Similarity (vectors are already L2 normalized)
    similarity = np.dot(vec1, vec2) * 100
    
    print(f"\nSimilarity between:")
    print(f"1. {img1_path}")
    print(f"2. {img2_path}")
    print(f"Score: {similarity:.4f}%")

if __name__ == "__main__":
    main()
