import os
# FORCE OpenMP COMPATIBILITY FOR MACOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import cv2
from PIL import Image
from embedder import CNNEmbedder
import torch.nn.functional as F
import torch

def calc_color_score(img1_path, img2_path):
    try:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None: return 0.0
        
        # Center Crop Both for fair comparison
        def crop_center(img):
            h, w, _ = img.shape
            cy, cx = h // 2, w // 2
            ch, cw = h // 2, w // 2 
            return img[cy - ch//2 : cy + ch//2, cx - cw//2 : cx + cw//2]

        img1 = crop_center(img1)
        img2 = crop_center(img2)
        
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        h_bins = 30
        s_bins = 32
        histSize = [h_bins, s_bins]
        ranges = [0, 180, 0, 256] 
        channels = [0, 1]
        
        hist1 = cv2.calcHist([hsv1], channels, None, histSize, ranges, accumulate=False)
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        hist2 = cv2.calcHist([hsv2], channels, None, histSize, ranges, accumulate=False)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0.0, score)
    except Exception as e:
        print(e)
        return 0.0

def main():
    paths = [
        "/Users/ajisampurno/.gemini/antigravity/brain/f97d3ede-b1a7-40a2-8f0c-e05f4bd5dfd8/uploaded_image_0_1769233294956.jpg", # Real Shirt
        "/Users/ajisampurno/.gemini/antigravity/brain/f97d3ede-b1a7-40a2-8f0c-e05f4bd5dfd8/uploaded_image_1_1769233294956.jpg", # Fabric Close-up
        "/Users/ajisampurno/.gemini/antigravity/brain/f97d3ede-b1a7-40a2-8f0c-e05f4bd5dfd8/uploaded_image_2_1769233294956.jpg"  # Mockup
    ]
    names = ["Real Shirt", "Fabric Close-up", "Mockup"]
    
    embedder = CNNEmbedder()
    
    print("Generating embeddings...")
    vecs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        vecs.append(embedder.encode(img))
    
    vecs = np.stack(vecs) # (3, 384)
    # Normalize for Cosine Similarity
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    
    print("\n--- Pairwise Analysis ---")
    print(f"{'Pair':<30} | {'Motif':<8} | {'Color':<8} | {'Final':<8}")
    print("-" * 65)
    
    pairs = [(0, 1), (0, 2), (1, 2)]
    
    for i, j in pairs:
        # Motif Score (Cosine Similarity)
        motif = np.dot(vecs[i], vecs[j])
        
        # Color Score
        color = calc_color_score(paths[i], paths[j])
        
        final = (motif * 0.5) + (color * 0.5)
        
        print(f"{names[i]} vs {names[j]:<15} | {motif:.4f}   | {color:.4f}   | {final:.4f}")

if __name__ == "__main__":
    main()
