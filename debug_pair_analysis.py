import cv2
import numpy as np
import os
from PIL import Image
from embedder import CNNEmbedder
import torch

# Define Paths
# Using the SAGE Physical Photo as a PROXY for the user's query (which was also a physical photo)
QUERY_PATH = "static/images/H723HEM 22198 SAGE_2025-12-12.jpg" 
TARGET_DIR = "static/images"
TARGET_FILES = [
    "MJ2219822198 KHAKI PERADA_2025-06-12.jpg", # The Mockup
    "Buring_H723_HEM 22198 KHAKI.jpg", # Is this physical?
]

def calc_color_score(img1_cv, img2_path, is_query=False):
    try:
        img2_cv = cv2.imread(img2_path)
        if img2_cv is None: return 0.0
        
        if is_query:
            h, w, _ = img1_cv.shape
            cy, cx = h // 2, w // 2
            ch, cw = h // 2, w // 2 
            img1_cv = img1_cv[cy - ch//2 : cy + ch//2, cx - cw//2 : cx + cw//2]
        
        hsv1 = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2HSV)
        
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
        print(f"Color Error: {e}")
        return 0.0

def main():
    print(f"Proxy Query (Physical SAGE): {QUERY_PATH}")
    
    # Init Embedder
    embedder = CNNEmbedder()
    
    if not os.path.exists(QUERY_PATH):
        print(f"Error: Query path {QUERY_PATH} does not exist.")
        return

    # Load Query Image
    query_pil = Image.open(QUERY_PATH).convert("RGB")
    query_cv = cv2.imread(QUERY_PATH)
    
    # Query Embedding
    query_emb = embedder.encode(query_pil)
    
    print("-" * 60)
    print(f"{'Filename':<40} | {'Motif':<10} | {'Color':<10} | {'Final':<10}")
    print("-" * 60)
    
    for fname in TARGET_FILES:
        target_path = os.path.join(TARGET_DIR, fname)
        if not os.path.exists(target_path):
            print(f"File not found: {fname}")
            continue
            
        # Target Embedding
        target_pil = Image.open(target_path).convert("RGB")
        target_emb = embedder.encode(target_pil)
        
        # Cosine Similarity
        query_norm = query_emb / np.linalg.norm(query_emb)
        target_norm = target_emb / np.linalg.norm(target_emb)
        motif_score = np.dot(query_norm, target_norm)
        
        # Color Score
        color_score = calc_color_score(query_cv, target_path, is_query=True)
        
        # Final Score (50-50)
        final_score = (motif_score * 0.5) + (color_score * 0.5)
        
        print(f"{fname[:40]:<40} | {motif_score:.4f}     | {color_score:.4f}     | {final_score:.4f}")

if __name__ == "__main__":
    main()
