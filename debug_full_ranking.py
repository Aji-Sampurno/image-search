import os
# FORCE OpenMP COMPATIBILITY FOR MACOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
import pickle
from PIL import Image
from embedder import CNNEmbedder
import faiss

# Configuration
DATA_DIR = "data"
VECTORS_FILE = os.path.join(DATA_DIR, "batik_vectors.npy")
PATHS_FILE = os.path.join(DATA_DIR, "batik_paths.pkl")
INDEX_FILE = os.path.join(DATA_DIR, "batik.faiss")

# Proxy Query: The SAGE physical photo which is visually similar to the input
QUERY_PATH = "static/images/H723HEM 22198 SAGE_2025-12-12.jpg" 

# Target Mockup to track
TARGET_SUBSTRING = "MJ2219822198 KHAKI"

def calc_color_score(img1_cv, img2_path):
    try:
        img2_cv = cv2.imread(img2_path)
        if img2_cv is None: return 0.0
        
        # Query Center Crop (Simulation)
        h, w, _ = img1_cv.shape
        cy, cx = h // 2, w // 2
        ch, cw = h // 2, w // 2 
        img1_cv_crop = img1_cv[cy - ch//2 : cy + ch//2, cx - cw//2 : cx + cw//2]
        
        hsv1 = cv2.cvtColor(img1_cv_crop, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2HSV)
        
        histSize = [30, 32]
        ranges = [0, 180, 0, 256] 
        channels = [0, 1]
        
        hist1 = cv2.calcHist([hsv1], channels, None, histSize, ranges, accumulate=False)
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        hist2 = cv2.calcHist([hsv2], channels, None, histSize, ranges, accumulate=False)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0.0, score)
    except Exception as e:
        return 0.0

def main():
    print(f"Loading resources...")
    embedder = CNNEmbedder()
    index = faiss.read_index(INDEX_FILE)
    with open(PATHS_FILE, "rb") as f:
        image_paths = pickle.load(f)
        
    print(f"Query: {QUERY_PATH}")
    query_pil = Image.open(QUERY_PATH).convert("RGB")
    query_cv = cv2.imread(QUERY_PATH)
    
    # Embedding
    query_vector = embedder.encode(query_pil)
    query_vector = query_vector.reshape(1, -1)
    
    # Search ALL
    print("Searching entire index...")
    distances, indices = index.search(query_vector, index.ntotal)
    
    candidates = []
    
    print("Re-ranking...")
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue
        
        path = image_paths[idx]
        motif_score = float(distances[0][i])
        
        color_score = calc_color_score(query_cv, path)
        final_score = (motif_score * 0.5) + (color_score * 0.5)
        
        candidates.append({
            "filename": os.path.basename(path),
            "motif": motif_score,
            "color": color_score,
            "final": final_score
        })
        
    # Sort
    candidates.sort(key=lambda x: x['final'], reverse=True)
    
    print("\n" + "="*80)
    print(f"{'Rank':<5} | {'Filename':<40} | {'Motif':<8} | {'Color':<8} | {'Final':<8}")
    print("="*80)
    
    # Print Top 20
    for i in range(min(20, len(candidates))):
        c = candidates[i]
        print(f"{i+1:<5} | {c['filename'][:40]:<40} | {c['motif']:.4f}   | {c['color']:.4f}   | {c['final']:.4f}")
        
    print("\n" + "="*80)
    print(f"SEARCHING FOR TARGET: {TARGET_SUBSTRING}")
    print("="*80)
    
    found = False
    for i, c in enumerate(candidates):
        if TARGET_SUBSTRING in c['filename']:
            print(f"RANK {i+1:<5} | {c['filename'][:40]:<40} | {c['motif']:.4f}   | {c['color']:.4f}   | {c['final']:.4f}")
            found = True
            
    if not found:
        print("Target NOT FOUND in results.")

if __name__ == "__main__":
    main()
