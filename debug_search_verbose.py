import os
# FORCE OpenMP COMPATIBILITY FOR MACOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import pickle
# Import Embedder (Torch) BEFORE FAISS to prevent OpenMP Segfault on Mac
from embedder import CNNEmbedder
import faiss
import cv2
from PIL import Image

# Configuration
DATA_DIR = "data"
VECTORS_FILE = os.path.join(DATA_DIR, "batik_vectors.npy")
PATHS_FILE = os.path.join(DATA_DIR, "batik_paths.pkl")
INDEX_FILE = os.path.join(DATA_DIR, "batik.faiss")

# Copied from app.py
def calc_color_score(img1_cv, img2_path, is_query=False):
    try:
        # Read target
        img2_cv = cv2.imread(img2_path)
        if img2_cv is None: return 0.0
        
        # Center Crop if it is the query
        if is_query:
            h, w, _ = img1_cv.shape
            cy, cx = h // 2, w // 2
            ch, cw = h // 2, w // 2 # 50% crop
            img1_cv = img1_cv[cy - ch//2 : cy + ch//2, cx - cw//2 : cx + cw//2]
        
        # Convert to HSV
        hsv1 = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2HSV)
        
        # Compute Histograms
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
    query_path = "/Users/ajisampurno/.gemini/antigravity/brain/f97d3ede-b1a7-40a2-8f0c-e05f4bd5dfd8/uploaded_image_1769224071894.png"
    
    print(f"Loading resources from {DATA_DIR}...")
    index = faiss.read_index(INDEX_FILE)
    with open(PATHS_FILE, "rb") as f:
        image_paths = pickle.load(f)
    
    print(f"Index Size: {index.ntotal}")
    print(f"Paths Size: {len(image_paths)}")
    
    embedder = CNNEmbedder()
    
    print(f"Processing Query: {query_path}")
    image = Image.open(query_path).convert("RGB")
    
    # Prepare CV2
    open_cv_query = np.array(image) 
    open_cv_query = open_cv_query[:, :, ::-1].copy()
    
    # Embed
    query_vector = embedder.encode(image).reshape(1, -1)
    
    # Search
    k = 20
    distances, indices = index.search(query_vector, k)
    
    print("\n--- Detailed Results ---")
    print(f"{'Filename':<40} | {'Motif':<8} | {'Color':<8} | {'Final':<8} | {'Status'}")
    print("-" * 100)
    
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue
        path = image_paths[idx]
        motif_score = float(distances[0][i])
        
        color_score = calc_color_score(open_cv_query, path, is_query=True)
        
        status = "✅ PASS"
        final_score = (motif_score * 0.5) + (color_score * 0.5)
        
        if color_score < 0.25:
            status = "❌ FILTERED (Color < 0.25)"
            final_score = 0.0
            
        print(f"{os.path.basename(path):<40} | {motif_score:.4f}   | {color_score:.4f}   | {final_score:.4f}   | {status}")

if __name__ == "__main__":
    main()
