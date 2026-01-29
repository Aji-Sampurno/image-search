import os
# FORCE OpenMP COMPATIBILITY FOR MACOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from embedder import CNNEmbedder
import faiss
import cv2
import pickle
import numpy as np
from PIL import Image

# Setup
DATA_DIR = "data"
IMAGES_DIR = "static/images"
INDEX_FILE = os.path.join(DATA_DIR, "batik.faiss")
PATHS_FILE = os.path.join(DATA_DIR, "batik_paths.pkl")

# Targets
# QUERY_FILENAME = "Buring_H723_HEM 22198 KHAKI.jpg" 
# Use the uploaded user image (Absolute Path)
QUERY_PATH = "/Users/ajisampurno/.gemini/antigravity/brain/9de87fe4-700b-4274-8c00-29385441c59f/uploaded_media_1769396360411.png"
TARGET_FILENAME = "MJ221982.00 22198 KHAKI PERADA_2025-09-27.jpg" # The missing mockup

def calc_color_score(img1_cv, img2_path):
    try:
        img2_cv = cv2.imread(img2_path)
        if img2_cv is None: return 0.0
        
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
        print(f"Color error: {e}")
        return 0.0

def main():
    print(f"Loading resources...")
    embedder = CNNEmbedder()
    index = faiss.read_index(INDEX_FILE)
    with open(PATHS_FILE, "rb") as f:
        paths = pickle.load(f)
        
    query_path = QUERY_PATH
    target_path = os.path.join(IMAGES_DIR, TARGET_FILENAME)
    
    if not os.path.exists(query_path):
        print(f"Query image {query_path} not found!")
        return
        
    if not os.path.exists(target_path):
        print(f"Target image {target_path} not found!")
        return


    print(f"Target: {TARGET_FILENAME}")

    # 1. Embed Query
    query_img = Image.open(query_path).convert("RGB")
    query_vec = embedder.encode(query_img).reshape(1, -1)
    
    print(f"\n--- Simulating Full Search (Top 50) ---")
    candidates = []
    
    # 2. Search Index (Top 50 like app.py)
    k_retrieval = 50
    distances, indices = index.search(query_vec, k_retrieval)
    
    found_target_in_top_50 = False
    
    # Query CV2 for Color
    query_cv = cv2.imread(query_path)
    h, w, _ = query_cv.shape
    cy, cx = h // 2, w // 2
    ch, cw = h // 2, w // 2 
    query_crop = query_cv[cy - ch//2 : cy + ch//2, cx - cw//2 : cx + cw//2]

    
    weights = [(0.5, 0.5), (0.7, 0.3), (0.8, 0.2)]
    
    for w_motif, w_color in weights:
        print(f"\n========================================")
        print(f"Testing Weights: Motif {w_motif}, Color {w_color}")
        print(f"========================================")
        
        current_candidates = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            if idx >= len(paths): continue
            
            filepath = paths[idx]
            # Pre-calc'd in candidate map? No, just recalc, it's fast enough for 50 items
            
            # Need strict raw scores again
            motif_score = float(distances[0][i])
            
            # Recalculate color (efficieny: low, but ok for debug)
            color_score = calc_color_score(query_crop, filepath)
            
            final = (motif_score * w_motif) + (color_score * w_color)
            
            current_candidates.append({
                "filename": os.path.basename(filepath),
                "motif": motif_score,
                "color": color_score,
                "final": final,
                "is_target": (os.path.basename(filepath) == TARGET_FILENAME)
            })
            
        current_candidates.sort(key=lambda x: x['final'], reverse=True)
        
        print(f"Top 5 Results:")
        for i, c in enumerate(current_candidates[:5]):
            mark = "✅ " if c['is_target'] else "   "
            print(f"{i+1}. {mark}{c['filename']} | Final: {c['final']:.4f}")
            
        target_pos = -1
        for i, c in enumerate(current_candidates):
            if c['is_target']:
                target_pos = i+1
                print(f"   ...\n✅ Target at #{target_pos} | Final: {c['final']:.4f} (Motif: {c['motif']:.4f}, Color: {c['color']:.4f})")
                break


if __name__ == "__main__":
    main()
