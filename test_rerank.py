import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import Torch FIRST to prevent OpenMP Segfault
import torch
import numpy as np
from embedder import CNNEmbedder # Uses torch

import cv2
import faiss
import pickle
from PIL import Image

# === CONFIG ===
INDEX_FILE = "data/batik.faiss"
PATHS_FILE = "data/batik_paths.pkl"
QUERY_PATH = "/Users/ajisampurno/.gemini/antigravity/brain/e8a00380-afa9-4f8f-9003-b1b3263e0fb5/uploaded_image_1768988755116.png"
TARGET_FILENAME = "MJ221983.00 22198 KHAKI PERADA_2025-09-27.jpg"

def get_sift_score(img1_path, img2_path):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Read images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return 0

    # Resize for speed if too large
    if img1.shape[0] > 1000:
        img1 = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)
    if img2.shape[0] > 1000:
        img2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5)

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except:
        return 0

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    return len(good_matches)

def main():
    print("1. Loading Index...")
    index = faiss.read_index(INDEX_FILE)
    with open(PATHS_FILE, "rb") as f:
        image_paths = pickle.load(f)
        
    embedder = CNNEmbedder()
    
    print("2. Embedding Query...")
    # Use the same embedding logic as app.py
    query_vector = embedder.encode(QUERY_PATH)
    query_vector = np.array([query_vector]).astype('float32')
    
    print("3. Retrieve Top 500 Candidates (FAISS)...")
    k = 500
    distances, indices = index.search(query_vector, k)
    
    candidates = []
    found_target_in_candidates = False
    
    print("4. Re-Ranking with SIFT...")
    
    ranked_results = []
    
    indices_list = indices[0]
    
    for rank, idx in enumerate(indices_list):
        if idx == -1: continue
        path = image_paths[idx]
        filename = os.path.basename(path)
        
        # Calculate SIFT Score
        sift_score = get_sift_score(QUERY_PATH, path)
        ranked_results.append({
            'path': path,
            'filename': filename,
            'faiss_rank': rank,
            'sift_score': sift_score
        })
        
        if filename == TARGET_FILENAME:
            print(f"   -> TARGET FOUND in Top 500 at FAISS Rank #{rank}. SIFT Score: {sift_score}")
            found_target_in_candidates = True

    if not found_target_in_candidates:
        print("   -> TARGET NOT FOUND in Top 500 FAISS results.")
        # Force calculate for target just to see
        target_full_path = "static/images/" + TARGET_FILENAME
        sift_score = get_sift_score(QUERY_PATH, target_full_path)
        print(f"   -> Manual SIFT check for target: {sift_score}")
        return

    # Sort by SIFT score descending
    ranked_results.sort(key=lambda x: x['sift_score'], reverse=True)
    
    print("\n=== TOP 10 AFTER RE-RANKING ===")
    for i, res in enumerate(ranked_results[:10]):
        prefix = ">>> " if res['filename'] == TARGET_FILENAME else "    "
        print(f"{prefix}Rank #{i+1} | Matches: {res['sift_score']:4d} | Was #{res['faiss_rank']} | {res['filename']}")

    # Find where target is now
    for i, res in enumerate(ranked_results):
        if res['filename'] == TARGET_FILENAME:
            print(f"\nTarget Final Rank: #{i+1}")
            break

if __name__ == "__main__":
    main()
