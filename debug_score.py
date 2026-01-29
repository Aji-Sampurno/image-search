import os
# Force OpenMP fix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from embedder import CNNEmbedder
from PIL import Image

# 1. Setup paths
# The image uploaded by the user
QUERY_PATH = "/Users/ajisampurno/.gemini/antigravity/brain/e8a00380-afa9-4f8f-9003-b1b3263e0fb5/uploaded_image_1768988755116.png"

# A file from the dataset that *should* match (Code 22198)
# I picked one from the previous 'find_by_name' output
TARGET_PATH = "static/images/Buring_H723_HEM 22198 KHAKI.jpg"

print(f"Query: {QUERY_PATH}")
print(f"Target: {TARGET_PATH}")

if not os.path.exists(QUERY_PATH):
    print("Error: Query image not found at path.")
    exit()

if not os.path.exists(TARGET_PATH):
    print("Error: Target image not found. Checking another variant...")
    # Try another one if that file doesn't exist
    TARGET_PATH = "static/images/qc/H723HEM 22198 SAGE_2025-12-12.jpg"
    if not os.path.exists(TARGET_PATH):
        print("Error: Target image not found in 'qc' folder either.")
        exit()

# 2. Initialize Embedder
print("Loading Embedder (ResNet50 + GeM)...")
embedder = CNNEmbedder()

# 3. Encode
print("Encoding Query...")
vec_q = embedder.encode(QUERY_PATH)

print(f"Encoding Target ({TARGET_PATH})...")
vec_t = embedder.encode(TARGET_PATH)

# 4. Calculate Similarity (Dot Product of L2 Normalized vectors = Cosine Sim)
score = np.dot(vec_q, vec_t)
print("-" * 30)
print(f"SIMILARITY SCORE: {score:.4f}")
print("-" * 30)

if score < 0.40:
    print("ANALYSIS: The score is VERY LOW (< 0.40).")
    print("Reason: The AI thinks these images are completely different.")
    print("Fix: We need better preprocessing (CLAHE) or Color Normalization.")
elif score < 0.60:
    print("ANALYSIS: The score is MEDIUM (0.40 - 0.60).")
    print("Reason: It sees some similarity, but the 'Strict Threshold' (0.40) might be barely passing or failing.")
else:
    print("ANALYSIS: The score is HIGH (> 0.60).")
    print("Reason: This SHOULD have been returned. The bug is in the FAISS index or Search Logic.")
