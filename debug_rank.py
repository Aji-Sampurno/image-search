import os
# Force OpenMP fix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import os
# Force OpenMP fix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pickle
from embedder import CNNEmbedder # Torch first
from PIL import Image
import faiss # FAISS last

# 1. Setup paths
QUERY_PATH = "/Users/ajisampurno/.gemini/antigravity/brain/e8a00380-afa9-4f8f-9003-b1b3263e0fb5/uploaded_image_1768988755116.png"
INDEX_FILE = "data/batik.faiss"
PATHS_FILE = "data/batik_paths.pkl"
TARGET_FILENAME = "MJ221983.00 22198 KHAKI PERADA_2025-09-27.jpg"

print("Loading Index...")
index = faiss.read_index(INDEX_FILE)
print(f"Index loaded. Size: {index.ntotal}")

print("Loading Paths...")
with open(PATHS_FILE, "rb") as f:
    image_paths = pickle.load(f)

# 2. Encode Query
print("Encoding Query...")
embedder = CNNEmbedder()
query_vector = embedder.encode(QUERY_PATH).reshape(1, -1)

# 3. Search Full Dataset
print("Searching full dataset...")
k = index.ntotal # Search EVERYTHING
distances, indices = index.search(query_vector, k)

# 4. Find Rank of Target
print(f"Looking for images matching '{TARGET_FILENAME}'...")
found_ranks = []

for rank, idx in enumerate(indices[0]):
    path = image_paths[idx]
    filename = os.path.basename(path)
    if TARGET_FILENAME in filename:
        score = distances[0][rank]
        print(f"Rank #{rank+1}: {filename} (Score: {score:.4f})")
        found_ranks.append(rank + 1)
        if len(found_ranks) >= 5: # Just show top 5 matches
            break

if not found_ranks:
    print("Target image NOT found in search results?? (This shouldn't happen with k=total)")
else:
    print(f"\nSummary: Best match is at Rank #{found_ranks[0]}")
    if found_ranks[0] > 500:
        print("CONCLUSION: The image is ranked too low! Need better model or re-ranking.")
    else:
        print("CONCLUSION: It should be visible. Check filtering logic.")
