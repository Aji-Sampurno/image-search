import os
# FORCE OpenMP COMPATIBILITY FOR MACOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import pickle
import time
# Import Embedder (Torch) BEFORE FAISS to prevent OpenMP Segfault on Mac
from embedder import CNNEmbedder
import faiss

# Configuration
IMAGES_DIR = "static/images"
DATA_DIR = "data"
VECTORS_FILE = os.path.join(DATA_DIR, "batik_vectors.npy")
PATHS_FILE = os.path.join(DATA_DIR, "batik_paths.pkl")
INDEX_FILE = os.path.join(DATA_DIR, "batik.faiss")

def main():
    if not os.path.exists(IMAGES_DIR):
        print(f"Error: Images directory '{IMAGES_DIR}' not found.")
        return

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. Load Existing Data (if any)
    existing_paths = []
    existing_vectors = None
    existing_index = None

    if os.path.exists(PATHS_FILE) and os.path.exists(VECTORS_FILE) and os.path.exists(INDEX_FILE):
        print("Loading existing index for incremental update...")
        try:
            with open(PATHS_FILE, 'rb') as f:
                existing_paths = pickle.load(f)
            existing_vectors = np.load(VECTORS_FILE)
            existing_index = faiss.read_index(INDEX_FILE)
            print(f"Loaded {len(existing_paths)} existing images.")
        except Exception as e:
            print(f"Error loading existing index: {e}. Starting fresh.")
            existing_paths = []
            existing_vectors = None
            existing_index = None

    # 2. Scan for NEW Images
    print("Scanning images...")
    current_paths = []
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    for root, _, files in os.walk(IMAGES_DIR):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_exts:
                current_paths.append(os.path.join(root, file))
    
    if not current_paths:
        print("No images found in images directory.")
        return

    # Find difference
    existing_paths_set = set(existing_paths)
    new_paths = [p for p in current_paths if p not in existing_paths_set]
    
    # Also find deleted paths (optional cleanup, but for now let's just append new ones)
    # If we want to handle deletions, we'd need to rebuild the index or remove IDs. 
    # For now, let's focus on ADDITION (Incremental).
    
    if not new_paths:
        print("No new images found. Index is up to date.")
        return

    print(f"Found {len(new_paths)} NEW images to index.")
    
    # 3. Initialize Embedder only if needed
    print("Initializing Embedder...")
    embedder = CNNEmbedder()

    # 4. Generate Embeddings for NEW images
    new_vectors = []
    failed_paths = []
    
    start_time = time.time()
    for idx, path in enumerate(new_paths):
        try:
            vec = embedder.encode(path)
            new_vectors.append(vec)
            if idx % 10 == 0: # More frequent updates for smaller batches
                print(f"[{idx+1}/{len(new_paths)}] Processed...")
        except Exception as e:
            print(f"Failed to process {path}: {e}")
            failed_paths.append(path)
            
    print(f"New embeddings finished in {time.time() - start_time:.2f}s")
    
    if not new_vectors:
        print("No valid new vectors generated.")
        return

    # Convert to matrix
    new_vectors_np = np.stack(new_vectors).astype('float32') # (N, D)
    print(f"New vectors shape: {new_vectors_np.shape}")
    
    # Filter valid new paths
    valid_new_paths = [p for p in new_paths if p not in failed_paths]

    # 5. Merge with Existing
    if existing_vectors is not None:
        final_vectors = np.vstack((existing_vectors, new_vectors_np))
        final_paths = existing_paths + valid_new_paths
        
        # Update Index
        # Note: IndexFlatIP doesn't support easy deletion, but addition is fine.
        # However, to be safe and clean, we often just create a new index from the combined vectors 
        # because adding to FlatIP is cheap, but we already have the object.
        # Let's just .add() to the existing index object.
        existing_index.add(new_vectors_np)
        final_index = existing_index
    else:
        final_vectors = new_vectors_np
        final_paths = valid_new_paths
        
        d = final_vectors.shape[1]
        final_index = faiss.IndexFlatIP(d)
        final_index.add(final_vectors)

    print(f"Total Index Size: {final_index.ntotal}")

    # 6. Save Updated Artifacts
    print("Saving updated artifacts...")
    np.save(VECTORS_FILE, final_vectors)
    with open(PATHS_FILE, 'wb') as f:
        pickle.dump(final_paths, f)
    faiss.write_index(final_index, INDEX_FILE)
    
    print("DONE. Index updated.")

if __name__ == "__main__":
    main()
