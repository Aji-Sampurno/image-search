import os
# FORCE OpenMP COMPATIBILITY FOR MACOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import pickle
import time
# Replace Torch Embedder with ONNX Embedder
from onnx_embedder import ONNXEmbedder

# Configuration
# Use Absolute Path to be safe
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "static/images")
DATA_DIR = os.path.join(BASE_DIR, "data")

VECTORS_FILE = os.path.join(DATA_DIR, "batik_vectors.npy")
PATHS_FILE = os.path.join(DATA_DIR, "batik_paths.pkl")

def load_existing_index():
    if os.path.exists(VECTORS_FILE) and os.path.exists(PATHS_FILE):
        print(f"Loading existing index from {DATA_DIR}...")
        try:
            vectors = np.load(VECTORS_FILE)
            with open(PATHS_FILE, 'rb') as f:
                paths = pickle.load(f)
            
            if len(vectors) != len(paths):
                print("Warning: Size mismatch in existing index. Rebuilding from scratch recommended.")
                return None, None
            
            print(f"Loaded {len(paths)} existing items.")
            return vectors, paths
        except Exception as e:
            print(f"Error loading existing index: {e}")
            return None, None
    else:
        print("No existing index found. Starting fresh.")
        return None, None

def main():
    if not os.path.exists(IMAGES_DIR):
        print(f"Error: Images directory '{IMAGES_DIR}' not found.")
        return

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. Scan for Images
    print("Scanning images...")
    current_paths_set = set()
    current_paths_list = [] # For consistent ordering if needed, though mostly we use set for diff
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    for root, _, files in os.walk(IMAGES_DIR):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_exts:
                full_path = os.path.join(root, file)
                current_paths_set.add(full_path)
                current_paths_list.append(full_path)
    
    if not current_paths_set:
        print("No images found in images directory.")
        return

    print(f"Found {len(current_paths_set)} images on disk.")
    
    # 2. Load Existing Data & Calculate Diff
    existing_vectors, existing_paths = load_existing_index()
    
    vectors_to_keep = []
    paths_to_keep = []
    
    existing_paths_set = set()
    
    if existing_paths is not None:
        # Check for removed files
        kept_indices = []
        removed_count = 0
        
        for i, path in enumerate(existing_paths):
            if path in current_paths_set:
                kept_indices.append(i)
                existing_paths_set.add(path)
            else:
                removed_count += 1
        
        if kept_indices:
            vectors_to_keep = existing_vectors[kept_indices]
            paths_to_keep = [existing_paths[i] for i in kept_indices]
        
        if removed_count > 0:
            print(f"Removing {removed_count} images that are no longer on disk.")
    
    # Identify NEW files
    new_paths = []
    for path in current_paths_list:
        if path not in existing_paths_set:
            new_paths.append(path)
            
    if not new_paths:
        print("No new images to index and index is clean. Exiting.")
        # We might still want to save if we removed items, so let's check
        if existing_paths is not None and len(paths_to_keep) != len(existing_paths):
             print(f"Saving cleaned index ({len(paths_to_keep)} items)...")
             np.save(VECTORS_FILE, vectors_to_keep)
             with open(PATHS_FILE, 'wb') as f:
                 pickle.dump(paths_to_keep, f)
             print("Index updated (removals only).")
                                      
        return

    print(f"Found {len(new_paths)} NEW images to process.")
    
    # 3. Initialize Embedder
    print("Initializing ONNX Embedder...")
    embedder = ONNXEmbedder() 

    # 4. Generate Embeddings for NEW files only
    new_vectors = []
    valid_new_paths = []
    failed_paths = []
    
    start_time = time.time()
    for idx, path in enumerate(new_paths):
        try:
            vec = embedder.encode(path)
            new_vectors.append(vec)
            valid_new_paths.append(path)
            
            if (idx + 1) % 10 == 0: 
                print(f"[{idx+1}/{len(new_paths)}] Processed...")
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saving progress so far...")
            break
        except Exception as e:
            print(f"Failed to process {path}: {e}")
            failed_paths.append(path)
            
    total_time = time.time() - start_time
    print(f"Embedding finished in {total_time:.2f}s")
    
    if not new_vectors and not vectors_to_keep: # No existing data and no new data
        print("No valid vectors generated.")
        return

    # 5. Merge & Save
    final_vectors = []
    final_paths = paths_to_keep + valid_new_paths
    
    if len(new_vectors) > 0:
        new_vectors_np = np.stack(new_vectors).astype('float32')
        if len(vectors_to_keep) > 0:
            final_vectors = np.vstack((vectors_to_keep, new_vectors_np))
        else:
            final_vectors = new_vectors_np
    else:
        if len(vectors_to_keep) > 0:
            final_vectors = vectors_to_keep
        else:
            final_vectors = np.empty((0, 384), dtype='float32') # Should catch above, but safety first

    print(f"Final vectors shape: {final_vectors.shape}")
    
    print("Saving artifacts...")
    np.save(VECTORS_FILE, final_vectors)
    with open(PATHS_FILE, 'wb') as f:
        pickle.dump(final_paths, f)
        
    print(f"DONE. Index updated with {len(final_vectors)} images.")
    print(f"Files saved to: {VECTORS_FILE}")
    print("IMPORTANT: Don't forget to push these files to GitHub!")

if __name__ == "__main__":
    main()
