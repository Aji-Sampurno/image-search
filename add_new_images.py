import os
# Force OpenMP fix for macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pickle
import faiss
import time
from embedder import CNNEmbedder
from PIL import Image

# === CONFIG ===
DATA_DIR = "data"
IMAGES_DIR = "static/images"
INDEX_FILE = os.path.join(DATA_DIR, "batik.faiss")
PATHS_FILE = os.path.join(DATA_DIR, "batik_paths.pkl")

def get_all_image_paths(directory):
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                full_path = os.path.join(root, file)
                # Keep paths relative logic consistent with build_index.py?
                # build_index.py usually stores them as strings.
                # Let's verify if they are absolute or relative.
                # Usually we want consistency.
                image_paths.append(full_path)
    return set(image_paths) # Return as set for fast lookup

def main():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(PATHS_FILE):
        print(f"Error: Index or Paths file not found in {DATA_DIR}.")
        print("Please run 'python3 build_index.py' at least once to create the initial index.")
        return

    print("Loading existing index and paths...")
    index = faiss.read_index(INDEX_FILE)
    
    with open(PATHS_FILE, "rb") as f:
        existing_paths_list = pickle.load(f)
    
    existing_paths_set = set(existing_paths_list)
    print(f"Current Index Size: {index.ntotal}")

    print("Scanning directory for new images...")
    current_files_set = get_all_image_paths(IMAGES_DIR)
    
    # Identify new files
    # Note: We need to ensure path normalization match. 
    # Usually absolute vs relative is the catch.
    # Assuming existing_paths_list contains whatever `os.path.join(root, file)` produced previously.
    
    new_files = []
    for fpath in current_files_set:
        if fpath not in existing_paths_set:
            new_files.append(fpath)
            
    if not new_files:
        print("No new images found. Database is up to date.")
        return

    print(f"Found {len(new_files)} new images.")
    print("Initializing Embedder (High-Res 800px)...")
    embedder = CNNEmbedder()
    
    new_vectors = []
    valid_new_paths = []
    
    t0 = time.time()
    
    for i, path in enumerate(new_files):
        print(f"[{i+1}/{len(new_files)}] Processing: {os.path.basename(path)}")
        try:
            vector = embedder.encode(path)
            new_vectors.append(vector)
            valid_new_paths.append(path)
        except Exception as e:
            print(f"  Error processing {path}: {e}")

    if not new_vectors:
        print("No valid vectors generated.")
        return

    # Convert to numpy
    new_vectors_np = np.stack(new_vectors).astype('float32')
    
    print(f"Adding {len(new_vectors_np)} vectors to index...")
    index.add(new_vectors_np)
    
    print("Updating paths list...")
    updated_paths = existing_paths_list + valid_new_paths
    
    print("Saving artifacts...")
    faiss.write_index(index, INDEX_FILE)
    
    with open(PATHS_FILE, 'wb') as f:
        pickle.dump(updated_paths, f)
        
    t_total = time.time() - t0
    print("-" * 30)
    print(f"SUCCESS. Added {len(new_vectors_np)} images in {t_total:.2f}s.")
    print(f"New Index Size: {index.ntotal}")
    print("Please restart 'app.py' to load the changes.")

if __name__ == "__main__":
    main()
