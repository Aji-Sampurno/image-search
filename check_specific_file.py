import pickle
import os

DATA_DIR = "data"
PATHS_FILE = os.path.join(DATA_DIR, "batik_paths.pkl")
TARGET_FILENAME = "MJ221982.00 22198 KHAKI PERADA_2025-09-27.jpg"

if os.path.exists(PATHS_FILE):
    with open(PATHS_FILE, "rb") as f:
        paths = pickle.load(f)
        
    print(f"Total paths in index: {len(paths)}")
    
    found = False
    for i, p in enumerate(paths):
        if os.path.basename(p) == TARGET_FILENAME:
            print(f"✅ FOUND at index {i}: {p}")
            found = True
            break
            
    if not found:
        print(f"❌ NOT FOUND. {TARGET_FILENAME} is NOT in the index.")
else:
    print("Paths file not found.")
