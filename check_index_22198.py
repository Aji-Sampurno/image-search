import pickle
import os

DATA_DIR = "data"
PATHS_FILE = os.path.join(DATA_DIR, "batik_paths.pkl")

if os.path.exists(PATHS_FILE):
    with open(PATHS_FILE, "rb") as f:
        paths = pickle.load(f)
        
    print(f"Total paths in index: {len(paths)}")
    
    target_code = "22198"
    found_count = 0
    print(f"\nChecking for files containing '{target_code}':")
    for p in paths:
        filename = os.path.basename(p)
        if target_code in filename and filename.startswith("MJ"):
            print(f"FOUND: {filename}")
            found_count += 1
            
    if found_count == 0:
        print("\nNo MJ files with code 22198 found in index.")
    else:
        print(f"\nFound {found_count} matching MJ files in index.")
else:
    print("Paths file not found.")
