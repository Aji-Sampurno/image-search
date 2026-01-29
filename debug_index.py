import pickle
import os

PATHS_FILE = "data/batik_paths.pkl"
TARGET_FILENAME = "22198"

print(f"Loading paths from {PATHS_FILE}...")
with open(PATHS_FILE, "rb") as f:
    paths = pickle.load(f)

print(f"Total paths: {len(paths)}")

found_count = 0
for i, p in enumerate(paths):
    if TARGET_FILENAME in p:
        print(f"Found at index {i}: {p}")
        found_count += 1
        
print("-" * 20)
print(f"Total entries matching '{TARGET_FILENAME}': {found_count}")

if found_count == 0:
    print("CRITICAL: The file is NOT in the index paths!")
else:
    print("OK: File(s) are in the index.")
