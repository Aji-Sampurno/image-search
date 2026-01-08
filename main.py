import argparse
import numpy as np
import os
import sys
import pickle
import time
from batik_embedder import BatikEmbedder

# Optional imports for Firebase
try:
    import firebase_admin
    from firebase_admin import credentials, storage
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def scan_images(directory):
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_exts:
                image_paths.append(os.path.join(root, file))
    return image_paths

def cmd_index(args):
    print(f"Index Mode: Scanning {args.dataset_dir}...")
    images = scan_images(args.dataset_dir)
    print(f"Found {len(images)} images.")
    
    if len(images) == 0:
        print("No images found. Exiting.")
        return

    print(f"Initializing Embedder on {args.device}...")
    embedder = BatikEmbedder(use_cuda=(args.device != "cpu"))
    
    index_data = {}
    
    start_time = time.time()
    for idx, img_path in enumerate(images):
        print(f"[{idx+1}/{len(images)}] Processing {os.path.basename(img_path)}...", end="\r")
        res = embedder.generate_embedding(img_path)
        if res["status"] == "success":
            # Store just the vector to save space
            # In a real system, you might store metadata too
            index_data[img_path] = res["vector"]
        else:
            print(f"\nFailed: {img_path} - {res['message']}")
            
    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.2f}s.")
    
    with open(args.output_index, 'wb') as f:
        pickle.dump(index_data, f)
    print(f"Index saved to {args.output_index} ({len(index_data)} items).")

def cmd_index_firebase(args):
    if not FIREBASE_AVAILABLE:
        print("Error: firebase-admin module not installed. Run 'pip install firebase-admin'.")
        return

    if not os.path.exists(args.cred_file):
        print(f"Error: Credential file '{args.cred_file}' not found.")
        return

    print(f"Initializing Firebase with {args.cred_file}...")
    cred = credentials.Certificate(args.cred_file)
    try:
        firebase_admin.initialize_app(cred, {
            'storageBucket': args.bucket_name
        })
    except ValueError:
        # App might already be initialized
        pass

    bucket = storage.bucket()
    print(f"Connected to bucket: {args.bucket_name}")
    print("Listing blobs (this might take a moment)...")
    blobs = list(bucket.list_blobs())
    print(f"Found {len(blobs)} objects.")

    print(f"Initializing Embedder on {args.device}...")
    embedder = BatikEmbedder(use_cuda=(args.device != "cpu"))
    
    index_data = {}
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    start_time = time.time()
    for idx, blob in enumerate(blobs):
        # Filter extensions
        ext = os.path.splitext(blob.name)[1].lower()
        if ext not in valid_exts:
            continue

        print(f"[{idx+1}/{len(blobs)}] Downloading {blob.name}...", end="\r")
        
        try:
            # Download as bytes
            image_bytes = blob.download_as_bytes()
            
            # Save locally for ORB verification in app.py
            local_filename = os.path.join("static/images", blob.name)
            # Ensure subdir exists if blob has folders
            os.makedirs(os.path.dirname(local_filename), exist_ok=True)
            with open(local_filename, "wb") as f_img:
                f_img.write(image_bytes)
            
            # Generate embedding from bytes
            res = embedder.generate_embedding(image_source=image_bytes, from_bytes=True)
            
            if res["status"] == "success":
                # Store public URL or gs:// path as the key
                # Using generation to cache bust if needed, or just media link
                # For simplicity, store the blob name or a constructable URL
                # We store the blob NAME as key, app.py will look in static/images
                index_data[blob.name] = res["vector"]
            else:
                print(f"\nFailed to process {blob.name}: {res['message']}")
                
        except Exception as e:
            print(f"\nError downloading/processing {blob.name}: {e}")

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.2f}s.")
    
    with open(args.output_index, 'wb') as f:
        pickle.dump(index_data, f)
    print(f"Index saved to {args.output_index} ({len(index_data)} items).")

def cmd_search(args):
    if not os.path.exists(args.index_file):
        print(f"Error: Index file '{args.index_file}' not found. Run 'index' command first.")
        return

    print(f"Loading Index from {args.index_file}...")
    with open(args.index_file, 'rb') as f:
        index_data = pickle.load(f)
    print(f"Loaded {len(index_data)} embeddings.")

    print(f"Generating embedding for query: {args.query_image}...")
    embedder = BatikEmbedder(use_cuda=(args.device != "cpu"))
    query_res = embedder.generate_embedding(args.query_image)
    
    if query_res["status"] == "error":
        print(f"Error processing query image: {query_res['message']}")
        return
        
    query_vec = query_res["vector"]
    
    # Compute similarities
    results = []
    for path, vec in index_data.items():
        score = cosine_similarity(query_vec, vec)
        results.append((path, score))
        
    # Sort descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n--- Top Search Results ---")
    top_k = min(args.top_k, len(results))
    for i in range(top_k):
        path, score = results[i]
        filename = os.path.basename(path)
        print(f"{i+1}. {filename} (Score: {score:.4f})")
        # In a real app we would output JSON or similar

def cmd_compare(args):
    # Backward compatibility with previous version
    embedder = BatikEmbedder(use_cuda=(args.device != "cpu"))
    embeddings = []
    
    for img_path in [args.image1, args.image2]:
        print(f"Processing {os.path.basename(img_path)}...")
        res = embedder.generate_embedding(img_path)
        if res["status"] == "success":
            embeddings.append(res["vector"])
        else:
            print(f"Error: {res['message']}")
            return

    score = cosine_similarity(embeddings[0], embeddings[1])
    print(f"\nSimilarity: {score:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Batik Search Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Index Command
    parser_index = subparsers.add_parser("index", help="Index a directory of images")
    parser_index.add_argument("dataset_dir", help="Path to directory containing batik images")
    parser_index.add_argument("--output_index", default="batik_index.pkl", help="Output path for the index file")
    parser_index.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    
    # Search Command
    parser_search = subparsers.add_parser("search", help="Search for similar images")
    parser_search.add_argument("query_image", help="Path to query image")
    parser_search.add_argument("--index_file", default="batik_index.pkl", help="Path to the index file")
    parser_search.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    parser_search.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    
    # Compare Command (Legacy)
    parser_compare = subparsers.add_parser("compare", help="Compare two specific images")
    parser_compare.add_argument("image1", help="Path to first image")
    parser_compare.add_argument("image2", help="Path to second image")
    parser_compare.add_argument("--device", default="cpu", help="Device (cpu/cuda)")

    # Firebase Index Command
    parser_fb = subparsers.add_parser("index-firebase", help="Index images from Firebase Storage")
    parser_fb.add_argument("bucket_name", help="Firebase Storage Bucket Name (e.g. 'my-app.appspot.com')")
    parser_fb.add_argument("cred_file", help="Path to Firebase Service Account JSON")
    parser_fb.add_argument("--output_index", default="batik_index_fb.pkl", help="Output path for the index file")
    parser_fb.add_argument("--device", default="cpu", help="Device (cpu/cuda)")

    args = parser.parse_args()
    
    if args.command == "index":
        cmd_index(args)
    elif args.command == "index-firebase":
        cmd_index_firebase(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
