from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import pickle
import os
import cv2
import io
from batik_embedder import BatikEmbedder

# Import utilities from main if needed, or re-implement
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def orb_verify(query_bytes, candidate_path):
    """
    Performs geometric verification using ORB feature matching.
    Returns a verification score (0.0 to 1.0).
    """
    try:
        # Decode Query
        nparr = np.frombuffer(query_bytes, np.uint8)
        img1 = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Load Candidate
        img2 = cv2.imread(candidate_path, cv2.IMREAD_GRAYSCALE)
        if img2 is None: return 0.0

        # CROP TO CENTER 50% (Remove Collar/Buttons/Sleeves)
        # This forces ORB to match the PATTERN, not the SHIRT.
        def crop_center(img):
            h, w = img.shape
            cy, cx = h // 2, w // 2
            # Crop 50%
            dy, dx = h // 4, w // 4
            return img[cy-dy:cy+dy, cx-dx:cx+dx]

        img1 = crop_center(img1)
        img2 = crop_center(img2)

        # ORB Detector
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None: return 0.0

        # Matcher with k-NN (k=2) for Ratio Test
        # This is CRITICAL for repetitive textures (like batik) to avoid false positives.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # CrossCheck OFF for kNN
        try:
            matches = bf.knnMatch(des1, des2, k=2)
        except Exception:
            return 0.0
        
        # Apply Lowe's Ratio Test
        # If the best match isn't significantly better than the second best, it's ambiguous (noise).
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        count = len(good_matches)
        
        # Scoring Logic with Stricter Requirements
        # We need a decent number of UNIQUE, GOOD matches.
        if count < 8: return 0.0 # Strict floor
        if count > 40: return 1.0
        return (count - 8) / 32.0
        
    except Exception as e:
        print(f"ORB Error: {e}")
        return 0.0

app = FastAPI(title="Batik Search Engine")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (external websites)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import urllib.parse

# Global variables
index_data = {}
embedder = None
INDEX_FILE = "batik_index.pkl"
FB_INDEX_FILE = "batik_index_fb.pkl"
BUCKET_NAME = "gooproper-aplikasi" # Hardcoded from user context

@app.on_event("startup")
async def load_resources():
    global index_data, embedder, INDEX_FILE
    
    # Initialize Embedder
    print("Initializing BatikEmbedder...")
    embedder = BatikEmbedder(use_cuda=False)
    
    # Prioritize Firebase Index if exists
    if os.path.exists(FB_INDEX_FILE):
        print(f"Loading Firebase index from {FB_INDEX_FILE}...")
        with open(FB_INDEX_FILE, 'rb') as f:
            index_data = pickle.load(f)
        print(f"Loaded {len(index_data)} items from Firebase.")
    elif os.path.exists(INDEX_FILE):
        print(f"Loading local index from {INDEX_FILE}...")
        with open(INDEX_FILE, 'rb') as f:
            index_data = pickle.load(f)
        print(f"Loaded {len(index_data)} items locally.")
    else:
        print(f"Warning: No index file found. Search will be empty.")

@app.post("/api/search")
async def search_image(file: UploadFile = File(...)):
    if not embedder:
        raise HTTPException(status_code=500, detail="Embedder not initialized")
    
    try:
        contents = await file.read()
        
        # Generate embedding from bytes
        res = embedder.generate_embedding(image_source=contents, from_bytes=True)
        
        if res["status"] == "error":
            raise HTTPException(status_code=400, detail=res["message"])
            
        q_vecs = res["vector"] # Now a dict: {'structure': ..., 'color': ...}
        
        # Compute Similarities with GATING
        results = []
        for filename, db_vecs in index_data.items():
            # 1. Structure Score (The Gatekeeper)
            # DINO vectors (normalized)
            try:
                # Handle legacy index (if flat array) vs new dict index
                if isinstance(db_vecs, np.ndarray):
                    # Legacy fallback (won't work well with Gating, but prevents crash)
                    # We skip or force re-index
                    continue 

                score_struct = np.dot(q_vecs["structure"], db_vecs["structure"])
                
                # GATE: Lower threshold to 0.45 to account for real-world photo variations
                if score_struct < 0.45:
                    final_score = 0.0
                    raw_debug = score_struct
                else:
                    # 2. Compute other scores
                    score_color = np.dot(q_vecs["color"], db_vecs["color"])
                    score_texture = np.dot(q_vecs["texture"], db_vecs["texture"])
                    
                    # Handle Frequency (Optional for backward compatibility if re-indexing fails)
                    if "frequency" in q_vecs and "frequency" in db_vecs:
                        score_freq = np.dot(q_vecs["frequency"], db_vecs["frequency"])
                    else:
                        score_freq = score_texture # Fallback
                    
                    # weighted combination
                    # MAXIMIZING TEXTURE/FREQ influence (50%)
                    # Minimizing Structure (20%)
                    raw_score = (0.2 * score_struct) + (0.3 * score_color) + (0.25 * score_texture) + (0.25 * score_freq)
                    
                    # Calibration
                    vector_score = max(0.0, min(1.0, (raw_score - 0.45) / 0.50))
                    
                    # STRICT COLOR PENALTY:
                    # Raised threshold to 0.85. 
                    # Black/Gold vs Orange MUST fail this.
                    if score_color < 0.82: # Slightly relaxed to 0.82 to be safe for lighting
                         vector_score *= 0.1 # Nuke it. 0% tolerance for wrong color.
                    
                    # 3. ORB Reranking (The Detail Checker)
                    orb_score = 0.0
                    local_path = os.path.join("static/images", filename)
                    # Run ORB on everything plausible
                    if vector_score > 0.1 and os.path.exists(local_path):
                        orb_score = orb_verify(contents, local_path)
                    
                    # ZERO TOLERANCE POLICY:
                    # The user requested "Langsung 0" (Straight to 0) if motifs don't match.
                    # We interpreted this as: If Geometric Verification fails, the score is 0.
                    
                    if orb_score > 0.1: 
                        # Valid geometric match -> Unlocks high scores
                        # Boost significantly
                        final_score = 0.8 + (orb_score * 0.2)
                        # Ensure it's at least as high as the vector score gave
                        final_score = max(final_score, vector_score)
                    else:
                        # No geometric confirmation -> REJECT COMPLETELY (0%)
                        # "Mirip Dikit" (Just similar) is not accepted.
                        final_score = 0.0
                    
                    final_score = final_score * 100
                    raw_debug = raw_score
                
            except Exception as e:
                print(f"Error comparing {filename}: {e}")
                continue

            # Determine URL type
            if filename.startswith("http"):
                url = filename
            elif os.path.exists(filename): 
                # Absolute local path (fallback)
                url = f"https://placehold.co/400x400/222/FFF?text={os.path.basename(filename)}"
            else:
                # Assume Firebase Blob Name
                # Construct Public URL
                safe_name = urllib.parse.quote(filename, safe='')
                url = f"https://firebasestorage.googleapis.com/v0/b/{BUCKET_NAME}/o/{safe_name}?alt=media"

            results.append({
                "filename": filename,
                "score": float(final_score),
                "raw_score": float(raw_debug), 
                "url": url
            })
            
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return Top 20
        return JSONResponse(content={"results": results[:20]})
        
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve Static Files (Frontend)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
# Optional: Mount a directory to serve local images if they are local files
# WARNING: This exposes the directory. For dev only.
# app.mount("/images", StaticFiles(directory="/path/to/local/images"), name="images")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
