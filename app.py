import os
# FORCE OpenMP COMPATIBILITY FOR MACOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import numpy as np
import pickle
import os
import time
import io
from PIL import Image
# Import Embedder (Torch) BEFORE FAISS to prevent OpenMP Segfault on Mac
from embedder import CNNEmbedder
import faiss
import cv2 # Import OpenCV - Must be AFTER Torch/Embedder

# --- SIFT Helper ---
def get_sift_score(query_img_cv, target_path):
    try:
        # Read target as Grayscale directly to save memory
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if target_img is None: return 0
        
        # Convert Query (RGB/BGR) to Gray if needed
        if len(query_img_cv.shape) == 3:
            query_gray = cv2.cvtColor(query_img_cv, cv2.COLOR_BGR2GRAY)
        else:
            query_gray = query_img_cv
            
        # Resize for speed (limit max dim to 800)
        def resize_if_big(img):
            h, w = img.shape
            if max(h, w) > 800:
                scale = 800.0 / max(h, w)
                return cv2.resize(img, (0,0), fx=scale, fy=scale)
            return img
            
        query_gray = resize_if_big(query_gray)
        target_img = resize_if_big(target_img)

        # Detect SIFT
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(query_gray, None)
        kp2, des2 = sift.detectAndCompute(target_img, None)
        
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0
            
        # FLANN Matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Lowe's Ratio Test
        good_matches = 0
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches += 1
                
        return good_matches
    except Exception as e:
        print(f"SIFT Error on {target_path}: {e}")
        return 0
# -------------------

# --- Color Helper ---
def calc_color_score(img1_cv, img2_path, is_query=False):
    try:
        # Read target
        img2_cv = cv2.imread(img2_path)
        if img2_cv is None: return 0.0
        
        # Center Crop if it is the query (to focus on object, ignore background)
        if is_query:
            h, w, _ = img1_cv.shape
            cy, cx = h // 2, w // 2
            ch, cw = h // 2, w // 2 # 50% crop
            img1_cv = img1_cv[cy - ch//2 : cy + ch//2, cx - cw//2 : cx + cw//2]
        
        # Convert to HSV
        hsv1 = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2HSV)
        
        # Compute Histograms (Hue: 30 bins, Saturation: 32 bins)
        h_bins = 30
        s_bins = 32
        histSize = [h_bins, s_bins]
        ranges = [0, 180, 0, 256] 
        channels = [0, 1]
        
        hist1 = cv2.calcHist([hsv1], channels, None, histSize, ranges, accumulate=False)
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        hist2 = cv2.calcHist([hsv2], channels, None, histSize, ranges, accumulate=False)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0.0, score)
        
    except Exception as e:
        print(f"Color Hist Error: {e}")
        return 0.0

# -------------------

# Configuration
# Configuration
# Use Absolute Path based on this file's location to avoid CWD issues in Passenger
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

VECTORS_FILE = os.path.join(DATA_DIR, "batik_vectors.npy")
PATHS_FILE = os.path.join(DATA_DIR, "batik_paths.pkl")
INDEX_FILE = os.path.join(DATA_DIR, "batik.faiss")
IMAGES_DIR = os.path.join(BASE_DIR, "static/images")

app = FastAPI(title="Batik Search Engine (CNN + FAISS)")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Resources
embedder = None
index = None
image_paths = []
is_ready = False

@app.on_event("startup")
async def startup_event():
    global embedder, index, image_paths, is_ready
    print("Startup: Initializing resources...")
    
    try:
        # 1. Load Embedder
        embedder = CNNEmbedder()
        print("Embedder loaded.")
        
        # 2. Check for Index
        if os.path.exists(INDEX_FILE) and os.path.exists(PATHS_FILE):
            print(f"Loading index from {INDEX_FILE}...")
            index = faiss.read_index(INDEX_FILE)
            
            with open(PATHS_FILE, "rb") as f:
                image_paths = pickle.load(f)
                
            if index.ntotal != len(image_paths):
                print(f"Warning: Index size ({index.ntotal}) does not match paths count ({len(image_paths)}).")
            
            is_ready = True
            print(f"System READY. Index size: {index.ntotal}")
        else:
            print("Warning: Index not found. Please run 'python3 build_index.py' to generate.")
            is_ready = False
            
    except Exception as e:
        print(f"Critical Startup Error: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "ok" if is_ready else "not_ready_index_missing",
        "index_size": index.ntotal if index else 0
    }

@app.post("/api/search")
async def search_image(file: UploadFile = File(...)):
    global is_ready, index, image_paths
    
    if not is_ready or index is None:
        raise HTTPException(status_code=503, detail="Index not ready. Run build_index.py first.")
    
    try:
        start_time = time.time()
        
        # 1. Read Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Prepare CV2 Image for Color Scoring
        open_cv_query = np.array(image) 
        open_cv_query = open_cv_query[:, :, ::-1].copy() # RGB to BGR
        
        # 2. Generate Embedding
        t0 = time.time()
        query_vector = embedder.encode(image)
        # FAISS expects (1, D)
        query_vector = query_vector.reshape(1, -1)
        t_emb = time.time() - t0
        
        # 3. FAISS Search (Retrieval Stage)
        t1 = time.time()
        k_retrieval = index.ntotal # Compare against ALL images (Maximum Recall, slower at scale)
        distances, indices = index.search(query_vector, k_retrieval)
        t_faiss = time.time() - t1
        
        # 4. Hybrid Re-Ranking (Motif + Color)
        candidates = []
        
        print(f"Top 5 Raw FAISS Scores: {distances[0][:5]}")

        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            if idx >= len(image_paths): continue
            
            # FAISS Score (Cosine Similarity approx)
            motif_score = float(distances[0][i])
            
            target_path = image_paths[idx]
            
            # STRICT PENALTY REMOVED
            # We just use the weighted average. weak color matches will just have lower scores,
            # but they won't be eliminated entirely.
            
            # Color Score
            color_score = calc_color_score(open_cv_query, target_path, is_query=True)
            
            # Weighted Final Score
            # Weight: 50% Motif, 50% Color - Balanced Approach
            final_score = (motif_score * 0.5) + (color_score * 0.5)
            
            # Filter out 0 score results early
            if final_score > 0:
                candidates.append({
                    "path": target_path,
                    "motif_score": motif_score,
                    "color_score": color_score,
                    "final_score": final_score
                })
            
        # Sort by Final Score
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
            
        # 5. Format Results
        results = []
        for cand in candidates:
            # Double check (though we filtered above)
            if cand['final_score'] <= 0: continue
            
            path = cand['path']
            rel_path = os.path.relpath(path, IMAGES_DIR)
            url = f"/images/{rel_path}"
            
            results.append({
                "filename": os.path.basename(path),
                "score": cand['final_score'] * 100,
                "motif_score": cand['motif_score'] * 100, # Debug info
                "color_score": cand['color_score'] * 100, # Debug info
                "url": url,
                "matches": 0 
            })
            
        total_time = time.time() - start_time
        print(f"Search: {len(results)} results in {total_time:.3f}s (Emb: {t_emb:.3f}s, FAISS: {t_faiss:.3f}s)")
        
        return JSONResponse(content={"results": results})
        
    except Exception as e:
        print(f"Search Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# explicitly serve index.html for root to avoid 404s
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# Serve Static Files (Frontend)
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
