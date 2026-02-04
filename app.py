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
# Replace Torch Embedder with ONNX Embedder
from onnx_embedder import ONNXEmbedder
# Remove faiss import
import cv2 

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
# Use Absolute Path based on this file's location to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

VECTORS_FILE = os.path.join(DATA_DIR, "batik_vectors.npy")
PATHS_FILE = os.path.join(DATA_DIR, "batik_paths.pkl")
# Removed CONFIG/INDEX_FILE since we use pure numpy now
IMAGES_DIR = os.path.join(BASE_DIR, "static/images")

app = FastAPI(title="Batik Search Engine (ONNX + NumPy)")

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
image_vectors = None # Numpy Array
image_paths = []
is_ready = False

@app.on_event("startup")
async def startup_event():
    global embedder, image_vectors, image_paths, is_ready
    print("Startup: Initializing resources...")
    
    try:
        # 1. Load ONNX Embedder
        embedder = ONNXEmbedder()
        print("ONNX Embedder loaded.")
        
        # 2. Check for Vectors & Paths
        if os.path.exists(VECTORS_FILE) and os.path.exists(PATHS_FILE):
            print(f"Loading vectors from {VECTORS_FILE}...")
            image_vectors = np.load(VECTORS_FILE)
            
            with open(PATHS_FILE, "rb") as f:
                image_paths = pickle.load(f)
                
            if len(image_vectors) != len(image_paths):
                print(f"Warning: Vector count ({len(image_vectors)}) does not match paths count ({len(image_paths)}).")
            
            is_ready = True
            print(f"System READY. Database size: {len(image_vectors)}")
        else:
            print("Warning: Vectors/Paths not found. Please ensure data is indexed.")
            is_ready = False
            
    except Exception as e:
        print(f"Critical Startup Error: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "ok" if is_ready else "not_ready_data_missing",
        "index_size": len(image_vectors) if image_vectors is not None else 0
    }

@app.post("/api/search")
async def search_image(file: UploadFile = File(...)):
    global is_ready, image_vectors, image_paths
    
    if not is_ready or image_vectors is None:
        raise HTTPException(status_code=503, detail="Index not ready.")
    
    try:
        start_time = time.time()
        
        # 1. Read Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Prepare CV2 Image for Color Scoring
        open_cv_query = np.array(image) 
        open_cv_query = open_cv_query[:, :, ::-1].copy() # RGB to BGR
        
        # 2. Generate Embedding (ONNX)
        t0 = time.time()
        query_vector = embedder.encode(image)
        # query_vector is (384,)
        t_emb = time.time() - t0
        
        # 3. NumPy Search (Dot Product = Cosine Sim if Normalized)
        t1 = time.time()
        
        # Dot product
        # (N, D) @ (D,) -> (N,)
        scores = np.dot(image_vectors, query_vector)
        
        # Get Top K candidates (e.g., top 100 for re-ranking)
        # We want DESCENDING order
        top_k = min(100, len(scores))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        # Sort the top indices by score descending
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        t_search = time.time() - t1
        
        # 4. Hybrid Re-Ranking (Motif + Color)
        candidates = []
        
        print(f"Top 5 Raw Scores: {scores[top_indices][:5]}")

        for i, idx in enumerate(top_indices):
            if idx >= len(image_paths): continue
            
            motif_score = float(scores[idx])
            target_path = image_paths[idx]
            
            # Color Score
            color_score = calc_color_score(open_cv_query, target_path, is_query=True)
            
            # Weighted Final Score
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
            if cand['final_score'] <= 0: continue
            
            path = cand['path']
            rel_path = os.path.relpath(path, IMAGES_DIR)
            url = f"/images/{rel_path}"
            
            results.append({
                "filename": os.path.basename(path),
                "score": cand['final_score'] * 100,
                "motif_score": cand['motif_score'] * 100, 
                "color_score": cand['color_score'] * 100, 
                "url": url,
                "matches": 0 
            })
            
        total_time = time.time() - start_time
        print(f"Search: {len(results)} results in {total_time:.3f}s (Emb: {t_emb:.3f}s, Search: {t_search:.3f}s)")
        
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
