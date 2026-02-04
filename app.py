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
# Remove faiss & cv2
# import cv2 

# --- Color Helper (PIL + Numpy implementation) ---
def calc_color_score_pil(img1_pil, img2_path, is_query=False):
    try:
        # Read target
        if not os.path.exists(img2_path): return 0.0
        
        try:
            img2_pil = Image.open(img2_path).convert("HSV")
        except:
            return 0.0

        # Center Crop if it is the query
        if is_query:
            w, h = img1_pil.size
            cx, cy = w // 2, h // 2
            cw, ch = w // 2, h // 2 # 50% crop
            img1_pil = img1_pil.crop((cx - cw//2, cy - ch//2, cx + cw//2, cy + ch//2))
        
        # Ensure query is HSV
        if img1_pil.mode != 'HSV':
            img1_pil = img1_pil.convert("HSV")
            
        # Compute Histograms
        # PIL histogram is a concatenation of histograms for each channel
        # HSV -> 3 channels * 256 bins = 768 params
        hist1 = np.array(img1_pil.histogram())
        hist2 = np.array(img2_pil.histogram())
        
        # We only care about Hue (0-255) and Saturation (256-511)
        # Value (512-767) is less important for color matching
        hist1 = hist1[:512]
        hist2 = hist2[:512]
        
        # Normalize
        norm1 = np.linalg.norm(hist1)
        norm2 = np.linalg.norm(hist2)
        
        if norm1 > 0: hist1 = hist1 / norm1
        if norm2 > 0: hist2 = hist2 / norm2
        
        # Correlation
        # Simple dot product on normalized histograms usually approximates correlation well enough for rankings
        # Or standard correlation coefficient:
        
        m1 = np.mean(hist1)
        m2 = np.mean(hist2)
        
        num = np.sum((hist1 - m1) * (hist2 - m2))
        den = np.sqrt(np.sum((hist1 - m1)**2)) * np.sqrt(np.sum((hist2 - m2)**2))
        
        if den == 0: return 0.0
        
        score = num / den
        return max(0.0, float(score))
        
    except Exception as e:
        print(f"Color Hist Error: {e}")
        return 0.0

# -------------------

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

VECTORS_FILE = os.path.join(DATA_DIR, "batik_vectors.npy")
PATHS_FILE = os.path.join(DATA_DIR, "batik_paths.pkl")
IMAGES_DIR = os.path.join(BASE_DIR, "static/images")

app = FastAPI(title="Batik Search Engine (ONNX + NumPy + PIL)")

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
        
        # 2. Generate Embedding (ONNX)
        t0 = time.time()
        query_vector = embedder.encode(image)
        t_emb = time.time() - t0
        
        # 3. NumPy Search
        t1 = time.time()
        scores = np.dot(image_vectors, query_vector)
        
        top_k = min(100, len(scores))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        t_search = time.time() - t1
        
        # 4. Hybrid Re-Ranking (Motif + Color)
        candidates = []
        
        print(f"Top 5 Raw Scores: {scores[top_indices][:5]}")

        for i, idx in enumerate(top_indices):
            if idx >= len(image_paths): continue
            
            motif_score = float(scores[idx])
            target_path = image_paths[idx]
            
            # Color Score (PIL)
            color_score = calc_color_score_pil(image, target_path, is_query=True)
            
            # Weighted Final Score
            final_score = (motif_score * 0.5) + (color_score * 0.5)
            
            if final_score > 0:
                candidates.append({
                    "path": target_path,
                    "motif_score": motif_score,
                    "color_score": color_score,
                    "final_score": final_score
                })
            
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

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
