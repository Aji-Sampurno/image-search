import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import cv2
import numpy as np
from skimage import feature
import warnings
import io

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class BatikEmbedder:
    def __init__(self, use_cuda=False):
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        print(f"Loading BatikEmbedder on {self.device}...")
        
        # 1. Structural Model: DINOv2 (Small for speed/efficiency balance)
        # using facebook/dinov2-small from Hugging Face
        model_name = "facebook/dinov2-small"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Weights Tuned for "Signal-Based" Motif Matching (v3)
        # Reduced Structure (DINO) to 30% to stop "Similar Shirt Shape" matches.
        # Boosted Texture/Frequency to 40% combined to distinguish Grids vs Swirls.
        self.weights = {
            "structure": 0.30,
            "color": 0.30,
            "texture": 0.20,
            "frequency": 0.20
        }

    def _get_structure_embedding(self, image):
        """
        Uses DINOv2 to get texture-focused embedding (Patch Pooling).
        Instead of CLS token (object-level), we take the mean of patch tokens (texture-level).
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # outputs.last_hidden_state shape: [1, 257, 384] (1 CLS + 256 Patches)
            # Skip the first token (CLS) and average the rest (Patches)
            patch_tokens = outputs.last_hidden_state[:, 1:, :] 
            mean_patch = patch_tokens.mean(dim=1) # [1, 384]
            
        return F.normalize(mean_patch, p=2, dim=1).cpu().numpy().flatten()

    def _get_color_embedding(self, cv_image):
        """
        Extracts robust color features.
        Uses CLAHE to equalize lighting before histogram calculation.
        """
        # Convert to LAB
        lab_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_image)
        
        # Apply CLAHE to L-channel (fix lighting glare/shadows)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge back
        lab_image = cv2.merge((l, a, b))
        
        # 3D Histogram with fewer bins for 'fuzzy' matching
        # Bins: L(4), A(8), B(8) - Rougher lightness, nuanced color
        hist = cv2.calcHist([lab_image], [0, 1, 2], None, [4, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # 2. Color Statistics (Mean & Std Dev for Contrast)
        (l, a, b) = cv2.split(lab_image)
        stats = [
            np.mean(l), np.std(l), # Lightness/Contrast
            np.mean(a), np.std(a), # Green-Red
            np.mean(b), np.std(b)  # Blue-Yellow
        ]
        stats = np.array(stats)
        stats = stats / 255.0 # Simple normalization
        
        # Concatenate histogram and stats
        combined = np.concatenate([hist, stats])
        # L2 Normalize
        norm = np.linalg.norm(combined)
        return combined / (norm + 1e-7)

    def _get_texture_embedding(self, cv_image):
        """
        Extracts texture/line style using LBP (Local Binary Patterns) and Edges.
        Good for 'ornament density' and 'stroke style'.
        """
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 1. Local Binary Patterns (LBP) for micro-texture
        # Radius 3, Points 24 captures fine detail
        lbp = feature.local_binary_pattern(gray, P=24, R=3, method="uniform")
        (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-7)
        
        # 2. Edge Density & Sharpness (Canny)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / edges.size
        # Sharpness estimate (variance of Laplacian)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0 # scale down
        
        feat = np.concatenate([lbp_hist, [edge_density, sharpness]])
        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-7)

    def _get_frequency_embedding(self, cv_image):
        """
        Uses FFT to capture repetition rhythm and symmetry.
        Useful for geometric (Ceplok) vs organic patterns.
        """
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Fast Fourier Transform
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # Radial average (reduce 2D spectrum to 1D) profile
        h, w = magnitude_spectrum.shape
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Bin the radial distance to get a frequency profile (e.g., 32 bins)
        r_int = r.astype(int)
        tbin = np.bincount(r_int.ravel(), weights=magnitude_spectrum.ravel())
        nr = np.bincount(r_int.ravel())
        radial_profile = tbin / (nr + 1e-7)
        
        # Resize/Crop to fixed size (first 50 low-mid frequencies are most important for pattern)
        fixed_len = 50
        if len(radial_profile) > fixed_len:
            radial_profile = radial_profile[:fixed_len]
        else:
            radial_profile = np.pad(radial_profile, (0, fixed_len - len(radial_profile)))
            
        norm = np.linalg.norm(radial_profile)
        return radial_profile / (norm + 1e-7)

    def generate_embedding(self, image_source=None, from_bytes=False):
        """
        Master function to generate the fused fine-grained embedding.
        Args:
            image_source: File path (str) OR image bytes (if from_bytes=True)
            from_bytes: Boolean flag to indicate if source is bytes
        """
        try:
            # Load images
            if from_bytes:
                # Expecting image_source to be bytes
                image_bytes = image_source
                # PIL for Torch/DINO
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                # OpenCV for others
                nparr = np.frombuffer(image_bytes, np.uint8)
                cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # Expecting image_source to be path
                pil_image = Image.open(image_source).convert("RGB")
                cv_image = cv2.imread(image_source)

            # Check for valid load
            if cv_image is None or pil_image is None:
                raise ValueError("Failed to decode image data")
            
            # --- PREPROCESSING: CENTER CROP ---
            # Focus on the pattern, remove background/borders
            h, w, _ = cv_image.shape
            crop_frac = 0.75 # Keep central 75%
            start_y, start_x = int(h * (1 - crop_frac) / 2), int(w * (1 - crop_frac) / 2)
            end_y, end_x = int(h * (1 + crop_frac) / 2), int(w * (1 + crop_frac) / 2)
            
            cv_image = cv_image[start_y:end_y, start_x:end_x]
            pil_image = pil_image.crop((start_x, start_y, end_x, end_y))
            # ----------------------------------

            # Resize consistently for CV
            cv_image = cv2.resize(cv_image, (1024, 1024))
            
            # Extract Components
            emb_struct = self._get_structure_embedding(pil_image)
            emb_color = self._get_color_embedding(cv_image)
            emb_texture = self._get_texture_embedding(cv_image)
            emb_freq = self._get_frequency_embedding(cv_image)
            
            # Weighted Concatenation
            # We treat them as separate block vectors. 
            # In a real search functionality, you might want to compute distance per-section and weigh the distances.
            # Here, for a single vector output, we scale the vector components themselves.
            
            # Return components separately for Gated Scoring in app.py
            # We no longer fuse them here.
            
            return {
                "vector": {
                    "structure": emb_struct,
                    "color": emb_color,
                    "texture": emb_texture,
                    "frequency": emb_freq
                },
                "status": "success"
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Simple self-test
    embedder = BatikEmbedder()
    print("Embedder initialized successfully.")
