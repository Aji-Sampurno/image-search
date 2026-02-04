import onnxruntime as ort
import numpy as np
from PIL import Image, ImageOps
import os

class ONNXEmbedder:
    def __init__(self, model_path="dinov2_quant.onnx"):
        print(f"Loading ONNX Model from {model_path}...")
        
        # Initialize ONNX Runtime
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # Preprocessing Constants
        # Mean & Std for ImageNet (Layout: RGB) -> Normalize
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        self.target_size = (518, 518)

    def preprocess(self, image_pil):
        """
        Resize -> CenterCrop -> Normalize -> NCHW -> Float32
        """
        # 1. Resize & Center Crop (Manual implementation of Torchvision)
        w, h = image_pil.size
        # Resize logic: Resize smallest edge to 518, maintain aspect ratio
        if w < h:
            new_w = 518
            new_h = int(h * (518 / w))
        else:
            new_h = 518
            new_w = int(w * (518 / h))
            
        img_resized = image_pil.resize((new_w, new_h), Image.BICUBIC)
        
        # Center Crop
        left = (new_w - 518) // 2
        top = (new_h - 518) // 2
        img_cropped = img_resized.crop((left, top, left + 518, top + 518))
        
        # 2. To Tensor (Normalize)
        # Convert to numpy (H, W, 3)
        img_np = np.array(img_cropped).astype(np.float32) / 255.0
        
        # (H, W, 3) -> (3, H, W)
        img_chw = img_np.transpose(2, 0, 1)
        
        return img_chw

    def encode(self, image_input):
        """
        Generates Multi-View Robust embedding using ONNX Runtime
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        # Create Views
        views = []
        
        # 1. Original
        views.append(image)
        
        # 2. Rotations
        views.append(image.rotate(90, expand=True))
        views.append(image.rotate(180, expand=True))
        views.append(image.rotate(270, expand=True))
        
        # 3. Mirror
        views.append(ImageOps.mirror(image))
        
        # 4. Zoom (Center Crop 75%)
        w, h = image.size
        c = min(w, h) * 0.75
        left = (w - c)/2
        top = (h - c)/2
        views.append(image.crop((left, top, left+c, top+c)))
        
        # 5. Macro Zoom (Center Crop 50%)
        c2 = min(w, h) * 0.50
        left2 = (w - c2)/2
        top2 = (h - c2)/2
        views.append(image.crop((left2, top2, left2+c2, top2+c2)))
        
        # Batch Process
        batch_np = []
        for v in views:
            batch_np.append(self.preprocess(v))
            
        # Stack -> (Batch, 3, 518, 518)
        batch_tensor = np.stack(batch_np)
        
        # Normalize
        # (Batch, 3, 518, 518) - (1,3,1,1) / (1,3,1,1)
        batch_tensor = (batch_tensor - self.mean) / self.std
        
        # Run Inference
        outputs = self.session.run(None, {self.input_name: batch_tensor})
        features = outputs[0] # (Batch, 384)
            
        # Average Pooling
        avg_feature = np.mean(features, axis=0) # (384,)
        
        # L2 Normalize
        norm = np.linalg.norm(avg_feature)
        if norm > 0:
            avg_feature = avg_feature / norm
            
        return avg_feature.astype(np.float32)
