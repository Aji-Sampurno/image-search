import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def quantize_model():
    model_fp32 = "dinov2.onnx"
    model_quant = "dinov2_quant.onnx"
    
    print(f"Quantizing {model_fp32} to {model_quant}...")
    
    quantize_dynamic(
        model_input=model_fp32,
        model_output=model_quant,
        weight_type=QuantType.QUInt8
    )
    
    size_mb = os.path.getsize(model_quant) / (1024 * 1024)
    print(f"Quantization complete! New size: {size_mb:.2f} MB")

if __name__ == "__main__":
    quantize_model()
