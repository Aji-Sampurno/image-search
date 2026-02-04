import torch
import torch.hub
import numpy as np

def export_model():
    print("Loading DINOv2 model via Torch Hub...")
    # Load the exact same model as embedder.py
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()

    # Create dummy input (Batch size 1, 3 channels, 518x518 resolution)
    dummy_input = torch.randn(1, 3, 518, 518)

    output_file = "dinov2.onnx"
    print(f"Exporting to {output_file}...")

    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print("Export complete!")

if __name__ == "__main__":
    export_model()
