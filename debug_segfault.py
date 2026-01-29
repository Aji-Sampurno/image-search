import os
# Try the fix immediately to see if it works
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("Importing torch...")
import torch
print("Importing faiss...")
import faiss
print("Importing embedding models...")
from torchvision import models

print("Initializing MobileNet...")
m = models.mobilenet_v3_large()
print("Success!")
