
# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Environment Variables
ENV HF_HOME=/app/model_cache
ENV PYTHONUNBUFFERED=1

# Install PyTorch CPU version specifically to reduce image size (Cloud Run runs on CPU)
# We do this before requirements.txt to ensure the CPU version is used
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements file
COPY requirements.txt .

# Install other dependencies
# (Pip will detect torch is already installed and skip it if version matches)
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the ResNet50 model to bake it into the image.
# This prevents downloading it on every Cloud Run cold start.
RUN python -c "from torchvision import models; \
    import torch; \
    from torch.hub import load_state_dict_from_url; \
    models.resnet50(weights=models.ResNet50_Weights.DEFAULT)"

# Copy the rest of the application code
# This will copy app.py, batik_embedder.py, main.py, pickle files, and static/ folder
COPY . .

# Expose port 8080 (Cloud Run expected port)
EXPOSE 8080

# Run the application using Uvicorn
# Use shell form to allow variable expansion for $PORT
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
