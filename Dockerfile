# ==============================================================================
# Dockerfile: Reproducible QLoRA Fine-Tuning Environment with CUDA Support
# ==============================================================================

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Metadata
LABEL maintainer="AI Engineering Team"
LABEL description="Reproducible container for LLM Fine-Tuning Pipeline with QLoRA & SFTTrainer"

# Set environment flags
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements.txt pyproject.toml /workspace/

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source repository
COPY . /workspace/

# Expose TensorBoard default port
EXPOSE 6006

# Default command
CMD ["bash"]
