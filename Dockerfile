# =========================
# Build argument for base image selection
# =========================
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# =========================
# Stage 1: Base image with common dependencies
# =========================
FROM ${BASE_IMAGE} AS base

# Build arguments (defaults OK for standalone builds)
ARG COMFYUI_VERSION=latest
ARG CUDA_VERSION_FOR_COMFY
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# System deps
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    git \
    wget \
    aria2 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# uv + venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# comfy-cli for installing ComfyUI
RUN uv pip install comfy-cli pip setuptools wheel

# Install ComfyUI into /comfyui
RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia; \
    else \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia; \
    fi

# Optional: upgrade torch for specific CUDA versions
RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
      uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
    fi

# Working directory for ComfyUI
WORKDIR /comfyui

# Support for network volume path mapping if used by your start script
ADD src/extra_model_paths.yaml ./

# Back to root
WORKDIR /

# Runtime deps for the handler
RUN uv pip install runpod requests websocket-client

# App code and scripts
ADD src/start.sh handler.py test_input.json ./
RUN chmod +x /start.sh

# Custom node helper scripts (if you use them at runtime)
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

ENV PIP_NO_INPUT=1

COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# Default command
CMD ["/start.sh"]

# =========================
# Stage 2: Downloader (optional baseline models into /comfyui/models)
# =========================
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG MODEL_TYPE=flux1-dev-fp8

WORKDIR /comfyui
RUN mkdir -p models/checkpoints models/vae models/unet models/clip

# SDXL option
RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
      wget -q -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors && \
      wget -q -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
      wget -q -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors; \
    fi

# SD3 option
RUN if [ "$MODEL_TYPE" = "sd3" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
    fi

# FLUX schnell option
RUN if [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
      wget -q -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -q -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
    fi

# FLUX dev option
RUN if [ "$MODEL_TYPE" = "flux1-dev" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
      wget -q -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -q -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors; \
    fi

# FLUX dev fp8 minimal option
RUN if [ "$MODEL_TYPE" = "flux1-dev-fp8" ]; then \
      wget -q -O models/checkpoints/flux1-dev-fp8.safetensors https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors; \
    fi

# =========================
# Stage 3: Final image
# =========================
FROM base AS final

# Symlink so your pod-style paths resolve 1:1
RUN mkdir -p /workspace/runpod-slim && ln -s /comfyui /workspace/runpod-slim/ComfyUI

# Configure git and clone custom nodes in a single RUN to ensure non-interactive mode
# Note: ComfyUI-Manager is already installed in the base image, so we skip it here
RUN git config --global credential.helper "" && \
    git config --global http.postBuffer 524288000 && \
    mkdir -p /workspace/runpod-slim/ComfyUI/custom_nodes && \
    cd /workspace/runpod-slim/ComfyUI/custom_nodes && \
    git clone --depth=1 https://github.com/chrisgoringe/cg-use-everywhere.git && \
    git clone --depth=1 https://github.com/city96/ComfyUI-GGUF.git && \
    git clone --depth=1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && \
    git clone --depth=1 https://github.com/ltdrdata/comfyui-impact-subpack.git && \
    git clone --depth=1 https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git && \
    git clone --depth=1 https://github.com/gseth/ControlAltAI-Nodes.git

# Bake your exact model set into the same paths your workflow expects
RUN mkdir -p \
  /workspace/runpod-slim/ComfyUI/models/vae \
  /workspace/runpod-slim/ComfyUI/models/diffusion_models \
  /workspace/runpod-slim/ComfyUI/models/text_encoders \
  /workspace/runpod-slim/ComfyUI/models/unet \
  /workspace/runpod-slim/ComfyUI/models/upscale_models && \
  aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    -d /workspace/runpod-slim/ComfyUI/models/vae -o ae.safetensors \
    https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors && \
  aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    -d /workspace/runpod-slim/ComfyUI/models/diffusion_models -o flux1-dev-kontext_fp8_scaled.safetensors \
    https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors && \
  aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    -d /workspace/runpod-slim/ComfyUI/models/text_encoders -o clip_l.safetensors \
    https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
  aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    -d /workspace/runpod-slim/ComfyUI/models/text_encoders -o t5xxl_fp16.safetensors \
    https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors && \
  aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    -d /workspace/runpod-slim/ComfyUI/models/unet -o ultrarealFineTune_v4.gguf \
    "https://civitai.com/api/download/models/1413133?format=GGUF" && \
  aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    -d /workspace/runpod-slim/ComfyUI/models/upscale_models -o 4x-ClearRealityV1.pth \
    https://huggingface.co/skbhadra/ClearRealityV1/resolve/bc01e27b38eec683dc6e3161dd56069c78e015ac/4x-ClearRealityV1.pth

# Face detector model for Impact Pack detailer
RUN mkdir -p /workspace/runpod-slim/ComfyUI/models/bbox && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
      -d /workspace/runpod-slim/ComfyUI/models/bbox -o face_yolov8m.pt \
      https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/detection/bbox/face_yolov8m.pt

RUN mkdir -p /workspace/runpod-slim/ComfyUI/user/default/workflows
COPY lew.json /workspace/runpod-slim/ComfyUI/user/default/workflows/lew.json

# Reuse the same entrypoint from base
CMD ["/start.sh"]
