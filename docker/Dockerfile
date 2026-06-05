FROM python:3.12-slim
ARG UID=1000
ARG GID=1000

# Torch flavour: "cpu" (default — light, ~2 GB image, works everywhere) or
# "cuda" (the default PyPI wheel with bundled NVIDIA CUDA libs, ~6 GB, needs
# the NVIDIA container runtime + `--gpus all` to actually use a GPU). The GPU
# build is opt-in via docker/compose.gpu.yaml / `make docker-*-gpu`.
ARG TORCH_VARIANT=cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PMB_HOME=/data \
    HF_HOME=/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/hf-cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g "${GID}" pmb \
    && useradd -m -u "${UID}" -g "${GID}" -s /bin/bash pmb \
    && mkdir -p /data /hf-cache /app \
    && chown -R pmb:pmb /data /hf-cache /app

WORKDIR /app

# Install dependencies first for better layer caching. The source is baked in
# so the image works on its own (`docker run pmb:local pmb ...`). Under compose we
# bind-mount the repo over /app, so the editable install then picks up live
# host edits instead of this baked copy.
COPY --chown=pmb:pmb pyproject.toml README.md ./
COPY --chown=pmb:pmb src ./src

# For the CPU build, pre-install the CPU-only torch from PyTorch's cpu index
# so the editable install below finds torch already satisfied and does NOT
# pull the multi-GB CUDA wheel. For the CUDA build, skip this and let
# `pip install -e .` resolve the default (GPU) wheel.
RUN pip install --upgrade pip \
    && if [ "$TORCH_VARIANT" = "cpu" ]; then \
         pip install --index-url https://download.pytorch.org/whl/cpu torch; \
       fi \
    && pip install -e ".[dev]"

USER pmb

CMD ["bash"]
