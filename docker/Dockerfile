FROM python:3.12-slim
ARG UID=1000
ARG GID=1000

# Torch flavour: "cpu" (default — light, ~2 GB image, works everywhere) or
# "cuda" (the default PyPI wheel with bundled NVIDIA CUDA libs, ~6 GB, needs
# the NVIDIA container runtime + `--gpus all` to actually use a GPU). The GPU
# build is opt-in via docker/compose.gpu.yaml / `make docker-*-gpu`.
ARG TORCH_VARIANT=cpu

# Extras to install alongside the core package, e.g. "dev" for pytest/ruff/
# hypothesis, "tui" for the textual dashboard. Defaults to none so a plain
# `docker build` produces a lean runtime image; local dev tooling opts in
# explicitly via `make docker-build` (compose.yaml passes PIP_EXTRAS=dev).
ARG PIP_EXTRAS=

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/pmb \
    PMB_HOME=/data \
    HF_HOME=/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/hf-cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g "${GID}" pmb \
    && useradd -m -u "${UID}" -g "${GID}" -s /bin/bash pmb \
    && mkdir -p /data /hf-cache /app \
    && chown -R pmb:pmb /data /hf-cache /app \
    # World-writable so the container works under any runtime UID (compose sets
    # `user:` to the host's id), incl. fresh named volumes that inherit these.
    && chmod -R 0777 /data /hf-cache /home/pmb

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
    && if [ -n "$PIP_EXTRAS" ]; then \
         pip install -e ".[$PIP_EXTRAS]"; \
       else \
         pip install -e "."; \
       fi

USER pmb

CMD ["bash"]
