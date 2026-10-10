# One image for every command of the pipeline.
#
#   docker build -t visual-synthesis:local .                                   # CPU
#   docker build -t visual-synthesis:local \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 .     # CUDA
#
# Data, weights, logs and results are mounted, never baked in:
#
#   docker run --rm -v $PWD/data:/app/data -v $PWD/models:/app/models \
#     -v $PWD/logs:/app/logs -v $PWD/results:/app/results \
#     visual-synthesis:local run all --from standardize --to postprocess
#
#   docker run --rm --gpus all ... visual-synthesis:local generate --limit 10
#   docker run --rm -p 8000:8000 ... visual-synthesis:local serve --host 0.0.0.0

FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/data/cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    VISUAL_SYNTHESIS_ROOT=/app

# libgl1 and libglib2.0-0 are needed by opencv (a NudeNet dependency).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch first, from the index that matches the target (CPU or a CUDA version).
RUN pip install --index-url "${TORCH_INDEX_URL}" torch

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install ".[all]"

COPY config/ ./config/
COPY tests/ ./tests/

EXPOSE 8000 7860

ENTRYPOINT ["visual-synthesis"]
CMD ["--help"]
