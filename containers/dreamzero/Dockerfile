FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ARG DREAMZERO_REPO=https://github.com/dreamzero0/dreamzero.git
ARG DREAMZERO_REF=ab790c1

ENV DEBIAN_FRONTEND=noninteractive
ENV WAM_DREAMZERO_REPO=/opt/dreamzero
ENV PATH=/opt/dreamzero/.venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libosmesa6 \
    libxrender1 \
    ninja-build \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.12 /uv /uvx /usr/local/bin/

WORKDIR /workspace/eazywam

COPY pyproject.toml README.md uv.lock ./
COPY src ./src
COPY containers/dreamzero/native-smoke.sh /usr/local/bin/wam-dreamzero-native-smoke

RUN chmod +x /usr/local/bin/wam-dreamzero-native-smoke \
    && git clone "${DREAMZERO_REPO}" /opt/dreamzero \
    && git -C /opt/dreamzero checkout "${DREAMZERO_REF}" \
    && cd /opt/dreamzero \
    && uv venv --python 3.11 .venv \
    && .venv/bin/python -m pip install --no-cache-dir -U pip setuptools wheel \
    && .venv/bin/python -m pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu129 \
      torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    && .venv/bin/python -m pip install --no-cache-dir -e . \
    && .venv/bin/python -m pip install --no-cache-dir flash-attn --no-build-isolation \
    && uv pip install --python /opt/dreamzero/.venv/bin/python /workspace/eazywam

WORKDIR /workspace/eazywam

CMD ["wam", "--help"]
