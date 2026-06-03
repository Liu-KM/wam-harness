FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG COSMOS_POLICY_REPO=https://github.com/NVlabs/cosmos-policy.git
ARG COSMOS_POLICY_REF=18a2acc

ENV DEBIAN_FRONTEND=noninteractive
ENV WAM_COSMOS_POLICY_REPO=/opt/cosmos-policy
ENV PATH=/opt/cosmos-policy/.venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libegl1-mesa-dev \
    libgl1 \
    libgl1-mesa-dri \
    libglib2.0-0 \
    tree \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.12 /uv /uvx /usr/local/bin/

WORKDIR /workspace/wam-harness

COPY pyproject.toml README.md uv.lock ./
COPY src ./src
COPY containers/cosmos-policy/native-smoke.sh /usr/local/bin/wam-cosmos-policy-native-smoke

RUN chmod +x /usr/local/bin/wam-cosmos-policy-native-smoke \
    && git clone "${COSMOS_POLICY_REPO}" /opt/cosmos-policy \
    && git -C /opt/cosmos-policy checkout "${COSMOS_POLICY_REF}" \
    && cd /opt/cosmos-policy \
    && uv sync --extra cu128 --group libero --python 3.10 \
    && uv pip install --python /opt/cosmos-policy/.venv/bin/python /workspace/wam-harness

WORKDIR /workspace/wam-harness

CMD ["wam", "--help"]
