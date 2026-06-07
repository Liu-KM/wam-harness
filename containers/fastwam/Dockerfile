FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ARG LIBERO_REPO=https://github.com/Lifelong-Robot-Learning/LIBERO.git
ARG LIBERO_REF=master

ENV DEBIAN_FRONTEND=noninteractive
ENV WAM_FASTWAM_VENV=/opt/wam-fastwam-venv
ENV WAM_CACHE_DIR=/mnt/wam-cache
ENV WAM_TRACE_DIR=/mnt/runs
ENV WAM_LIBERO_DIR=/opt/LIBERO
ENV LIBERO_CONFIG_PATH=/mnt/wam-cache/libero/config
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
ENV TOKENIZERS_PARALLELISM=false
ENV WANDB_MODE=offline
ENV PATH="/opt/wam-fastwam-venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libosmesa6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir -U pip uv huggingface_hub

WORKDIR /workspace/wam-harness

COPY pyproject.toml README.md uv.lock ./
COPY src ./src
COPY containers/fastwam/native-smoke.sh /usr/local/bin/wam-fastwam-native-smoke
COPY scripts/setup_fastwam_native_env.sh ./scripts/setup_fastwam_native_env.sh
COPY scripts/fastwam-libero-eval.sh ./scripts/fastwam-libero-eval.sh

RUN chmod +x /usr/local/bin/wam-fastwam-native-smoke \
      ./scripts/setup_fastwam_native_env.sh \
      ./scripts/fastwam-libero-eval.sh \
    && ln -sf /workspace/wam-harness/scripts/fastwam-libero-eval.sh /usr/local/bin/wam-fastwam-libero-eval \
    && ./scripts/setup_fastwam_native_env.sh \
      --venv "${WAM_FASTWAM_VENV}" \
      --libero-dir /opt/LIBERO \
      --cache-dir /mnt/wam-cache \
      --harness-dir /workspace/wam-harness \
      --libero-ref "${LIBERO_REF}" \
      --clone

WORKDIR /workspace/wam-harness

CMD ["wam", "--help"]
