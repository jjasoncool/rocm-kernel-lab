ARG VERSION=6.4
FROM rocm/dev-ubuntu-24.04:${VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    vim \
    rocblas-dev \
    miopen-hip-dev \
    rocprofiler \
    roctracer-dev \
    hipblaslt-dev \
    && rm -rf /var/lib/apt/lists/*

# Python (用於 benchmark 畫圖和專案二)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --break-system-packages \
    matplotlib pandas

WORKDIR /workspace
