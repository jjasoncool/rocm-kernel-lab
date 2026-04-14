# hip-operator-lab

HIP kernel implementations of common AI operators on AMD GPUs (RX 9070 XT / ROCm).

## Quick start

```bash
# Build & start container
docker compose up -d --build
docker exec -it hip-operator-lab bash

# Inside container
cd /workspace/src
make
./bin/vector_add
```

## Profiling

```bash
rocprof --stats ./bin/vector_add
rocprof --hip-trace ./bin/vector_add
```

## Project structure

```
src/
  vector_add.cpp      # Day 1: vector addition
  # matrix_mul.cpp    # Day 2+: GEMM (naive → tiled)
  # relu.cpp          # activation operator
  # softmax.cpp       # softmax operator
  # layernorm.cpp     # layer normalization
  Makefile
benchmarks/           # profiling results & plots
```

## Hardware

- 2× AMD Radeon RX 9070 XT (RDNA 4, 16 GB)
- ROCm 6.4
