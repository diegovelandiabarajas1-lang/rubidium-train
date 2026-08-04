#!/bin/bash
# ============================================================
# RUBIDIUM CPU - Build Script (Linux/Kaggle)
# ============================================================
echo "============================================================"
echo "Building Rubidium CPU Training Engine"
echo "============================================================"

# Install build dependencies
echo "--- Installing dependencies ---"
apt-get update -qq 2>/dev/null
apt-get install -y -qq g++ make cmake libomp-dev 2>/dev/null || true

# Create build directory
mkdir -p build_cpu
cd build_cpu

# Configure with OpenMP
echo "--- Configuring ---"
cmake ../src \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -fopenmp -std=c++17" \
    -DCMAKE_EXE_LINKER_FLAGS="-lgomp"

# Build
echo "--- Building ---"
make -j$(nproc)

echo ""
echo "============================================================"
echo "Build complete!"
echo "============================================================"
echo ""
echo "Usage:"
echo "  ./rubidium_cpu_train train <corpus_dir>          - Train from scratch"
echo "  ./rubidium_cpu_train finetune <base> <corpus_dir> - LoRA fine-tune"
echo ""
echo "Examples:"
echo "  ./rubidium_cpu_train train ../data"
echo "  ./rubidium_cpu_train finetune model_final.bin ../data"
