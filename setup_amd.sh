#!/bin/bash
# ============================================================
# RUBIDIUM - Setup para AMD Developer Cloud
# Script de instalación y configuración
# ============================================================

echo "============================================================"
echo "RUBIDIUM - Configuración AMD Developer Cloud"
echo "============================================================"

# 1. Verificar GPU AMD
echo -e "\n--- Verificando GPU AMD ---"
if command -v rocminfo &> /dev/null; then
    rocminfo | head -20
else
    echo "rocminfo no encontrado. Instalando..."
    sudo apt-get update
    sudo apt-get install -y rocm-dev rocm-libs
fi

# 2. Instalar dependencias
echo -e "\n--- Instalando dependencias ---"
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-dev \
    libomp-dev

# 3. Instalar PyTorch con soporte ROCm
echo -e "\n--- Instalando PyTorch ROCm ---"
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# 4. Verificar instalación
echo -e "\n--- Verificando PyTorch ---"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No disponible\"}')"

# 5. Clonar repositorio
echo -e "\n--- Clonando rubidium-train ---"
if [ ! -d "rubidium-train" ]; then
    git clone https://github.com/diegovelandiabarajas1-lang/rubidium-train.git
fi
cd rubidium-train

# 6. Crear directorio de datos
echo -e"\n--- Preparando datos ---"
mkdir -p data
mkdir -p checkpoints
mkdir -p output

# 7. Entrenar tokenizer
echo -e "\n--- Entrenando tokenizer BPE + Unigram ---"
python3 src/tokenizer.py resources/ tokenizer.json

# 8. Tokenizar corpus
echo -e "\n--- Tokenizando corpus ---"
python3 -c "
from src.tokenizer import BPETokenizer, CorpusTokenizer
tokenizer = BPETokenizer.load('tokenizer.json')
corpus_tokenizer = CorpusTokenizer(tokenizer)
corpus_tokenizer.tokenize_directory('resources/', 'data/')
"

echo -e "\n============================================================"
echo "CONFIGURACIÓN COMPLETADA"
echo "============================================================"
echo "Para entrenar ejecuta:"
echo "  python3 src/train_amd.py"
echo ""
echo "Para verificar GPU:"
echo "  rocminfo"
echo "  python3 -c 'import torch; print(torch.cuda.is_available())'"
