# RUBIDIUM - Guía AMD Developer Cloud

## Configuración Completa para Entrenamiento

### 1. Crear Cuenta en AMD Developer Cloud

1. Ve a https://developer.amd.com/
2. Crea una cuenta gratuita
3. Solicita acceso al GPU Cloud
4. Espera aprobación (generalmente 1-2 días)

### 2. Conectar al Instance

```bash
# SSH a tu instancia
ssh -i tu-key.pem ubuntu@tu-ip

# Verificar GPU
rocminfo
```

### 3. Instalar Dependencias

```bash
# Ejecutar script de setup
chmod +x setup_amd.sh
./setup_amd.sh
```

### 4. Preparar Datos

```bash
# Entrenar tokenizer
python3 src/tokenizer.py resources/ tokenizer.json

# Tokenizar corpus
python3 -c "
from src.tokenizer import BPETokenizer, CorpusTokenizer
tokenizer = BPETokenizer.load('tokenizer.json')
corpus_tokenizer = CorpusTokenizer(tokenizer)
corpus_tokenizer.tokenize_directory('resources/', 'data/')
"
```

### 5. Entrenar Modelo

```bash
# Entrenamiento completo
python3 src/train_amd.py

# O con screen/tmux para sesiones largas
screen -S train
python3 src/train_amd.py
# Ctrl+A, D para desconectar
screen -r train para reconectar
```

### 6. Monitorear Progreso

```bash
# Ver logs
tail -f output/train.log

# Ver GPU usage
watch -n 1 rocm-smi
```

---

## Especificaciones del Modelo

| Parámetro | Valor |
|-----------|-------|
| Vocabulario | 32,000 tokens (BPE + Unigram) |
| Dimensión | 2,048 |
| Heads | 32 |
| Capas | 10 |
| FFN | 8,192 |
| Contexto | 512 tokens |
| Parámetros | ~505M |
| VRAM requerida | ~8.5 GB |

---

## Estructura de Archivos

```
rubidium-train/
├── src/
│   ├── model.h          # Header del modelo
│   ├── model.cu         # Implementación CUDA
│   ├── cuda_kernels.cu  # Kernels CUDA
│   ├── tokenizer.h      # Tokenizer C++
│   ├── tokenizer.py     # Tokenizer Python
│   ├── train_amd.py     # Script de entrenamiento
│   └── main.cpp         # Entry point CUDA
├── resources/           # Corpus原始
├── data/               # Corpus tokenizado
├── checkpoints/        # Checkpoints
├── tokenizer.json      # Tokenizer entrenado
├── setup_amd.sh        # Script de setup
└── CMakeLists.txt      # Build CUDA
```

---

## Comandos Útiles

```bash
# Verificar espacio en disco
df -h

# Verificar memoria
free -h

# Matar proceso de entrenamiento
pkill -f train_amd.py

# Continuar desde checkpoint
python3 src/train_amd.py --resume checkpoints/model_step_50000.pt

# Generar texto
python3 src/generate.py --checkpoint model_final.pt --prompt "Hola"
```

---

## Solución de Problemas

### Error: GPU no disponible
```bash
# Verificar ROCm
rocminfo
rocm-smi

# Reinstalar PyTorch ROCm
pip3 uninstall torch
pip3 install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

### Error: Memoria insuficiente
```bash
# Reducir batch size en train_amd.py
# Cambiar BS de 2 a 1
```

### Error: Tokenizer no encontrado
```bash
# Re-entrenar tokenizer
python3 src/tokenizer.py resources/ tokenizer.json
```

---

## Recursos

- AMD Developer Cloud: https://developer.amd.com/
- ROCm Documentation: https://rocm.docs.amd.com/
- PyTorch ROCm: https://pytorch.org/docs/stable/notes/rocm.html
