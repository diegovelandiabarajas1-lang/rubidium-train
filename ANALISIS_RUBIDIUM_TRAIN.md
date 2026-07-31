# ANÁLISIS EXHAUSTIVO: RUBIDIUM-TRAIN

## Descripción General del Proyecto

**Rubidium-Train** es un motor de entrenamiento de modelos Transformer de lenguaje escrito desde cero en C++/CUDA. Implementa un modelo de lenguaje a nivel de carácter (character-level) con arquitectura GPT-like, incluyendo:

- Entrenamiento completo forward + backward pass
- Kernels CUDA custom para todas las operaciones
- Optimizador AdamW integrado
- Generación de texto con sampling (temperature + top-k)
- Exportación de modelos a formato binario (`RBN1`) y conversión a Python pickle

El proyecto está diseñado para ejecutarse en GPUs NVIDIA (específicamente targets `sm_60` - NVIDIA P100, optimizado para Kaggle).

---

## Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje principal | C++17 |
| GPU Compute | CUDA 17 (sm_60) |
| Álgebra lineal | cuBLAS |
| Redes neuronales | cuDNN (opcional) |
| RNG GPU | cuRAND |
| Build system | CMake 3.18+ |
| Plataforma build Windows | Visual Studio 17 2022 |
| Conversión modelos | Python 3 (pickle, numpy) |
| Formato binario | RBN1 (custom) |

---

## Estructura de Directorios Completa

```
rubidium-train/
├── CMakeLists.txt              # Configuración CMake (proyecto principal)
├── build.bat                   # Script de compilación para Windows
├── convert_to_pickle.py        # Conversor binario → pickle (Python)
├── ANALISIS_RUBIDIUM_TRAIN.md  # Este archivo
└── src/
    ├── main.cpp                # Entry point, training loop, CLI
    ├── model.h                 # Definiciones: ModelConfig, LayerWeights, ModelWeights, Activations, RubidiumTransformer
    ├── model.cu                # Implementación: forward, backward, optimizer, generate, save, free
    ├── tokenizer.h             # Tokenizer a nivel de carácter (encode/decode)
    ├── cuda_kernels.cuh        # Declaraciones de kernels CUDA y helpers
    ├── cuda_kernels.cu         # Implementación de ~15 kernels CUDA custom
    ├── autograd.h              # Motor autograd experimental (Arena allocator + computation graph)
    └── convert_to_pickle.cpp   # Conversor binario → pickle (C++ puro, sin Python)
```

---

## Funcionalidades Principales

### 1. Entrenamiento de un Transformer desde cero
- Tokenización a nivel de carácter (sin BPE/subword)
- Embedding de tokens + positional encoding aprendible
- L Transformer blocks con multi-head self-attention
- Feed-forward network (two-layer MLP)
- Layer Normalization en cada bloque
- LM head para predicción de siguiente token
- Cross-entropy loss

### 2. Kernels CUDA customizados
Todas las operaciones de forward y backward están implementadas como kernels CUDA propios:
- **GEMM** (cuBLAS wrappers): forward, backward dA, backward dB
- **LayerNorm**: forward + backward con shared memory
- **Softmax**: forward + backward numéricamente estable (max-subtraction)
- **ReLU**: forward + backward
- **Cross-Entropy**: forward + backward
- **Dropout**: forward + backward (con cuRAND)
- **Embedding**: forward + backward (atomicAdd)
- **AdamW**: step completo en GPU
- **Máscara causal**: apply_causal_mask
- **Utilidades**: residual_add, gpu_copy, gpu_zero, scale_add

### 3. Generación de texto
- Sampling con temperatura (temperature scaling)
- Top-k filtering
- Seed text configurable
- Generación autoregressive carácter a carácter

### 4. Checkpointing
- Guardado de modelo en formato binario RBN1 cada 5000 steps
- Guardado final del modelo (`model_final.bin`)

### 5. Conversión a Python pickle
- `convert_to_pickle.py`: lee binario RBN1, genera pickle compatible con Rust (rubidium-core)
- `convert_to_pickle.cpp`: implementación equivalente en C++ puro

### 6. Autograd experimental (parcialmente implementado)
- Arena allocator (512 MB default)
- Tensor struct con soporte de gradientes
- Grafo de computación con nodos
- Backward pass por diferenciación automática reversa

---

## Modelos de ML/AI Implementados

### Arquitectura: Character-level GPT / Rubidium Transformer

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `V` | 256 | Tamaño del vocabulario (nivel carácter) |
| `T` | 2048 | Block size / longitud máxima de secuencia |
| `D` | 2048 | Dimensión del modelo (d_model) |
| `H` | 32 | Número de cabezas de atención |
| `L` | 10 | Número de capas Transformer |
| `FF` | 8192 | Dimensión de la red feed-forward |
| `hd` | 64 | Dimensión por cabeza (D/H) |

### Componentes del modelo:
1. **Token Embedding**: V × D
2. **Positional Embedding**: T × D (aprendible, no sinusoidal)
3. **L bloques Transformer**:
   - LayerNorm → Multi-Head Self-Attention → Residual → LayerNorm → FFN → Residual
4. **LayerNorm final**
5. **LM Head**: linear(V, D) + bias

### Parámetros totales estimados:
```
~113M parámetros con la configuración por defecto
```

---

## Datasets Utilizados

El proyecto **no incluye datasets explícitos**. Está diseñado para consumir directorios de archivos `.txt`:

- **Entrada**: directorio con archivos de texto plano
- **Ruta por defecto**: `data/`
- **Ruta configurable**: primer argumento CLI
- **Formato**: busca `*.txt` vía glob

La tokenización es a nivel de carácter, por lo que funciona con cualquier texto en cualquier idioma. Los seeds de prueba están en español ("Hola", "Buenos dias", "Quien eres", "Que puedes hacer").

---

## Pipeline de Entrenamiento

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Carga de corpus (archivos .txt)                             │
├─────────────────────────────────────────────────────────────────┤
│ 2. Construcción de vocabulario (char → int)                     │
├─────────────────────────────────────────────────────────────────┤
│ 3. Inicialización del modelo (pesos Gaussianos σ=0.02)         │
├─────────────────────────────────────────────────────────────────┤
│ 4. Loop de entrenamiento (200,000 steps):                       │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │ a. LR Schedule: warmup 4000 steps + cosine decay       │  │
│    │ b. Gradient Accumulation (GA=16 micro-batches)         │  │
│    │    - Sample batch aleatorio del corpus                  │  │
│    │    - Copiar tokens a GPU                                │  │
│    │    - Forward pass (compute loss)                        │  │
│    │    - Backward pass (compute gradients)                  │  │
│    │ c. Gradient clipping (max_norm=1.0)                     │  │
│    │ d. Optimizer step (AdamW)                               │  │
│    └─────────────────────────────────────────────────────────┘  │
│    - Log每100 steps (loss, lr, speed, ETA)                     │
│    - Checkpoint每5000 steps (model_step_N.bin)                 │
├─────────────────────────────────────────────────────────────────┤
│ 5. Guardado final (model_final.bin)                             │
├─────────────────────────────────────────────────────────────────┤
│ 6. Quick test: generación de texto con 4 seeds                  │
├─────────────────────────────────────────────────────────────────┤
│ 7. Liberación de memoria GPU                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Hiperparámetros de entrenamiento:
| Parámetro | Valor |
|---|---|
| Batch size (BS) | 2 |
| Gradient accumulation (GA) | 16 |
| Effective batch size | 32 |
| Max steps | 200,000 |
| Learning rate | 3e-4 |
| β1 (Adam) | 0.9 |
| β2 (Adam) | 0.999 |
| ε (Adam) | 1e-8 |
| Weight decay | 0.1 |
| Warmup steps | 4,000 |
| Gradient clipping | 1.0 |

### Learning Rate Schedule:
- **Warmup**: lineal de 0 a lr durante los primeros 4000 steps
- **Decay**: cosine annealing desde lr hasta 0 durante el resto del entrenamiento

---

## Dependencias Principales

### Requisitos de compilación:
| Dependencia | Tipo | Requerido |
|---|---|---|
| NVIDIA CUDA Toolkit | Runtime | Sí |
| cuBLAS | Library | Sí |
| cuDNN | Library | No (opcional) |
| cuRAND | Library | Sí |
| CMake ≥ 3.18 | Build tool | Sí |
| Visual Studio 2022 | IDE/compiler | Sí (Windows) |
| C++17 compiler | Compiler | Sí |

### Para conversión de modelos:
| Dependencia | Tipo | Requerido |
|---|---|---|
| Python 3 | Runtime | Solo para `convert_to_pickle.py` |
| numpy | Python lib | Solo para `convert_to_pickle.py` |
| pickle | Python stdlib | Solo para `convert_to_pickle.py` |

### No usa:
- PyTorch / TensorFlow / frameworks de ML
- Librerías de tokenizer externas
- Nessuna dependencia de C++ externa (todo custom)

---

## Hallazgos Interesantes

### 1. Arquitectura "desde cero"
No se usa ningún framework de ML. Todo está implementado desde los kernels CUDA hasta el optimizador AdamW. Esto es notable por:
- Implementar backpropagation manualmente
- cuBLAS wrappers propios para GEMM
- Layer Norm, Softmax, ReLU todos como kernels CUDA custom
- Even includes `glob.h` (POSIX) para carga de archivos

### 2. Dual pickle converter
Existen dos implementaciones para convertir modelos a pickle:
- **Python** (`convert_to_pickle.py`): más simple, usa numpy
- **C++** (`convert_to_pickle.cpp`): reimplementa el formato pickle desde cero con opcodes

### 3. Autograd experimental incompleto
`autograd.h` contiene un motor autograd que:
- Implementa Arena allocator (512 MB)
- Define nodos de grafo de computación
- Tiene tipos de operación (OP_ADD, OP_MATMUL, etc.)
- **No está integrado** con el modelo principal (model.cu no lo usa)

### 4. Formato binario RBN1
- Header: magic "RBN1" (4 bytes)
- Config: 6 ints (V, T, D, H, L, FF)
- Mapa char_to_id: 256 ints
- Pesos: secuencial por capa

### 5. Atención multi-head simplificada
El backward pass de atención usa un enfoque "simplified" donde se usa `d_ao` (output attention) para aproximar los gradientes de Q, K, V. Esto puede causar diferencias numéricas vs un backward exacto.

### 6. Target GPU: NVIDIA P100 (Kaggle)
- Arquitectura CUDA: sm_60
- El CMakeLists.txt menciona paths de Kaggle para cuDNN
- Optimizaciones: `-O3`, `--use_fast_math`, `-march=native`

### 7. Tokenización a nivel de carácter
- No usa BPE, WordPiece, ni SentencePiece
- Vocabulario dinámico (máx 256, un byte = un token)
- Funciona con cualquier idioma sin preprocesamiento

### 8. Seeding de cuRAND
El dropout kernel usa `curand_init(42, i, 0, &state)` con semilla fija y thread ID como offset, lo que podría causar patrones de dropout repetitivos.

### 9. Manejo de memoria GPU intensivo
- En `backward()`, se hacen múltiples `cudaMalloc`/`cudaFree` dentro del loop de capas
- No se usa memory pooling para gradientes intermedios

### 10. Seeds de generación en español
Los seeds de prueba ("Hola", "Buenos dias", "Quien eres", "Que puedes hacer") sugieren que el proyecto está orientado a generación de texto en español.

---

## Posibles Mejoras

### Críticas
1. **Backward pass de atención incompleto**: El backward de Q, K, V está simplificado. Un backward exacto de multi-head attention requiere gradientes separados para Q, K, V con máscara causal.

2. **Memory leaks en backward**: Se hacen `cudaMalloc`/`cudaFree` en cada iteración del loop de capas, lo cual es lento y puede causar fragmentación. Se debería pre-allocar todo el gradiente intermedio al inicio.

3. **Dropout no integrado**: Aunque `dropout_forward`/`dropout_backward` están implementados, nunca se llaman en el forward/backward del modelo. No hay regularización.

4. **Checkpoint sin optimizer state**: Los checkpoints guardan solo pesos, no los momentos del optimizador (m, v). Esto impide reanudar entrenamiento exacto.

### Rendimiento
5. **Gradient checkpointing**: No se usa. Podría reducir uso de VRAM significativamente a cambio de recomputar activaciones.

6. **Mixed precision (FP16/BF16)**: Todo está en FP32. Mixed precision podría 2x-3x el throughput.

7. **Flash Attention**: No se implementa. Flash Attention 2+ reduce el uso de memoria de O(T²) a O(T).

8. **Multi-GPU**: No hay soporte. Podría añadirse con NCCL.

9. **Memory pooling**: Reusar buffers de gradiente intermedio en lugar de malloc/free por capa.

### Funcionalidad
10. **Data loading asíncrono**: Actualmente carga todo el corpus en RAM. Para datasets grandes se necesita streaming o memory mapping.

11. **LR scheduler configurable**: El warmup + cosine está hardcodeado. Podría hacerse configurable.

12. **Métricas de validación**: No hay split train/val. Solo se reporta loss de entrenamiento.

13. **Logging**: Solo imprime a stdout. Podría integrar TensorBoard o Weights & Biases.

14. **Autograd integrado**: El `autograd.h` tiene potencial pero no está conectado al modelo. Podría reemplazar el backward manual.

15. **Soporte para GPU compute capability > 60**: Actualmente solo sm_60. Podría detectar automáticamente la GPU.

16. **Residuos de `convert_to_pickle.cpp`**: El PickleWriter tiene una función `write_float()` incompleta (comment: "Actually this is getting too complex"). Debería completarse o eliminarse.

17. **`linear_backward` con bug potencial**: La línea 569 usa `nullptr` como puntero en lugar de crear un vector de unos, lo que causaría undefined behavior con cuBLAS.

18. **No hay validación del corpus**: Si el directorio está vacío o no tiene `.txt`, falla silenciosamente.

### Código
19. **Falta `.gitignore`**: No hay `.gitignore`, los binarios build se commitearían accidentalmente.

20. **Falta README**: No hay documentación de uso, solo el build.bat.

21. **Inconsistencia C++/POSIX**: `glob.h` es POSIX, no portable a Windows. El build.bat asume Visual Studio pero `main.cpp` usa `glob()`.

22. **Usa `srand(42)` / `rand()`:** El generador de números aleatorios de C++ estándar es de baja calidad para inicialización de pesos. Debería usar `<random>` o cuRAND.

---

## Resumen

Rubidium-Train es un proyecto ambicioso y educativo que implementa un Transformer completo de entrenamiento en CUDA/C++ desde cero. A pesar de estar en fase prototipo (backward simplificado, sin dropout, autograd no integrado), demuestra una comprensión profunda de:
- Arquitectura Transformer
- Programación GPU CUDA
- Optimización numérica (mixed precision, gradient clipping)
- Álgebra lineal en GPU (cuBLAS)

Es ideal como base para aprender sobre training engines custom o como punto de partida para un sistema de entrenamiento más robusto.
