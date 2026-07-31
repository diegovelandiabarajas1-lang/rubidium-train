# Analisis Detallado - rubidium-train

> Motor de entrenamiento Transformer escrito en CUDA/C++ desde cero, sin frameworks de deep learning.

---

## 1. Tipo de Modelo

**Language Model (Decoder-only Transformer)** - Modelo de lenguaje autoregresivo para prediccion de siguiente token (next-token prediction). Entrenado a nivel de caracteres (character-level).

- **Dominio**: NLP - Modelado de lenguaje
- **Tipo de tarea**: Prediccion secuencial (generacion de texto)
- **Tokenizacion**: Character-level (256 caracteres ASCII)
- **Idioma del corpus de prueba**: Espanol (`"Hola"`, `"Buenos dias"`, `"Quien eres"`, `"Que puedes hacer"`)
- **Formato de inferencia**: Autoregresivo con temperature sampling + top-k

---

## 2. Arquitectura del Modelo

### 2.1 Configuracion por defecto (`model.h:18`)

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| `V` | 256 | Tamano del vocabulario (caracteres ASCII) |
| `T` | 256 | Longitud maxima de contexto (block_size / sequence length) |
| `D` | 2048 | Dimension del embedding (d_model) |
| `H` | 32 | Numero de cabezas de atencion (n_head) |
| `L` | 10 | Numero de capas Transformer (n_layer) |
| `FF` | 8192 | Dimension de la capa feed-forward (d_ff) |
| `hd` | 64 | Dimension por cabeza (D/H = 2048/32) |

### 2.2 Conteo de Parametros

Calculado en `main.cpp:109-112`:

```
Token Embedding:     V * D         =     524,288
Position Embedding:  T * D         =     524,288
Per-layer (x10):
  LayerNorm x2:      2 * (2*D)     =      16,384
  Q/K/V/O weights:   4 * D^2       =  16,777,216
  Q/K/V/O biases:    4 * D         =       8,192
  FFN W1, W2:        2 * D * FF    =  33,554,432
  FFN b1, b2:        FF + D        =      10,240
  Subtotal/layer:                  =   50,366,464
  x10 layers:                      =  503,664,640
Final LayerNorm:     2 * D         =       4,096
LM Head (W+b):      V*D + V        =     524,544
─────────────────────────────────────────────────
TOTAL:                             ≈ 505,241,856 (~505M parametros)
```

### 2.3 Arquitectura Interna por Capa Transformer

Cada una de las 10 capas (`model.cu:160-213`) implementa **Pre-Norm Transformer Block**:

```
Input x0
  │
  ├─► LayerNorm 1 (pre-norm)
  │     └─► Linear(D → D) x4: Q, K, V projections
  │           └─► Multi-Head Self-Attention (H=32 heads, hd=64)
  │                 └─► Q·K^T / sqrt(hd)  →  Causal Mask  →  Softmax  →  ·V
  │                       └─► Linear(D → D) output projection
  │
  ├─► Residual Add (x0 + attention_output)  →  x1
  │
  ├─► LayerNorm 2 (pre-norm)
  │     └─► Linear(D → FF=8192)  →  ReLU  →  Linear(FF → D)
  │
  └─► Residual Add (x1 + ffn_output)  →  output
```

**Detalles clave de la implementacion:**
- **Atencion**: Multi-head con cuBLAS para las multiplicaciones Q·K^T y attn·V (`model.cu:175-197`)
- **Mascara causal**: Implementada como kernel CUDA que pone `-1e9` en posiciones futuras (`cuda_kernels.cu:521-533`)
- **Activacion FFN**: ReLU (no GELU como en GPT) (`model.cu:208`)
- **Normalizacion**: LayerNorm con epsilon=1e-5 (`cuda_kernels.cuh:35`)
- **Sin Dropout**: Aunque el kernel esta implementado, NO se aplica en el forward pass

### 2.4 Pipeline Completo del Forward Pass (`model.cu:151-226`)

1. **Token Embedding** + **Positional Embedding** (suma aditiva)
2. **L x Transformer Blocks** (Pre-LN + Attention + Residual + Pre-LN + FFN + Residual)
3. **Final LayerNorm**
4. **LM Head** (Linear projection a vocabulario)
5. **Cross-Entropy Loss** (si hay targets)

---

## 3. Hiperparametros Configurados

### 3.1 Hiperparametros de Entrenamiento (`main.cpp:67-70`)

| Hiperparametro | Valor | Descripcion |
|---------------|-------|-------------|
| `BS` | 2 | Batch size por GPU |
| `GA` | 16 | Gradient accumulation steps |
| **Effective Batch** | **32** | BS * GA = 2 * 16 |
| `max_steps` | 200,000 | Pasos totales de entrenamiento |
| `lr` | 3e-4 | Learning rate maximo |
| `b1` | 0.9 | Adam beta1 (momento) |
| `b2` | 0.999 | Adam beta2 (segundo momento) |
| `eps` | 1e-8 | Adam epsilon (estabilidad numerica) |
| `wd` | 0.1 | Weight decay (regularizacion L2) |
| `warmup` | 4,000 | Pasos de warmup del LR |
| `gc` | 1.0 | Gradient clipping (norma maxima) |
| `temperature` | 0.7 | Temperatura para inferencia |
| `top_k` | 40 | Top-k sampling en generacion |

### 3.2 LR Schedule (`main.cpp:123-128`)

**Cosine Annealing con Linear Warmup:**

```
if step < warmup (4000):
    lr_t = lr * step / warmup    (linear warmup)
else:
    p = (step - warmup) / (max_steps - warmup)
    lr_t = lr * 0.5 * (1 + cos(pi * p))   (cosine decay a 0)
```

### 3.3 Inicializacion de Pesos (`model.cu:81-120`)

- **Seed**: `srand(42)` (reproducible)
- **Metodo**: Distribucion Normal (Box-Muller) con std=0.02 para todos los pesos entrenables
- **Bias**: Inicializados en 0
- **LayerNorm weights**: Inicializados en 1.0, bias en 0.0

---

## 4. Pipeline de Preprocesamiento de Datos

### 4.1 Carga de Corpus (`main.cpp:20-41`)

- **Formato**: Archivos `.txt` en directorio (por defecto `"data"`)
- **Carga**: Usa `glob()` para encontrar todos los `.txt`, los lee como binario y los concatena con `\n`
- **Encoding**: Raw bytes (sin normalizacion Unicode)

### 4.2 Tokenizacion (`tokenizer.h`, `main.cpp:82-98`)

- **Tipo**: Character-level tokenizer (256 posibles caracteres ASCII)
- **Vocabulario**: Se construye dinamicamente desde el corpus
- **Mapeo**: `char_to_id` (map unsigned char → int) e `id_to_char` (inverso)
- **Encoding**: Cada byte se mapea a su ID correspondiente
- **Desconocidos**: Caracteres no vistos se mapean a ID 0

### 4.3 Preparacion de Batches (`main.cpp:134-142`)

```cpp
// Muestreo aleatorio de indices
for (auto &i : idx) i = rand() % (n - cfg.T - 1);

// Ventanas deslizantes de longitud T
for (int b = 0; b < BS; b++)
    for (int t = 0; t < cfg.T; t++) {
        h_tok[b*T+t] = data[idx[b]+t];      // tokens de entrada
        h_tgt[b*T+t] = data[idx[b]+t+1];     // targets (shifted by 1)
    }
```

- **Sin validacion/train split**: Todos los datos se usan para entrenamiento
- **Sin augmentation**: Datos sin transformaciones
- **Transferencia GPU**: Cada batch se copia a GPU con `cudaMemcpy`

---

## 5. Metricas de Evaluacion

### 5.1 Metrica Principal

- **Cross-Entropy Loss** (perplexity implicita): Implementada como kernel CUDA numericamente estable con log-sum-exp trick (`cuda_kernels.cu:306-359`)

### 5.2 Metricas Registradas (`main.cpp:170-178`)

```cpp
// Loss suavizado (exponential moving average)
smooth_loss = 0.98 * smooth_loss + 0.02 * avg_loss

// Logging cada 100 pasos
printf("Step %d/%d | loss: %.4f | lr: %.2e | %.1f steps/s | ETA: %.0fmin\n",
       step, max_steps, smooth_loss, lr_t, sps, eta);
```

- **Loss promedio**: Promedio del loss sobre los GA mini-batches
- **Loss suavizado**: EMA con factor 0.98
- **Velocidad**: Steps por segundo
- **Tiempo estimado**: ETA en minutos

### 5.3 Evaluacion Cualitativa (`main.cpp:196-201`)

```cpp
// Generacion con seeds en espanol
std::vector<std::string> seeds = {"Hola", "Buenos dias", "Quien eres", "Que puedes hacer"};
// Genera 120 caracteres por seed, temp=0.7, top_k=40
```

**No hay conjuntos de validacion ni metricas cuantitativas de evaluacion aparte del loss de entrenamiento.**

---

## 6. Resultados de Entrenamiento

### 6.1 Datos Disponibles

No se proporcionan logs de entrenamiento ni checkpoints entrenados en el repositorio. La configuracion permite estimar:

- **200,000 pasos** a ~1-2 steps/segundo (estimado en P100)
- **Tiempo estimado**: ~28-56 horas
- **Checkpoints**: Cada 5,000 pasos (`model_step_5000.bin`, ..., `model_step_200000.bin`)
- **Modelo final**: `model_final.bin` en formato binario personalizado `RBN1`

### 6.2 Formato del Modelo Exportado

```
Magic: "RBN1" (4 bytes)
Config: V, T, D, H, L, FF (6 x int32)
Char map: 256 x int32
Pesos: token_emb, pos_emb, L capas (LN + QKVO + FFN), ln_f, lm_head
```

### 6.3 Conversion a Pickle

- **`convert_to_pickle.py`**: Convierte el binario RBN1 a pickle de Python compatible con Rust (rubidium-core)
- **`convert_to_pickle.cpp`**: Implementacion en C++ del mismo conversor
- **Formato destino**: Diccionario Python con numpy arrays, listo para inferencia en Rust

---

## 7. Configuracion de Hardware/GPU

### 7.1 Target Principal

- **GPU**: NVIDIA P100 (Tesla P100)
- **Arquitectura CUDA**: `sm_60` (Pascal)
- **VRAM**: ~16 GB (HBM2)
- **Entorno**: Kaggle Notebooks (referencia a rutas de Kaggle en `CMakeLists.txt:12-13`)

### 7.2 Dependencias

| Componente | Uso |
|-----------|-----|
| CUDA Toolkit | Runtime GPU, kernels custom |
| cuBLAS | Multiplicaciones de matriz (GEMM) para attention, linears |
| cuDNN | Opcional (no se usa activamente en kernels) |
| cuRAND | Generacion de numeros aleatorios para dropout (definido, no usado) |
| CMake 3.18+ | Sistema de build |
| C++17 | Estandar del compilador |
| CUDA 17 | Estandar del compilador CUDA |

### 7.3 Flags de Optimizacion (`CMakeLists.txt:73-84`)

```cmake
# CUDA
-O3                          # Optimizacion maxima
--use_fast_math              # Operaciones matematicas rapidas (menos precision)
-lineinfo                    # Info de lineas para profiling
-gencode=arch=compute_60,code=sm_60

# C++
-O3                         # Optimizacion maxima
-march=native                # Optimizar para CPU local (build host)
```

### 7.4 Uso Estimado de VRAM

```
Pesos del modelo:        ~2.0 GB  (505M * 4 bytes)
Gradientes:              ~2.0 GB
Adam m + v moments:      ~4.0 GB  (2 * 505M * 4 bytes)
Activaciones (max_BT):   ~1.5 GB  (estimado para BS=2, T=256)
Bufferes temporales:     ~0.5 GB
─────────────────────────────────
TOTAL estimado:         ~10.0 GB
```

Dentro del limite de 16 GB de la P100.

---

## 8. Optimizaciones Aplicadas

### 8.1 En el Compute

| Optimizacion | Ubicacion | Descripcion |
|-------------|-----------|-------------|
| **cuBLAS GEMM** | `cuda_kernels.cu:38-78` | Multiplicaciones de matriz optimizadas por NVIDIA |
| **Fast Math** | `CMakeLists.txt:76` | `--use_fast_math` para operaciones float rapidas |
| **Shared Memory** | Todos los kernels | Reduccion paralela con shared memory para sumas/max |
| **Kernel Fusion** (limitado) | `cuda_kernels.cu:462-480` | AdamW step fusionado en un solo kernel |
| **Gradient Accumulation** | `main.cpp:67` | GA=16 para simular batch grande con poca VRAM |
| **Gradient Clipping** | `model.cu:367-398` | Norma maxima de 1.0 para estabilidad |

### 8.2 En la Arquitectura

| Optimizacion | Descripcion |
|-------------|-------------|
| **Pre-Norm** | LayerNorm antes de attention/FFN (mas estable que Post-Norm) |
| **Weight Tying** (implicito) | token_emb y lm_head comparten dimension V*D |
| **Efficient Attention** | cuBLAS para Q*K^T y attn*V (no kernel custom) |
| **Causal Mask** | Kernel simple que pone -1e9 en posiciones futuras |

### 8.3 En la Infraestructura

| Optimizacion | Descripcion |
|-------------|-------------|
| **Serialización binaria** | Formato RBN1 compacto (sin metadata innecesaria) |
| **Conversion pickle** | Exporta a formato estandar para inferencia en otros lenguajes |
| **Arena Allocator** (autograd.h) | Asignador de memoria en bloque (512MB) para grafo de computacion |

---

## 9. Analisis de Overfitting/Underfitting

### 9.1 Factores que Sugieren Underfitting

1. **Batch size efectivo pequeno**: BS=2 con GA=16 da effective batch=32, pero el modelo tiene ~505M parametros. Para un modelo de este tamano, se esperaria batch sizes mas grandes.
2. **Sin validacion**: No hay forma de medir overfitting objetivamente.
3. **Dropout no utilizado**: Aunque el kernel existe (`cuda_kernels.cu:392-419`), **nunca se invoca** en el forward pass. Esto significa que el modelo no tiene regularizacion por dropout.
4. **Activacion ReLU**: Menos suave que GELU, podria limitar la capacidad de representacion.
5. **Capas limitadas**: Solo 10 capas para un modelo de ~505M parametros. Modelos similares (GPT-2 medium) usan 24 capas con dimensiones menores.

### 9.2 Factores de Regularizacion Presentes

1. **Weight Decay**: wd=0.1 (regularizacion L2 efectiva)
2. **Gradient Clipping**: max_norm=1.0 (prevenir explosion de gradientes)
3. **LR Warmup**: 4000 pasos de warmup (prevenir divergence inicial)
4. **Cosine LR Decay**: Reduce LR gradualmente

### 9.3 Riesgo de Overfitting

- **Alto riesgo** si el corpus es pequeno. Un modelo de ~505M parametros puede memorizar rapidamente corpus de texto pequenos.
- **Sin early stopping** ni monitoreo de validation loss.
- **Corpus no especificado**: El directorio `"data"` debe contener archivos `.txt`, pero no se proporciona informacion sobre el tamano del corpus.

### 9.4 Evaluacion Cualitativa

La unica forma de evaluar es la generacion con seeds (`main.cpp:196-201`). No hay metricas como:
- Perplexity en validation set
- BLEU/ROUGE scores
- Human evaluation

---

## 10. Problemas Identificados y Limitaciones

### 10.1 Bugs / Limitaciones Codigo

| Problema | Ubicacion | Impacto |
|---------|-----------|---------|
| **Backward de atencion simplificado** | `model.cu:297-308` | El backward NO propaga gradientes correctamente a traves de K y V en la attention. Solo propaga a traves de Q. Esto degrada significativamente el entrenamiento. |
| **Sin dropout en forward** | `model.cu:160-213` | El kernel de dropout esta implementado pero nunca se llama. No hay regularizacion por dropout. |
| **Alloc/dealloc en backward** | `model.cu:253-326` | Multiples `cudaMalloc`/`cudaFree` en cada backward pass. Ineficiente y fragmentation de memoria. |
| **GEMM backward dB** | `cuda_kernels.cu:549-574` | El calculo de gradiente de bias usa un placeholder `nullptr` como parametro de cuBLAS, lo cual es incorrecto. |
| **Vocabulario limitado a 256** | `model.h:18` | Solo soporta caracteres ASCII. Sin soporte para Unicode/UTF-8 multibyte. |
| **Sin validation split** | `main.cpp` | No hay separacion train/validation. Imposible detectar overfitting. |
| **`glob.h` no portable** | `main.cpp:15` | `glob.h` no existe en Windows. El `build.bat` sugiere Windows, pero el codigo fuente requiere POSIX. |
| **Linear backward bias** | `cuda_kernels.cu:560-573` | El calculo del gradiente del bias no es correcto - usa `nullptr` como matrix en cuBLAS. |

### 10.2 Limitaciones de Arquitectura

1. **Sin Multi-Query Attention (MQA) o Grouped-Query Attention (GQA)**: Cada head tiene sus propios Q, K, V. Para inferencia, esto es menos eficiente en memoria.
2. **Sin Flash Attention**: No se usa la implementacion eficiente de atencion.
3. **Sin Rotary Position Embeddings (RoPE)**: Usa positional embeddings absolutos aprendidos, que generalizan peor a secuencias largas.
4. **Sin bias en attention**: GPT-2/3 no usan bias en las proyecciones QKV, este modelo si.
5. **Sin scaled dot-product attention optimizado**: La implementacion es correcta pero no usa cuDNN para la fusión de kernels.

---

## 11. Proximos Pasos Sugeridos

### 11.1 Correcciones Criticas

1. **Corregir el backward de attention**: Implementar la propagacion correcta de gradientes a traves de Q, K, V y las mascaras de atencion. Sin esto, el modelo no puede aprender representaciones de atencion efectivas.

2. **Implementar dropout en forward**: Activar el dropout (p=0.1 recomendado) en:
   - Despues de la capa de atencion (antes del residual)
   - Despues de la capa FFN (antes del residual)

3. **Corregir el backward del bias**: La funcion `linear_backward` tiene un bug con `nullptr` en cuBLAS.

4. **Reemplazar `glob.h`**: Usar alternativa portable o pre-cargar la lista de archivos en el script de build.

### 11.2 Mejoras de Rendimiento

5. **Flash Attention**: Implementar o integrar Flash Attention 2 para reducir uso de O(T^2) en memoria y mejorar throughput.

6. **Mixed Precision (FP16/BF16)**: Reducir uso de VRAM a la mitad y aumentar throughput con tensor cores.

7. **Pooling de memoria**: Reemplazar los multiples `cudaMalloc`/`cudaFree` en backward con un memory pool o arena.

8. **Compilation kernels**: Fusionar kernels simples (residual_add, scale_add) cuando sea posible.

### 11.3 Mejoras de Arquitectura

9. **GQA (Grouped Query Attention)**: Compartir K y V entre grupos de heads para reducir parametros y memoria.

10. **RoPE (Rotary Position Embeddings)**: Reemplazar positional embeddings absolutos para mejor generalizacion a secuencias largas.

11. **SwiGLU en FFN**: Reemplazar ReLU por SwiGLU (como en LLaMA) para mejor rendimiento.

12. **Scaling Laws**: Realizar experimentos con diferentes configuraciones (D, L, FF) para encontrar la optima para el hardware disponible.

### 11.4 Mejoras de Entrenamiento

13. **Validation split**: Separar 5-10% de datos para validacion y monitorear overfitting.

14. **Learning Rate finder**: Encontrar el LR optimal antes del entrenamiento completo.

15. **Wandb/TensorBoard**: Integrar logging para monitorear metricas en tiempo real.

16. **EMA weights**: Mantener copia de pesos con exponential moving average para evaluacion.

17. **Curriculum learning**: Entrenar primero en secuencias cortas y aumentar gradualmente.

### 11.5 Pipeline de Datos

18. **Tokenizer BPE/WordPiece**: Reemplazar character-level por subword tokenization para mejor cobertura.

19. **Data loader eficiente**: Implementar mmap o streaming en vez de cargar todo el corpus en RAM.

20. **Data augmentation**: Considerar masking, synonym replacement, o back-translation.

---

## 12. Resumen Ejecutivo

| Aspecto | Estado |
|---------|--------|
| **Arquitectura** | Transformer decoder-only, 505M params, Pre-Norm, ReLU |
| **Implementacion** | CUDA/C++ desde cero, kernels custom, cuBLAS para GEMM |
| **Entrenamiento** | AdamW, cosine LR, 200K steps, batch=32 |
| **Hardware** | NVIDIA P100 (sm_60), ~10GB VRAM estimado |
| **Madurez** | Prototipo funcional con bugs criticos en backward pass |
| **Prioridad #1** | Corregir backward de attention (bloquea aprendizaje efectivo) |
| **Prioridad #2** | Activar dropout y agregar validation split |
| **Prioridad #3** | Mixed precision + Flash Attention para eficiencia |

El proyecto demuestra una implementacion completa de un Transformer desde scratch en CUDA, pero el **backward de attention simplificado** es una limitacion critica que impide el entrenamiento efectivo del modelo. Corregir esto deberia ser la prioridad inmediata antes de cualquier otra optimizacion.
