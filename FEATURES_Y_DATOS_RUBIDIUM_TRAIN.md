# FEATURES_Y_DATOS_RUBIDIUM_TRAIN.md

Documentación completa de datos, features, transformaciones y utilidades del proyecto **rubidium-train**.

---

## 1. Fuentes de datos utilizadas

| Fuente | Descripción | Ubicación |
|--------|-------------|-----------|
| Corpus de texto plano | Archivos `.txt` concatenados | Directorio `data/` (configurable via CLI) |
| Formato binario del modelo | Checkpoints RBN1 | `checkpoints/model_step_*.bin` |
| Conversión a pickle | Modelo exportado para inferencia Rust/Python | `model_final.pkl` |

### Carga de corpus (`src/main.cpp:20-41`)

```cpp
std::string load_corpus(const std::string &path) {
    glob_t glob_result;
    glob((path + "/*.txt").c_str(), GLOB_NOSORT, nullptr, &glob_result);
    // Concatena todos los archivos .txt con "\n" como separador
}
```

- Se usan **archivos `.txt`** como fuente de entrenamiento.
- Se concatenan en un solo string con `\n` entre archivos.
- No hay soporte para otros formatos (CSV, JSON, etc.).
- El corpus se carga completamente en memoria RAM como `std::string`.

---

## 2. Tokenización y vocabulario

### Tokenizer (`src/tokenizer.h`)

| Característica | Valor |
|----------------|-------|
| Tipo | **Character-level tokenizer** (nivel de carácter) |
| Vocabulario máximo | 256 (bytes, `unsigned char`) |
| IDs | Asignados secuencialmente en orden de aparición |
| Carácter desconocido | ID `0` para caracteres no vistos |

### Proceso de tokenización (`src/main.cpp:82-98`)

```cpp
// Construcción del vocabulario
for (unsigned char c : full_text) {
    if (c2i.find(c) == c2i.end()) {
        int id = c2i.size();
        c2i[c] = id;
        i2c[id] = c;
    }
}
int V = c2i.size();  // Tamaño real del vocabulario

// Encoding del corpus
std::vector<int> data(full_text.size());
for (size_t i = 0; i < full_text.size(); i++)
    data[i] = c2i[(unsigned char)full_text[i]];
```

- No se usa BPE ni subword tokenization.
- Cada carácter se mapea a un entero único.
- El vocabulario se adapta dinámicamente al corpus.

---

## 3. Features creadas

### 3.1 Token Embeddings

| Propiedad | Valor | Descripción |
|-----------|-------|-------------|
| Forma | `[V, D]` | Matriz de embeddings de tokens |
| Inicialización | Normal gaussiana `N(0, 0.02)` | (`src/model.cu:93`) |
| Parámetros | `V * D` | Ej: 256 * 2048 = 524,288 |

### 3.2 Positional Embeddings

| Propiedad | Valor | Descripción |
|-----------|-------|-------------|
| Forma | `[1, T, D]` | Embeddings posicionales aprendidos |
| Tipo | Absolutos (learned) | No rotary, no sinusoidal |
| Inicialización | Normal gaussiana `N(0, 0.02)` | (`src/model.cu:94`) |
| Parámetros | `T * D` | Ej: 256 * 2048 = 524,288 |

### 3.3 Multi-Head Attention

| Propiedad | Valor |
|-----------|-------|
| Cabezas (H) | 32 |
| Dimensión por cabeza (hd) | `D / H` = 64 |
| Masks | Causal mask (autoregressive) |
| Escala | `1 / sqrt(hd)` |

### 3.4 Feed-Forward Network

| Propiedad | Valor |
|-----------|-------|
| Dimensión oculta (FF) | 8192 (4x el modelo) |
| Activación | ReLU (no GELU) |
| Capas | Linear → ReLU → Linear |

### 3.5 Layer Normalization

| Propiedad | Valor |
|-----------|-------|
| Tipo | Pre-LayerNorm (antes de cada sub-bloque) |
| Epsilon | `1e-5` |
| Inicialización | pesos=1.0, bias=0.0 |

---

## 4. Transformaciones aplicadas

### 4.1 Pipeline de Forward (`src/model.cu:151-226`)

```
Input tokens [B, T]
    ↓
Token Embedding [B, T, D] + Positional Embedding [1, T, D]
    ↓
Para cada capa L:
    ├── LayerNorm → Q, K, V
    ├── Multi-Head Attention (con causal mask + softmax)
    ├── Proyección de salida + Residual
    ├── LayerNorm
    ├── FFN (Linear → ReLU → Linear) + Residual
    ↓
LayerNorm final
    ↓
LM Head (Linear → logits [B, T, V])
    ↓
Cross-Entropy Loss
```

### 4.2 Normalización de attention scores

```cpp
float scale = 1.0f / sqrtf((float)cfg.hd);
// Se aplica antes de softmax
```

### 4.3 Causal Mask

```cpp
// Se aplica -1e9 a posiciones futuras
if (t2 > t1) att[idx] = -1e9f;
```

### 4.4 Softmax

- Implementación custom en CUDA con numerically stable max subtraction.
- Shmem usage: `2 * threads * sizeof(float)`.

### 4.5 Cross-Entropy Loss

- Implementación custom con **label smoothing implícito** vía epsilon `1e-10`.
- Promedio sobre batch y secuencia.

---

## 5. Limpieza de datos

| Operación | Implementada | Detalle |
|-----------|-------------|---------|
| Remoción de caracteres especiales | No | Se leen bytes crudos |
| Normalización Unicode | No | Solo `unsigned char` (0-255) |
| Filtrado de líneas vacías | No | Se conservan tal cual |
| Separación de archivos | Sí | `\n` entre archivos |
| Deduplicación | No | Los archivos se concatenan directamente |

**Observación**: La limpieza es mínima. El modelo procesa bytes crudos sin transformaciones lingüísticas.

---

## 6. Ingeniería de features

### 6.1 Configuración por defecto (`src/main.cpp:63-64`)

```cpp
ModelConfig cfg;
cfg.init(256, 256, 2048, 32, 10, 8192);
// V=256, T=256, D=2048, H=32, L=10, FF=8192
```

| Parámetro | Símbolo | Valor por defecto | Descripción |
|-----------|---------|-------------------|-------------|
| Vocabulario | V | 256 | Tamaño máximo (se ajusta al corpus) |
| Contexto | T | 256 | Longitud máxima de secuencia |
| Dimensión modelo | D | 2048 | Dimensión del embedding |
| Cabezas atención | H | 32 | Número de cabezas de atención |
| Capas transformer | L | 10 | Número de bloques transformer |
| Dimensión FF | FF | 8192 | Dimensión de la red feed-forward |

### 6.2 Hiperparámetros de entrenamiento (`src/main.cpp:67-70`)

```cpp
int BS = 2, GA = 16, max_steps = 200000;
float lr = 3e-4f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.1f;
int warmup = 4000;
float gc = 1.0f;  // gradient clipping
```

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| Batch size (BS) | 2 | Muestras por batch |
| Gradient accumulation (GA) | 16 | Acumulación de gradientes |
| Effective batch size | 32 | `BS * GA` |
| Learning rate | 3e-4 | Tasa de aprendizaje máxima |
| β1 | 0.9 | Momento de primer orden (Adam) |
| β2 | 0.999 | Momento de segundo orden (Adam) |
| Epsilon | 1e-8 | Estabilidad numérica |
| Weight decay | 0.1 | Regularización L2 |
| Warmup steps | 4000 | Pasos de calentamiento LR |
| Gradient clipping | 1.0 | Norma máxima de gradientes |
| Max steps | 200,000 | Total de pasos de entrenamiento |

### 6.3 Learning Rate Schedule (`src/main.cpp:123-128`)

```
if step < warmup:
    lr_t = lr * step / warmup        # Lineal creciente
else:
    p = (step - warmup) / (max_steps - warmup)
    lr_t = lr * 0.5 * (1 + cos(π * p))  # Cosine annealing
```

---

## 7. Distribución de datos

### 7.1 Muestreo de batches (`src/main.cpp:134-141`)

```cpp
// Muestreo aleatorio uniforme
for (auto &i : idx) i = rand() % (n - cfg.T - 1);

// Construcción de secuencias input/target
for (int b = 0; b < BS; b++)
    for (int t = 0; t < cfg.T; t++) {
        h_tok[b*cfg.T+t] = data[idx[b]+t];     // Input
        h_tgt[b*cfg.T+t] = data[idx[b]+t+1];   // Target (shifted by 1)
    }
```

| Aspecto | Descripción |
|---------|-------------|
| Tipo de muestreo | Aleatorio uniforme con `rand()` |
| Overlapping | Sí, las secuencias pueden solaparse |
| Sin estratificación | No hay Agrupamiento por longitud o contenido |
| Semilla | `rand()` no tiene semilla fija (comportamiento indefinido entre ejecuciones) |

### 7.2 Distribución del corpus

- No se reporta estadísticas de distribución de tokens.
- No hay balanceo de clases (tarea de modelado de lenguaje).
- No hay split train/validación/test (todo se usa para entrenamiento).

---

## 8. Balanceo de clases

**No aplica**. Este es un modelo de modelado de lenguaje (next-token prediction). No hay clases que balancear; la distribución de tokens del corpus es directamente la distribución objetivo.

---

## 9. Augmentation de datos

**No implementado**. El pipeline de entrenamiento no aplica augmentation:
- No se usa back-translation.
- No se usa masking o corruption de tokens.
- No se usan variaciones de contexto.
- No se usa mixup ni cutmix.

---

## 10. Guardado y carga de modelos

### 10.1 Formato binario RBN1

**Guardado** (`src/model.cu:460-496`):

```cpp
void RubidiumTransformer::save(const char *path) {
    fwrite("RBN1", 1, 4, f);                    // Magic bytes
    fwrite(&cfg.V, sizeof(int), 1, f);           // Vocab size
    fwrite(&cfg.T, sizeof(int), 1, f);           // Block size
    fwrite(&cfg.D, sizeof(int), 1, f);           // D model
    fwrite(&cfg.H, sizeof(int), 1, f);           // N heads
    fwrite(&cfg.L, sizeof(int), 1, f);           // N layers
    fwrite(&cfg.FF, sizeof(int), 1, f);          // D FF
    fwrite(map, sizeof(int), 256, f);            // Char-to-ID map
    // ... todos los pesos (float32, little-endian)
}
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| Magic | 4 bytes | `RBN1` |
| V, T, D, H, L, FF | 6 × int32 | Configuración del modelo |
| char_to_id | 256 × int32 | Mapa de caracteres a IDs |
| Pesos | float32[] | Todos los pesos del modelo |

**Checkpoints** (`src/main.cpp:182-186`):

```cpp
if (step % 5000 == 0) {
    char path[256];
    sprintf(path, "checkpoints/model_step_%d.bin", step);
    model.save(path);
}
```

- Checkpoints cada **5,000 pasos**.
- Modelo final guardado como `model_final.bin`.

### 10.2 Conversión a Python Pickle

**Script**: `convert_to_pickle.py`

```bash
python convert_to_pickle.py model_final.bin model_final.pkl
```

| Propiedad | Descripción |
|-----------|-------------|
| Formato | Python pickle protocol 2 |
| Compatibilidad | Rust inference (`rubidium-core`), Python |
| Contenido | Todos los pesos + configuración + vocabulario |
| Pesos numpy | Formato `float32`, shapes preservados |

**Conversión alternativa en C++** (`src/convert_to_pickle.cpp`):
- Generador de pickle nativo sin dependencia de Python.
- Misma funcionalidad que el script Python.
- Opcodes de pickle escritos manualmente.

### 10.3 Estructura del modelo guardado

```
Dict {
    'vocab_size': int,
    'block_size': int,
    'd_model': int,
    'n_head': int,
    'n_layer': int,
    'd_ff': int,
    'char_to_id': dict,
    'id_to_char': dict,
    'token_emb': ndarray [V, D],
    'pos_emb': ndarray [1, T, D],
    'ln_f_w': ndarray [D],
    'ln_f_b': ndarray [D],
    'lm_w': ndarray [V, D],
    'lm_b': ndarray [V],
    'layers': [
        {
            'ln1_w', 'ln1_b': ndarray [D],
            'attn_wq_w', 'attn_wq_b': ndarray [D, D] / [D],
            'attn_wk_w', 'attn_wk_b': ndarray [D, D] / [D],
            'attn_wv_w', 'attn_wv_b': ndarray [D, D] / [D],
            'attn_wo_w', 'attn_wo_b': ndarray [D, D] / [D],
            'ln2_w', 'ln2_b': ndarray [D],
            'ff_w1_w', 'ff_w1_b': ndarray [FF, D] / [FF],
            'ff_w2_w', 'ff_w2_b': ndarray [D, FF] / [D],
        },
        ...  (L capas)
    ]
}
```

---

## 11. Versionado de datos y modelos

**No implementado**. No hay:
- Sistema de versionado de datasets (no se usa DVC, Delta Lake, etc.).
- Tags de versionado en checkpoints.
- Hashes o checksums de datos.
- Metadata de entrenamiento en los checkpoints.
- Registro de experimentos (no hay MLflow, W&B, etc.).

Los checkpoints se identifican únicamente por número de step:
```
checkpoints/model_step_5000.bin
checkpoints/model_step_10000.bin
...
model_final.bin
```

---

## 12. Utilidades y kernels CUDA

### 12.1 Kernels implementados (`src/cuda_kernels.cu`)

| Kernel | Función | Descripción |
|--------|---------|-------------|
| `layer_norm_fwd_kernel` | Forward | LayerNorm con shared memory |
| `layer_norm_bwd_kernel` | Backward | Gradientes de LayerNorm |
| `softmax_fwd_kernel` | Forward | Softmax numerically stable |
| `softmax_bwd_kernel` | Backward | Gradientes de softmax |
| `relu_fwd_kernel` | Forward | ReLU pointwise |
| `relu_bwd_kernel` | Backward | Gradientes de ReLU |
| `cross_entropy_fwd_kernel` | Forward | Cross-entropy con label smoothing implícito |
| `cross_entropy_bwd_kernel` | Backward | Gradientes de cross-entropy |
| `dropout_fwd_kernel` | Forward | Dropout con cuRAND Philox4 |
| `dropout_bwd_kernel` | Backward | Gradientes de dropout |
| `embedding_fwd_kernel` | Forward | Lookup de embeddings |
| `embedding_bwd_kernel` | Backward | Acumulación de gradientes embeddings |
| `adamw_step_kernel` | Optimizer | AdamW step con weight decay |
| `causal_mask_kernel` | Utility | Máscara causal (-1e9 en posiciones futuras) |
| `residual_add_kernel` | Utility | Suma residual |
| `scale_add_kernel` | Utility | Escalar y sumar |
| `copy_kernel` | Utility | Copia de memoria GPU |
| `zero_fill_kernel` | Utility | Rellenar con ceros |

### 12.2 Operaciones cuBLAS

| Operación | Descripción |
|-----------|-------------|
| `gemm_forward` | C = α * A * B + β * C (row-major via column-major API) |
| `gemm_backward_dA` | dA = dout * B^T |
| `gemm_backward_dB` | dB = A^T * dout |
| `linear_forward` | out = inp @ W^T + b |
| `linear_backward` | Backward completo de capa lineal |

### 12.3 Infraestructura GPU

| Componente | Detalle |
|------------|---------|
| Arquitectura CUDA | sm_60 (P100) |
| cuBLAS | Obligatorio |
| cuRAND | Para dropout |
| cuDNN | Opcional (no se usa activamente) |
| Arena allocator | 512 MB para autograd engine (`src/autograd.h`) |

---

## 13. Autograd Engine (`src/autograd.h`)

Motor de diferenciación automática reverse-mode implementado desde cero.

| Componente | Descripción |
|------------|-------------|
| Arena Allocator | Asignación de memoria lineal sin GC |
| Tensor | Estructura con puntero GPU + gradiente |
| Node | Nodo del grafo computacional |
| OpTypes | ADD, SUB, MUL, DIV, MATMUL, RELU, SOFTMAX, LAYER_NORM, etc. |
| Backward | Propagación reversa de gradientes |

**Nota**: El autograd engine está definido pero no se usa en el loop de entrenamiento principal. El entrenamiento usa backward passes manuales escritos directamente.

---

## 14. Generación de texto (`src/model.cu:403-455`)

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| Temperature | 0.7 | Controla aleatoriedad |
| Top-k | 40 | Sampling restringido a top-k tokens |
| Max chars | 120 | Longitud máxima de generación |
| Seed strings | "Hola", "Buenos dias", etc. | Seeds de prueba al final del entrenamiento |

---

## 15. Conteo de parámetros (`src/main.cpp:109-112`)

```cpp
long long tp = V*D*2 + T*D;  // Embeddings
for (int l = 0; l < L; l++)
    tp += 4*D*D + 8*D + 2*D*FF + 2*FF;  // Por capa
tp += 2*D + V*D + V;  // Final LN + LM head
```

| Componente | Fórmula | Ejemplo (V=256, D=2048, L=10, FF=8192) |
|------------|---------|----------------------------------------|
| Token embedding | V × D | 524,288 |
| Position embedding | T × D | 524,288 |
| Por capa (×10) | 4D² + 8D + 2D×FF + 2×FF | ~50.3M |
| Final LN | 2D | 4,096 |
| LM head | V × D + V | 524,544 |
| **Total** | | **~51.9M** |

---

## 16. Limitaciones identificadas

1. **Sin validación**: No hay split train/valid para monitorear overfitting.
2. **Sin logging estructurado**: Solo printf a stdout.
3. **Sin reproducibilidad**: `rand()` sin semilla fija, CUDA sin determinismo.
4. **Sin early stopping**: Entrena fijamente 200K steps.
5. **Sin augmentation**: Corpus sin transformaciones.
6. **Sin versionado**: Sin tracking de experimentos.
7. **Tokenizer básico**: Solo character-level, sin BPE/subword.
8. **ReLU en FFN**: No usa GELU como GPT-2/3/4.
9. **Backward simplificado**: La backward de atención es aproximada (`src/model.cu:297-308`).
10. **Memoria**: Checkpoints se guardan frecuentemente (cada 5K steps) sin compresión.
