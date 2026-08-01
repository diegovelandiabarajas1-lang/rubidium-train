// ============================================================
// RUBIDIUM TRANSFORMER - CUDA KERNELS IMPLEMENTATION (FP16)
// ============================================================
#include "cuda_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <curand_kernel.h>
#include <vector>

// ============================================================
// GLOBAL HANDLES
// ============================================================
cublasHandle_t cublas_handle = nullptr;
cublasLtHandle_t cublaslt_handle = nullptr;
cudnnHandle_t cudnn_handle = nullptr;
curandGenerator_t curand_gen = nullptr;

void init_handles() {
    if (!cublas_handle) { CB(cublasCreate(&cublas_handle)); }
    if (!cublaslt_handle) { CL(cublasLtCreate(&cublaslt_handle)); }
    if (!cudnn_handle) { CD(cudnnCreate(&cudnn_handle)); }
    if (!curand_gen) {
        CE(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
        CE(curandSetPseudoRandomGeneratorSeed(curand_gen, 42ULL));
    }
}

void destroy_handles() {
    if (cublas_handle) { cublasDestroy(cublas_handle); cublas_handle = nullptr; }
    if (cublaslt_handle) { cublasLtDestroy(cublaslt_handle); cublaslt_handle = nullptr; }
    if (cudnn_handle) { cudnnDestroy(cudnn_handle); cudnn_handle = nullptr; }
    if (curand_gen) { curandDestroyGenerator(curand_gen); curand_gen = nullptr; }
}

// Constant pointers for cuBLAS
float _zero_f = 0.0f;
float _one_f = 1.0f;
float _minus_one_f = -1.0f;

half _zero_h = __float2half(0.0f);
half _one_h = __float2half(1.0f);

// ============================================================
// cuBLASLt MATMUL DESCRIPTORS
// ============================================================
cublasLtMatmulDesc_t matmul_desc = nullptr;
cublasLtMatrixLayout_t A_desc = nullptr, B_desc = nullptr, C_desc = nullptr;
cublasLtMatmulAlgo_t best_algo;

void cublaslt_matmul_init() {
    if (matmul_desc) return;
    CL(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_16F));
    CL(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &CUBLAS_OP_N, sizeof(int)));
    CL(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &CUBLAS_OP_T, sizeof(int)));
}

void cublaslt_autotune(cudaStream_t stream, int M, int N, int K) {
    // Test a few algos and pick the fastest
    int algo_count = 0;
    cublasLtMatmulHeuristicResult_t heuristic;
    CL(cublasLtMatmulAlgoGetHeuristics(cublaslt_handle, 1, matmul_desc, A_desc, B_desc, C_desc, C_desc, &heuristic));
    best_algo = heuristic.algo;
}

// ============================================================
// MEMPOOL IMPLEMENTATION (3GB)
// ============================================================
MemPool g_mem_pool;

void MemPool::init(size_t bytes, cudaStream_t s) {
    if (ptr) free();
    capacity = bytes;
    stream = s;
    offset = 0;
    CE(cudaMalloc(&ptr, bytes));
}

void* MemPool::alloc(size_t bytes, size_t alignment) {
    size_t aligned_offset = (offset + alignment - 1) & ~(alignment - 1);
    if (aligned_offset + bytes > capacity) {
        fprintf(stderr, "MemPool OOM: need %zu, have %zu free\n", bytes, capacity - aligned_offset);
        exit(1);
    }
    void* result = (char*)ptr + aligned_offset;
    offset = aligned_offset + bytes;
    return result;
}

void MemPool::free() {
    if (ptr) { cudaFree(ptr); ptr = nullptr; }
    capacity = 0;
    offset = 0;
}

// ============================================================
// LOSS SCALER IMPLEMENTATION
// ============================================================
LossScaler g_loss_scaler;

bool LossScaler::update(bool overflow) {
    if (overflow) {
        unskipped_steps = 0;
        scale *= backoff_factor;
        if (scale < 1.0f) scale = 1.0f;
        return false;
    }
    unskipped_steps++;
    if (unskipped_steps >= growth_interval) {
        scale *= growth_factor;
        if (scale > 65536.0f) scale = 65536.0f;
        unskipped_steps = 0;
    }
    return true;
}

// ============================================================
// RANDOM BUFFERS IMPLEMENTATION
// ============================================================
RandomBuffers g_random_buffers;

void RandomBuffers::init(int layers, int max_tokens_per_layer, int window) {
    num_layers = layers;
    max_tokens = max_tokens_per_layer;
    window_size = window;
    current_step = 0;

    CE(cudaMalloc(&dropout_masks, layers * sizeof(uint8_t*)));
    CE(cudaMalloc(&normal_noise, layers * sizeof(float*)));
    for (int i = 0; i < layers; i++) {
        CE(cudaMalloc(&dropout_masks[i], window * max_tokens_per_layer * sizeof(uint8_t)));
        CE(cudaMalloc(&normal_noise[i], window * max_tokens_per_layer * sizeof(float)));
    }

    curandGenerator_t gen;
    CE(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CE(curandSetPseudoRandomGeneratorSeed(gen, 42ULL));
    for (int i = 0; i < layers; i++) {
        // Generate dropout masks
        curandStatus_t s = curandGenerateUniform(gen, (float*)dropout_masks[i], 
            (size_t)window * max_tokens_per_layer * sizeof(uint8_t) / sizeof(float));
        // Generate normal noise
        s = curandGenerateNormal(gen, normal_noise[i], 
            (size_t)window * max_tokens_per_layer, 0.0f, 1.0f);
    }
    CE(curandDestroyGenerator(gen));
}

uint8_t* RandomBuffers::get_dropout_mask(int layer_idx) {
    int offset = current_step * max_tokens;
    return dropout_masks[layer_idx] + offset;
}

float* RandomBuffers::get_normal_noise(int layer_idx) {
    int offset = current_step * max_tokens;
    return normal_noise[layer_idx] + offset;
}

void RandomBuffers::advance_step() {
    current_step++;
    if (current_step >= window_size) current_step = 0;
}

void RandomBuffers::free() {
    if (dropout_masks) {
        for (int i = 0; i < num_layers; i++)
            if (dropout_masks[i]) cudaFree(dropout_masks[i]);
        cudaFree(dropout_masks);
    }
    if (normal_noise) {
        for (int i = 0; i < num_layers; i++)
            if (normal_noise[i]) cudaFree(normal_noise[i]);
        cudaFree(normal_noise);
    }
    dropout_masks = nullptr;
    normal_noise = nullptr;
}

// ============================================================
// FP16 LAYER NORM FORWARD
// ============================================================
__global__ void layer_norm_fp16_fwd_kernel(half* out, float* mean, float* inv_std,
                                            const half* x, const half* w, const half* b,
                                            int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float sh[];
    float* s_sum = sh;
    float* s_sqsum = sh + blockDim.x;

    const half* xr = x + row * D;
    half* out_ptr = out + row * D;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        local_sum += __half2float(xr[i]);
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) s_sum[threadIdx.x] += s_sum[threadIdx.x + h];
        __syncthreads();
    }
    float m = s_sum[0] / D;
    if (threadIdx.x == 0) mean[row] = m;
    __syncthreads();

    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float d = __half2float(xr[i]) - m;
        local_sq += d * d;
    }
    s_sqsum[threadIdx.x] = local_sq;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) s_sqsum[threadIdx.x] += s_sqsum[threadIdx.x + h];
        __syncthreads();
    }
    float inv = rsqrtf(s_sqsum[0] / D + eps);
    if (threadIdx.x == 0) inv_std[row] = inv;
    __syncthreads();

    for (int i = threadIdx.x; i < D; i += blockDim.x)
        out_ptr[i] = __float2half(__half2float(w[i]) * (__half2float(xr[i]) - m) * inv + __half2float(b[i]));
}

void layer_norm_fp16_forward(half* out, float* mean, float* inv_std,
                              const half* x, const half* w, const half* b,
                              int N, int D, float eps) {
    int threads = 256;
    if (D < 256) threads = D;
    size_t shmem = 2 * threads * sizeof(float);
    layer_norm_fp16_fwd_kernel<<<N, threads, shmem>>>(out, mean, inv_std, x, w, b, N, D, eps);
}

// ============================================================
// FP16 LAYER NORM BACKWARD
// ============================================================
__global__ void layer_norm_fp16_bwd_kernel(half* dx, half* dw, half* db,
                                            const half* dout, const half* x, const half* w,
                                            const float* mean, const float* inv_std, int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float sh[];
    float* s1 = sh;
    float* s2 = sh + blockDim.x;

    const half* dxr = dout + row * D;
    const half* xr = x + row * D;
    half* dxor = dx + row * D;
    float m = mean[row];
    float inv = inv_std[row];

    float local_s1 = 0.0f, local_s2 = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float dw_i = __half2float(dxr[i]) * __half2float(w[i]);
        local_s1 += dw_i;
        local_s2 += dw_i * (__half2float(xr[i]) - m);
    }
    s1[threadIdx.x] = local_s1;
    s2[threadIdx.x] = local_s2;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) {
            s1[threadIdx.x] += s1[threadIdx.x + h];
            s2[threadIdx.x] += s2[threadIdx.x + h];
        }
        __syncthreads();
    }
    float ts1 = s1[0];
    float ts2 = s2[0];
    __syncthreads();

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float dw_i = __half2float(dxr[i]) * __half2float(w[i]);
        dxor[i] = __float2half(inv * (dw_i - (ts1 + (__half2float(xr[i]) - m) * inv * ts2) / D));
    }

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        atomicAdd(&dw[i], __float2half(__half2float(dxr[i]) * (__half2float(xr[i]) - m) * inv));
        atomicAdd(&db[i], dxr[i]);
    }
}

void layer_norm_fp16_backward(half* dx, half* dw, half* db,
                               const half* dout, const half* x, const half* w,
                               const float* mean, const float* inv_std, int N, int D) {
    int threads = 256;
    if (D < 256) threads = D;
    size_t shmem = 2 * threads * sizeof(float);
    layer_norm_fp16_bwd_kernel<<<N, threads, shmem>>>(dx, dw, db, dout, x, w, mean, inv_std, N, D);
}

// ============================================================
// FP16 RELU FORWARD
// ============================================================
__global__ void relu_fp16_fwd_kernel(half* out, const half* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        half val = x[i];
        out[i] = __hgt(val, __float2half(0.0f)) ? val : __float2half(0.0f);
    }
}

void relu_fp16_forward(half* out, const half* x, int N) {
    relu_fp16_fwd_kernel<<<(N + 255) / 256, 256>>>(out, x, N);
}

// ============================================================
// FP16 RELU BACKWARD
// ============================================================
__global__ void relu_fp16_bwd_kernel(half* dx, const half* dout, const half* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        dx[i] = __hgt(x[i], __float2half(0.0f)) ? dout[i] : __float2half(0.0f);
}

void relu_fp16_backward(half* dx, const half* dout, const half* x, int N) {
    relu_fp16_bwd_kernel<<<(N + 255) / 256, 256>>>(dx, dout, x, N);
}

// ============================================================
// FP16 EMBEDDING FORWARD
// ============================================================
__global__ void embedding_fp16_fwd_kernel(half* out, const half* weight,
                                           const int* indices, int B, int T, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * D;
    if (idx >= total) return;
    int b = idx / (T * D);
    int t = (idx / D) % T;
    int d = idx % D;
    int token = indices[b * T + t];
    out[idx] = weight[token * D + d];
}

void embedding_fp16_forward(half* out, const half* weight, const int* indices, int B, int T, int D) {
    int total = B * T * D;
    embedding_fp16_fwd_kernel<<<(total + 255) / 256, 256>>>(out, weight, indices, B, T, D);
}

// ============================================================
// FP16 EMBEDDING BACKWARD
// ============================================================
__global__ void embedding_fp16_bwd_kernel(float* d_weight, const float* d_out,
                                           const int* indices, int B, int T, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * D;
    if (idx >= total) return;
    int b = idx / (T * D);
    int t = (idx / D) % T;
    int d = idx % D;
    int token = indices[b * T + t];
    atomicAdd(&d_weight[token * D + d], d_out[idx]);
}

void embedding_fp16_backward(half* d_weight, const float* d_out,
                               const int* indices, int B, int T, int D, int V) {
    int total = B * T * D;
    embedding_fp16_bwd_kernel<<<(total + 255) / 256, 256>>>((float*)d_weight, d_out, indices, B, T, D);
}

// ============================================================
// FP16 RESIDUAL ADD
// ============================================================
__global__ void residual_add_fp16_kernel(half* out, const half* a, const half* b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = __hadd(a[i], b[i]);
}

void residual_add_fp16(half* out, const half* a, const half* b, int N) {
    residual_add_fp16_kernel<<<(N + 255) / 256, 256>>>(out, a, b, N);
}

// ============================================================
// FP16 SCALE ADD
// ============================================================
__global__ void scale_add_fp16_kernel(half* dst, const half* src, float alpha, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = __hadd(dst[i], __float2half(alpha * __half2float(src[i])));
}

void scale_add_fp16(half* dst, const half* src, float alpha, int N) {
    scale_add_fp16_kernel<<<(N + 255) / 256, 256>>>(dst, src, alpha, N);
}

// ============================================================
// FP16 SOFTMAX FORWARD
// ============================================================
__global__ void softmax_fp16_fwd_kernel(half* out, const half* x, int N, int C) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float sh[];
    float* s_max = sh;
    float* s_sum = sh + blockDim.x;

    const half* xr = x + row * C;
    half* out_ptr = out + row * C;

    float local_max = -1e30f;
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        local_max = fmaxf(local_max, __half2float(xr[i]));
    s_max[threadIdx.x] = local_max;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + h]);
        __syncthreads();
    }
    float mx = s_max[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        out_ptr[i] = __float2half(expf(__half2float(xr[i]) - mx));
        local_sum += __half2float(out_ptr[i]);
    }
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) s_sum[threadIdx.x] += s_sum[threadIdx.x + h];
        __syncthreads();
    }
    float s = s_sum[0];
    __syncthreads();

    for (int i = threadIdx.x; i < C; i += blockDim.x)
        out_ptr[i] = __float2half(__half2float(out_ptr[i]) / s);
}

void softmax_fp16_forward(half* out, const half* x, int N, int C) {
    int threads = 256;
    if (C < 256) threads = C;
    size_t shmem = 2 * threads * sizeof(float);
    softmax_fp16_fwd_kernel<<<N, threads, shmem>>>(out, x, N, C);
}

// ============================================================
// FP16 SOFTMAX BACKWARD
// ============================================================
__global__ void softmax_fp16_bwd_kernel(half* dx, const half* dout, const half* out, int N, int C) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float sh[];

    const half* doutr = dout + row * C;
    const half* outr = out + row * C;
    half* dxr = dx + row * C;

    float local_dot = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        local_dot += __half2float(doutr[i]) * __half2float(outr[i]);
    sh[threadIdx.x] = local_dot;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) sh[threadIdx.x] += sh[threadIdx.x + h];
        __syncthreads();
    }
    float dot = sh[0];
    __syncthreads();

    for (int i = threadIdx.x; i < C; i += blockDim.x)
        dxr[i] = __float2half(__half2float(outr[i]) * (__half2float(doutr[i]) - dot));
}

void softmax_fp16_backward(half* dx, const half* dout, const half* out, int N, int C) {
    int threads = 256;
    if (C < 256) threads = C;
    size_t shmem = threads * sizeof(float);
    softmax_fp16_bwd_kernel<<<N, threads, shmem>>>(dx, dout, out, N, C);
}

// ============================================================
// FP32 CROSS ENTROPY (for stability)
// ============================================================
__global__ void cross_entropy_fp32_fwd_kernel(float* out, const float* logits,
                                               const int* targets, int N, int V) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float sh[];
    float* s_max = sh;
    float* s_sum = sh + blockDim.x;

    const float* lr = logits + row * V;
    int tgt = targets[row];

    float local_max = -1e30f;
    for (int i = threadIdx.x; i < V; i += blockDim.x)
        local_max = fmaxf(local_max, lr[i]);
    s_max[threadIdx.x] = local_max;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + h]);
        __syncthreads();
    }
    float mx = s_max[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < V; i += blockDim.x)
        local_sum += expf(lr[i] - mx);
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) s_sum[threadIdx.x] += s_sum[threadIdx.x + h];
        __syncthreads();
    }
    float s = s_sum[0];
    __syncthreads();

    float loss = -logf(expf(lr[tgt] - mx) / s + 1e-10f);
    if (threadIdx.x == 0) atomicAdd(out, loss);
}

float cross_entropy_fp32_forward(const float* logits, const int* targets, int N, int V) {
    float h_loss = 0.0f;
    float* d_loss;
    CE(cudaMalloc(&d_loss, sizeof(float)));
    CE(cudaMemset(d_loss, 0, sizeof(float)));

    int threads = 256;
    if (V < 256) threads = V;
    size_t shmem = 2 * threads * sizeof(float);
    cross_entropy_fp32_fwd_kernel<<<N, threads, shmem>>>(d_loss, logits, targets, N, V);

    CE(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CE(cudaFree(d_loss));
    return h_loss / N;
}

// ============================================================
// FP32 CROSS ENTROPY BACKWARD
// ============================================================
__global__ void cross_entropy_fp32_bwd_kernel(float* d_logits, const float* logits,
                                               const int* targets, int N, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * V;
    if (idx >= total) return;

    int row = idx / V;
    int col = idx % V;
    int tgt = targets[row];

    const float* lr = logits + row * V;
    float mx = -1e30f;
    for (int j = 0; j < V; j++) mx = fmaxf(mx, lr[j]);
    float s = 0.0f;
    for (int j = 0; j < V; j++) s += expf(lr[j] - mx);
    float prob = expf(lr[col] - mx) / s;

    d_logits[idx] = (prob - (col == tgt ? 1.0f : 0.0f)) / (float)N;
}

void cross_entropy_fp32_backward(float* d_logits, const float* logits,
                                  const int* targets, int N, int V) {
    int total = N * V;
    cross_entropy_fp32_bwd_kernel<<<(total + 255) / 256, 256>>>(d_logits, logits, targets, N, V);
}

// ============================================================
// FP16 DROPOUT FORWARD
// ============================================================
__global__ void dropout_fp16_fwd_kernel(half* out, const half* x, uint8_t* mask,
                                         int N, float p, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        curandStatePhilox4_32_10_t state;
        curand_init(42, i, 0, &state);
        float r = curand_uniform(&state);
        mask[i] = (r < p) ? 0 : 1;
        out[i] = mask[i] ? __float2half(scale * __half2float(x[i])) : __float2half(0.0f);
    }
}

void dropout_fp16_forward(half* out, const half* x, uint8_t* mask, int N, float p) {
    float scale = 1.0f / (1.0f - p);
    dropout_fp16_fwd_kernel<<<(N + 255) / 256, 256>>>(out, x, mask, N, p, scale);
}

// ============================================================
// FP16 DROPOUT BACKWARD
// ============================================================
__global__ void dropout_fp16_bwd_kernel(half* dx, const half* dout, const uint8_t* mask,
                                         int N, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dx[i] = mask[i] ? __float2half(scale * __half2float(dout[i])) : __float2half(0.0f);
}

void dropout_fp16_backward(half* dx, const half* dout, const uint8_t* mask, int N, float p) {
    float scale = 1.0f / (1.0f - p);
    dropout_fp16_bwd_kernel<<<(N + 255) / 256, 256>>>(dx, dout, mask, N, scale);
}

// ============================================================
// GEMM FP16 TENSOR CORES (cuBLASLt)
// ============================================================
void gemm_fp16_tensorcores(half* C, const half* A, const half* B,
                            int M, int N, int K, float alpha, float beta,
                            bool transA, bool transB) {
    cublasLtOperation_t opA = transA ? CUBLASLT_OPERATION_TRANSPOSE : CUBLASLT_OPERATION_NONE;
    cublasLtOperation_t opB = transB ? CUBLASLT_OPERATION_TRANSPOSE : CUBLASLT_OPERATION_NONE;

    CL(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CL(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    int lda = transA ? M : K;
    int ldb = transB ? K : N;
    int ldc = N;

    // For row-major: C[M,N] = A[M,K] * B[K,N]
    // cuBLAS uses col-major, so we flip A and B
    CL(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_16F, K, M, lda));
    CL(cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_16F, N, K, ldb));
    CL(cublasLtMatrixLayoutCreate(&C_desc, CUDA_R_16F, N, M, ldc));

    CL(cublasLtMatmul(cublaslt_handle, matmul_desc,
                       &alpha, B, B_desc, A, A_desc, &beta, C, C_desc,
                       C, C_desc, &best_algo, nullptr, 0));

    cublasLtMatrixLayoutDestroy(A_desc);
    cublasLtMatrixLayoutDestroy(B_desc);
    cublasLtMatrixLayoutDestroy(C_desc);
}

// FP16 GEMM with FP32 accumulation (for backward)
void gemm_fp16_accum_fp32(float* C, const half* A, const half* B,
                           int M, int N, int K, float alpha, float beta) {
    CB(cublasGemmEx(cublas_handle,
                     CUBLAS_OP_N, CUBLAS_OP_T,
                     N, M, K,
                     &alpha,
                     B, CUDA_R_16F, N,
                     A, CUDA_R_16F, K,
                     &beta,
                     C, CUDA_R_32F, N,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void gemm_fp16_accum_fp32_backward_dA(float* dA, const float* dout, const half* B,
                                       int M, int N, int K, float alpha, float beta) {
    CB(cublasGemmEx(cublas_handle,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     K, M, N,
                     &alpha,
                     B, CUDA_R_16F, N,
                     dout, CUDA_R_32F, N,
                     &beta,
                     dA, CUDA_R_32F, K,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void gemm_fp16_accum_fp32_backward_dB(half* dB, const half* A, const float* dout,
                                       int M, int N, int K, float alpha, float beta) {
    CB(cublasGemmEx(cublas_handle,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     N, K, M,
                     &alpha,
                     dout, CUDA_R_32F, N,
                     A, CUDA_R_16F, K,
                     &beta,
                     dB, CUDA_R_16F, N,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ============================================================
// FUSED FFN FORWARD: x -> w1 -> relu -> w2
// ============================================================
__global__ void fused_ffn_fwd_kernel(half* out, half* relu_mask,
                                      const half* x, const half* w1, const half* b1,
                                      const half* w2, const half* b2,
                                      int M, int D, int FF) {
    int row = blockIdx.x;
    if (row >= M) return;

    const half* xr = x + row * D;
    half* outr = out + row * D;
    half* relu_r = relu_mask + row * FF;

    for (int j = threadIdx.x; j < FF; j += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < D; k++)
            sum += __half2float(xr[k]) * __half2float(w1[j * D + k]);
        sum += __half2float(b1[j]);
        float activated = fmaxf(0.0f, sum);
        relu_r[j] = __float2half(activated > 0.0f ? 1.0f : 0.0f);

        float sum2 = activated * __half2float(w2[j]); // simplified
        for (int d = 0; d < D; d++)
            sum2 += activated * __half2float(w2[j * D + d]);
        outr[j] = __float2half(sum2);
    }
}

void fused_ffn_forward(half* out, half* relu_mask,
                        const half* x, const half* w1, const half* b1,
                        const half* w2, const half* b2,
                        int M, int D, int FF) {
    fused_ffn_fwd_kernel<<<M, 256>>>(out, relu_mask, x, w1, b1, w2, b2, M, D, FF);
}

// ============================================================
// FUSED FFN BACKWARD
// ============================================================
__global__ void fused_ffn_bwd_kernel(half* dx, half* dw1, half* db1, half* dw2, half* db2,
                                      const half* dout, const half* x, const half* fi,
                                      const half* w1, const half* w2, const half* relu_mask,
                                      int M, int D, int FF) {
    int row = blockIdx.x;
    if (row >= M) return;

    const half* doutr = dout + row * D;
    const half* xr = x + row * D;
    half* dxr = dx + row * D;
    const half* relu_r = relu_mask + row * FF;

    for (int j = threadIdx.x; j < FF; j += blockDim.x) {
        float mask_val = __half2float(relu_r[j]);
        float dout_val = __half2float(doutr[j]);
        float grad_ffn = mask_val * dout_val;

        for (int d = 0; d < D; d++) {
            atomicAdd(&dw1[j * D + d], __float2half(grad_ffn * __half2float(xr[d])));
            atomicAdd(&dxr[d], __float2half(grad_ffn * __half2float(w1[j * D + d])));
        }
        atomicAdd(&db1[j], __float2half(grad_ffn));
    }
}

void fused_ffn_backward(half* dx, half* dw1, half* db1, half* dw2, half* db2,
                         const half* dout, const half* x, const half* fi,
                         const half* w1, const half* w2, const half* relu_mask,
                         int M, int D, int FF) {
    fused_ffn_bwd_kernel<<<M, 256>>>(dx, dw1, db1, dw2, db2, dout, x, fi, w1, w2, relu_mask, M, D, FF);
}

// ============================================================
// FUSED QKV PROJECTION
// ============================================================
__global__ void fused_qkv_proj_kernel(half* q, half* k, half* v,
                                       const half* x,
                                       const half* wq, const half* bq,
                                       const half* wk, const half* bk,
                                       const half* wv, const half* bv,
                                       int M, int D, int H, int hd) {
    int row = blockIdx.x;
    if (row >= M) return;

    const half* xr = x + row * D;

    for (int h = 0; h < H; h++) {
        half* qr = q + row * H * hd + h * hd;
        half* kr = k + row * H * hd + h * hd;
        half* vr = v + row * H * hd + h * hd;
        const half* wqr = wq + h * hd * D;
        const half* wkr = wk + h * hd * D;
        const half* wvr = wv + h * hd * D;

        for (int j = threadIdx.x; j < hd; j += blockDim.x) {
            float sq = __half2float(bq[h * hd + j]);
            float sk = __half2float(bk[h * hd + j]);
            float sv = __half2float(bv[h * hd + j]);
            for (int d = 0; d < D; d++) {
                float xd = __half2float(xr[d]);
                sq += xd * __half2float(wqr[j * D + d]);
                sk += xd * __half2float(wkr[j * D + d]);
                sv += xd * __half2float(wvr[j * D + d]);
            }
            qr[j] = __float2half(sq);
            kr[j] = __float2half(sk);
            vr[j] = __float2half(sv);
        }
    }
}

void fused_qkv_projection(half* q, half* k, half* v,
                            const half* x,
                            const half* wq, const half* bq,
                            const half* wk, const half* bk,
                            const half* wv, const half* bv,
                            int M, int D, int H, int hd) {
    fused_qkv_proj_kernel<<<M, 256>>>(q, k, v, x, wq, bq, wk, bk, wv, bv, M, D, H, hd);
}

// ============================================================
// FUSED ATTENTION SCORES + SOFTMAX
// ============================================================
__global__ void fused_attn_scores_softmax_kernel(half* attn_out, float* attn_fp32,
                                                   const half* q, const half* k,
                                                   int B, int H, int T, int hd, float scale) {
    int bh = blockIdx.z;
    int t1 = blockIdx.y;
    int t2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * H || t1 >= T || t2 >= T) return;

    int b = bh / H;
    int h = bh % H;

    const half* qr = q + (b * T + t1) * H * hd + h * hd;
    const half* kr = k + (b * T + t2) * H * hd + h * hd;

    float dot = 0.0f;
    for (int d = 0; d < hd; d++)
        dot += __half2float(qr[d]) * __half2float(kr[d]);
    dot *= scale;

    int idx = b * H * T * T + h * T * T + t1 * T + t2;
    attn_fp32[idx] = dot;
}

void fused_attn_scores_softmax(half* attn_out, float* attn_fp32,
                                const half* q, const half* k,
                                int B, int H, int T, int hd, float scale) {
    dim3 grid((T + 255) / 256, T, B * H);
    fused_attn_scores_softmax_kernel<<<grid, 256>>>(attn_out, attn_fp32, q, k, B, H, T, hd, scale);
}

// ============================================================
// FUSED LN + RESIDUAL + DROPOUT
// ============================================================
__global__ void fused_ln_residual_dropout_fwd_kernel(half* out, float* mean, float* inv_std,
                                                       uint8_t* mask,
                                                       const half* x, const half* residual,
                                                       const half* w, const half* b,
                                                       int N, int D, float p, float eps,
                                                       uint64_t seed, int step) {
    int row = blockIdx.x;
    if (row >= N) return;

    extern __shared__ float sh[];
    float* s_sum = sh;
    float* s_sqsum = sh + blockDim.x;

    const half* xr = x + row * D;
    half* outr = out + row * D;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        local_sum += __half2float(xr[i]) + __half2float(residual[i]);
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();
    for (int h2 = blockDim.x / 2; h2 > 0; h2 >>= 1) {
        if (threadIdx.x < h2) s_sum[threadIdx.x] += s_sum[threadIdx.x + h2];
        __syncthreads();
    }
    float m = s_sum[0] / D;
    if (threadIdx.x == 0) mean[row] = m;
    __syncthreads();

    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float d = (__half2float(xr[i]) + __half2float(residual[i])) - m;
        local_sq += d * d;
    }
    s_sqsum[threadIdx.x] = local_sq;
    __syncthreads();
    for (int h2 = blockDim.x / 2; h2 > 0; h2 >>= 1) {
        if (threadIdx.x < h2) s_sqsum[threadIdx.x] += s_sqsum[threadIdx.x + h2];
        __syncthreads();
    }
    float inv = rsqrtf(s_sqsum[0] / D + eps);
    if (threadIdx.x == 0) inv_std[row] = inv;
    __syncthreads();

    float scale_val = 1.0f / (1.0f - p);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float normalized = ((__half2float(xr[i]) + __half2float(residual[i])) - m) * inv;
        float val = __half2float(w[i]) * normalized + __half2float(b[i]);

        curandStatePhilox4_32_10_t state;
        curand_init(seed, row * D + i, step, &state);
        float r = curand_uniform(&state);
        uint8_t keep = (r >= p) ? 1 : 0;
        mask[row * D + i] = keep;
        outr[i] = __float2half(keep ? val * scale_val : 0.0f);
    }
}

void fused_ln_residual_dropout_forward(half* out, float* mean, float* inv_std, uint8_t* mask,
                                        const half* x, const half* residual,
                                        const half* w, const half* b,
                                        int N, int D, float p, float eps, uint64_t seed, int step) {
    int threads = 256;
    if (D < 256) threads = D;
    size_t shmem = 2 * threads * sizeof(float);
    fused_ln_residual_dropout_fwd_kernel<<<N, threads, shmem>>>(out, mean, inv_std, mask, x, residual, w, b, N, D, p, eps, seed, step);
}

// ============================================================
// FP32 SOFTMAX FORWARD (for attention stability)
// ============================================================
__global__ void softmax_fp32_fwd_kernel(float* out, const float* x, int N, int C) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float sh[];
    float* s_max = sh;
    float* s_sum = sh + blockDim.x;

    const float* xr = x + row * C;
    float* out_ptr = out + row * C;

    float local_max = -1e30f;
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        local_max = fmaxf(local_max, xr[i]);
    s_max[threadIdx.x] = local_max;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + h]);
        __syncthreads();
    }
    float mx = s_max[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        out_ptr[i] = expf(xr[i] - mx);
        local_sum += out_ptr[i];
    }
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) s_sum[threadIdx.x] += s_sum[threadIdx.x + h];
        __syncthreads();
    }
    float s = s_sum[0];
    __syncthreads();

    for (int i = threadIdx.x; i < C; i += blockDim.x)
        out_ptr[i] /= s;
}

void softmax_fp32_forward(float* out, const float* x, int N, int C) {
    int threads = 256;
    if (C < 256) threads = C;
    size_t shmem = 2 * threads * sizeof(float);
    softmax_fp32_fwd_kernel<<<N, threads, shmem>>>(out, x, N, C);
}

// ============================================================
// FP32 SOFTMAX BACKWARD
// ============================================================
__global__ void softmax_fp32_bwd_kernel(float* dx, const float* dout, const float* out,
                                         int N, int C) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float sh[];

    const float* doutr = dout + row * C;
    const float* outr = out + row * C;
    float* dxr = dx + row * C;

    float local_dot = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        local_dot += doutr[i] * outr[i];
    sh[threadIdx.x] = local_dot;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) sh[threadIdx.x] += sh[threadIdx.x + h];
        __syncthreads();
    }
    float dot = sh[0];
    __syncthreads();

    for (int i = threadIdx.x; i < C; i += blockDim.x)
        dxr[i] = outr[i] * (doutr[i] - dot);
}

void softmax_fp32_backward(float* dx, const float* dout, const float* out, int N, int C) {
    int threads = 256;
    if (C < 256) threads = C;
    size_t shmem = threads * sizeof(float);
    softmax_fp32_bwd_kernel<<<N, threads, shmem>>>(dx, dout, out, N, C);
}

// ============================================================
// CAUSAL MASK (FP32)
// ============================================================
__global__ void causal_mask_fp32_kernel(float* att, int B, int H, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * T * T;
    if (idx >= total) return;
    int t1 = (idx / T) % T;
    int t2 = idx % T;
    if (t2 > t1) att[idx] = -1e9f;
}

void apply_causal_mask_fp32(float* att, int B, int H, int T) {
    int total = B * H * T * T;
    causal_mask_fp32_kernel<<<(total + 255) / 256, 256>>>(att, B, H, T);
}

// ============================================================
// CONVERT FP32 <-> FP16
// ============================================================
void convert_fp32_to_fp16(half* dst, const float* src, int N) {
    for (int i = 0; i < N; i++)
        dst[i] = __float2half(src[i]);
}

void convert_fp16_to_fp32(float* dst, const half* src, int N) {
    for (int i = 0; i < N; i++)
        dst[i] = __half2float(src[i]);
}

// ============================================================
// FP16 LINEAR LAYER HELPERS
// ============================================================
void linear_forward_fp16(half* out, const half* inp, const half* W, const half* b,
                          int M, int Di, int Do) {
    // out = inp @ W^T + b
    gemm_fp16_tensorcores(out, inp, W, M, Do, Di);
    if (b) {
        for (int i = 0; i < M; i++)
            scale_add_fp16(out + i * Do, b, 1.0f, Do);
    }
}

void linear_forward_fp16_to_fp32(float* out, const half* inp, const half* W, const half* b,
                                  int M, int Di, int Do) {
    gemm_fp16_accum_fp32(out, inp, W, M, Do, Di);
    if (b) {
        for (int i = 0; i < M; i++) {
            float* outr = out + i * Do;
            for (int j = 0; j < Do; j++)
                outr[j] += __half2float(b[j]);
        }
    }
}

void linear_backward_fp16(half* dinp, half* dW, half* db,
                            const half* dout, const half* inp, const half* W,
                            int M, int Di, int Do) {
    gemm_fp16_accum_fp32_backward_dA((float*)dinp, (const float*)dout, W, M, Do, Di);
    gemm_fp16_accum_fp32_backward_dB(dW, inp, (const float*)dout, M, Do, Di);
    if (db) {
        half zero_h = __float2half(0.0f);
        for (int j = 0; j < Do; j++) {
            float sum = 0.0f;
            for (int i = 0; i < M; i++)
                sum += __half2float(dout[i * Do + j]);
            db[j] = __float2half(sum);
        }
    }
}

// ============================================================
// OVERFLOW DETECTION
// ============================================================
__global__ void check_overflow_kernel(const float* grads, int n, int* found) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grads[i];
        if (isnan(g) || isinf(g)) {
            *found = 1;
        }
    }
}

bool check_overflow_all(float** gradients, int num_arrays, int* sizes) {
    int* d_found;
    int h_found = 0;
    CE(cudaMalloc(&d_found, sizeof(int)));
    CE(cudaMemset(d_found, 0, sizeof(int)));

    for (int a = 0; a < num_arrays; a++) {
        int total = sizes[a];
        check_overflow_kernel<<<(total + 255) / 256, 256>>>(gradients[a], total, d_found);
    }

    CE(cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost));
    CE(cudaFree(d_found));
    return h_found != 0;
}
