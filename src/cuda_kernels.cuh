// ============================================================
// RUBIDIUM TRANSFORMER - CUDA KERNELS HEADER (FP16 Optimized)
// ============================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <curand_kernel.h>
#include <cudnn.h>

// Error checking macros
#define CE(e) do { cudaError_t err = (e); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); } } while(0)
#define CB(s) do { cublasStatus_t err = (s); if (err != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, err); exit(1); } } while(0)
#define CD(s) do { cudnnStatus_t err = (s); if (err != CUDNN_STATUS_SUCCESS) { \
    fprintf(stderr, "cuDNN error %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(err)); exit(1); } } while(0)
#define CL(s) do { cublasStatus_t err = (s); if (err != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLASLt error %s:%d: %d\n", __FILE__, __LINE__, err); exit(1); } } while(0)

// ============================================================
// GLOBAL HANDLES
// ============================================================
extern cublasHandle_t cublas_handle;
extern cublasLtHandle_t cublaslt_handle;
extern cudnnHandle_t cudnn_handle;
extern curandGenerator_t curand_gen;

void init_handles();
void destroy_handles();

// ============================================================
// cuBLASLt MATMUL DESCRIPTORS
// ============================================================
extern cublasLtMatmulDesc_t matmul_desc;
extern cublasLtMatrixLayout_t A_desc, B_desc, C_desc;
extern cublasLtMatmulAlgo_t best_algo;

void cublaslt_matmul_init();
void cublaslt_autotune(cudaStream_t stream, int M, int N, int K);

// ============================================================
// MEMORY POOL (3GB)
// ============================================================
struct MemPool {
    void* ptr = nullptr;
    size_t capacity = 0;
    size_t offset = 0;
    cudaStream_t stream = 0;
    
    void init(size_t bytes, cudaStream_t s = 0);
    void* alloc(size_t bytes, size_t alignment = 256);
    void reset() { offset = 0; }
    void free();
};
extern MemPool g_mem_pool;

// ============================================================
// LOSS SCALER (Dynamic)
// ============================================================
struct LossScaler {
    float scale = 32768.0f;
    float growth_factor = 2.0f;
    float backoff_factor = 0.5f;
    int growth_interval = 2000;
    int unskipped_steps = 0;
    
    float get_scale() const { return scale; }
    bool update(bool overflow);
    bool is_valid_scale() const { return scale >= 1.0f && scale <= 65536.0f; }
};
extern LossScaler g_loss_scaler;

// ============================================================
// RANDOM BUFFERS (GPU curand, ventana 10K steps)
// ============================================================
struct RandomBuffers {
    curandGenerator_t gen = nullptr;
    uint8_t** dropout_masks = nullptr;   // [layers][window * max_tokens]
    float** normal_noise = nullptr;      // [layers][window * max_tokens]
    int num_layers = 0;
    int max_tokens = 0;
    int window_size = 10000;
    int current_step = 0;
    
    void init(int layers, int max_tokens_per_layer, int window = 10000);
    uint8_t* get_dropout_mask(int layer_idx);
    float* get_normal_noise(int layer_idx);
    void advance_step();
    void free();
};
extern RandomBuffers g_random_buffers;

// ============================================================
// FP16 KERNELS (FP16 in/out, FP32 compute for stability)
// ============================================================
// LayerNorm
void layer_norm_fp16_forward(half* out, float* mean, float* inv_std,
                              const half* x, const half* w, const half* b,
                              int N, int D, float eps = 1e-5f);
void layer_norm_fp16_backward(half* dx, half* dw, half* db,
                               const half* dout, const half* x, const half* w,
                               const float* mean, const float* inv_std, int N, int D);

// ReLU
void relu_fp16_forward(half* out, const half* x, int N);
void relu_fp16_backward(half* dx, const half* dout, const half* x, int N);

// Embedding
void embedding_fp16_forward(half* out, const half* weight, const int* indices, int B, int T, int D);
void embedding_fp16_backward(half* d_weight, const float* d_out, const int* indices, int B, int T, int D, int V);

// Residual & Scale
void residual_add_fp16(half* out, const half* a, const half* b, int N);
void scale_add_fp16(half* dst, const half* src, float alpha, int N);

// Softmax (FP16 in/out, FP32 accumulation internally)
void softmax_fp16_forward(half* out, const half* x, int N, int C);
void softmax_fp16_backward(half* dx, const half* dout, const half* out, int N, int C);

// Softmax FP32 (for attention stability)
void softmax_fp32_forward(float* out, const float* x, int N, int C);
void softmax_fp32_backward(float* dx, const float* dout, const float* out, int N, int C);

// Cross Entropy (FP32 for stability)
float cross_entropy_fp32_forward(const float* logits, const int* targets, int N, int V);
void cross_entropy_fp32_backward(float* d_logits, const float* logits, const int* targets, int N, int V);

// Dropout (FP16)
void dropout_fp16_forward(half* out, const half* x, uint8_t* mask, int N, float p);
void dropout_fp16_backward(half* dx, const half* dout, const uint8_t* mask, int N, float p);

// ============================================================
// GEMM FP16 TENSOR CORES (cuBLASLt)
// ============================================================
void gemm_fp16_tensorcores(half* C, const half* A, const half* B,
                           int M, int N, int K, float alpha = 1.0f, float beta = 0.0f,
                           bool transA = false, bool transB = false);

// FP16 GEMM with FP32 accumulation (for backward)
void gemm_fp16_accum_fp32(float* C, const half* A, const half* B, 
                          int M, int N, int K, float alpha = 1.0f, float beta = 0.0f);
void gemm_fp16_accum_fp32_backward_dA(float* dA, const float* dout, const half* B,
                                      int M, int N, int K, float alpha = 1.0f, float beta = 1.0f);
void gemm_fp16_accum_fp32_backward_dB(half* dB, const half* A, const float* dout,
                                      int M, int N, int K, float alpha = 1.0f, float beta = 1.0f);

// ============================================================
// FUSED KERNELS
// ============================================================
// FFN Fusion: Linear + ReLU + Linear (PRIORIDAD #1)
void fused_ffn_forward(half* out, half* relu_mask,
                       const half* x, const half* w1, const half* b1,
                       const half* w2, const half* b2,
                       int M, int D, int FF);

void fused_ffn_backward(half* dx, half* dw1, half* db1, half* dw2, half* db2,
                        const half* dout, const half* x, const half* fi,
                        const half* w1, const half* w2, const half* relu_mask,
                        int M, int D, int FF);

// QKV Projection Fusion (PRIORIDAD #2)
void fused_qkv_projection(half* q, half* k, half* v,
                          const half* x,
                          const half* wq, const half* bq,
                          const half* wk, const half* bk,
                          const half* wv, const half* bv,
                          int M, int D, int H, int hd);

// Attention Scores + Softmax Fusion
void fused_attn_scores_softmax(half* attn_out, float* attn_fp32,
                               const half* q, const half* k,
                               int B, int H, int T, int hd, float scale);

// LN + Residual + Dropout Fusion
void fused_ln_residual_dropout_forward(half* out, float* mean, float* inv_std, uint8_t* mask,
                                       const half* x, const half* residual,
                                       const half* w, const half* b,
                                       int N, int D, float p, float eps, uint64_t seed, int step);

// ============================================================
// UTILITY
// ============================================================
void apply_causal_mask_fp32(float* att, int B, int H, int T);
void convert_fp32_to_fp16(half* dst, const float* src, int N);
void convert_fp16_to_fp32(float* dst, const half* src, int N);

// Linear layer helpers (FP16)
void linear_forward_fp16(half* out, const half* inp, const half* W, const half* b, int M, int Di, int Do);
void linear_forward_fp16_to_fp32(float* out, const half* inp, const half* W, const half* b, int M, int Di, int Do);
void linear_backward_fp16(half* dinp, half* dW, half* db,
                          const half* dout, const half* inp, const half* W, int M, int Di, int Do);

// Overflow detection
__global__ void check_overflow_kernel(const float* grads, int n, int* found);
bool check_overflow_all(float** gradients, int num_arrays, int* sizes);

// ============================================================
// CONSTANT PTRS (for cublas)
// ============================================================
extern float _zero_f;
extern float _one_f;
extern float _minus_one_f;
#define ZERO_FLOAT_PTR (&_zero_f)
#define ONE_FLOAT_PTR (&_one_f)
#define MINUS_ONE_FLOAT_PTR (&_minus_one_f)

extern half _zero_h;
extern half _one_h;
#define ZERO_HALF_PTR (&_zero_h)
#define ONE_HALF_PTR (&_one_h)