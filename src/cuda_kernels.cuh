// ============================================================
// RUBIDIUM TRANSFORMER - CUDA KERNELS HEADER
// ============================================================
#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

// Error checking
#define CE(e) do { if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); } } while(0)

#define CB(s) do { if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, s); \
    exit(1); } } while(0)

// Global handles
extern cublasHandle_t cublas_handle;
extern cudnnHandle_t cudnn_handle;

void init_handles();
void destroy_handles();

// GEMM
void gemm_forward(float *C, const float *A, const float *B, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f);
void gemm_backward_dA(float *dA, const float *dout, const float *B, int M, int N, int K, float alpha = 1.0f, float beta = 1.0f);
void gemm_backward_dB(float *dB, const float *A, const float *dout, int M, int N, int K, float alpha = 1.0f, float beta = 1.0f);

// LayerNorm
void layer_norm_forward(float *out, float *mean, float *inv_std, const float *x, const float *w, const float *b, int N, int D, float eps = 1e-5f);
void layer_norm_backward(float *dx, float *dw, float *db, const float *dout, const float *x, const float *w, const float *mean, const float *inv_std, int N, int D);

// Softmax
void softmax_forward(float *out, const float *x, int N, int C);
void softmax_backward(float *dx, const float *dout, const float *out, int N, int C);

// ReLU
void relu_forward(float *out, const float *x, int N);
void relu_backward(float *dx, const float *dout, const float *x, int N);

// Cross Entropy
float cross_entropy_forward(const float *logits, const int *targets, int N, int V);
void cross_entropy_backward(float *d_logits, const float *logits, const int *targets, int N, int V);

// Dropout
void dropout_forward(float *out, const float *x, uint8_t *mask, int N, float p = 0.1f);
void dropout_backward(float *dx, const float *dout, const uint8_t *mask, int N, float p = 0.1f);

// Embedding
void embedding_forward(float *out, const float *weight, const int *indices, int B, int T, int D);
void embedding_backward(float *d_weight, const float *d_out, const int *indices, int B, int T, int D, int V);

// AdamW
void adamw_step(float *p, float *g, float *m, float *v, int N, float lr, float b1, float b2, float eps, float wd, int t);

// Utility
void residual_add(float *out, const float *a, const float *b, int N);
void gpu_copy(float *dst, const float *src, int N);
void gpu_zero(float *x, int N);
void scale_add(float *dst, const float *src, float alpha, int N);
void apply_causal_mask(float *att, int B, int H, int T);

// Linear layer helper
void linear_forward(float *out, const float *inp, const float *W, const float *b, int M, int Di, int Do);
void linear_backward(float *dinp, float *dW, float *db, const float *dout, const float *inp, const float *W, int M, int Di, int Do);
