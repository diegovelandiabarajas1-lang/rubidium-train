// ============================================================
// RUBIDIUM TRANSFORMER - CUDA KERNELS IMPLEMENTATION
// ============================================================
#include "cuda_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// ============================================================
// GLOBAL HANDLES
// ============================================================
cublasHandle_t cublas_handle = nullptr;
cudnnHandle_t cudnn_handle = nullptr;

void init_handles() {
    if (!cublas_handle) { CB(cublasCreate(&cublas_handle)); }
    if (!cudnn_handle) { CD(cudnnCreate(&cudnn_handle)); }
}

void destroy_handles() {
    if (cublas_handle) { cublasDestroy(cublas_handle); cublas_handle = nullptr; }
    if (cudnn_handle) { cudnnDestroy(cudnn_handle); cudnn_handle = nullptr; }
}

// Helpers for cuBLAS
float zero_float() { return 0.0f; }
float one_float() { return 1.0f; }
float minus_one_float() { return -1.0f; }

// ============================================================
// GEMM via cuBLAS
// C = alpha * A * B + beta * C (row-major)
// A: [M, K], B: [K, N], C: [M, N]
// ============================================================
void gemm_forward(float *C, const float *A, const float *B,
                  int M, int N, int K, float alpha, float beta) {
    // cuBLAS uses column-major: C_col(M,N) = B_col(N,K) * A_col(K,M)
    // Row-major A[M,K] = Col-major A_col[K,M] with lda=K
    // Row-major B[K,N] = Col-major B_col[N,K] with ldb=N
    // Row-major C[M,N] = Col-major C_col[N,M] with ldc=N
    CB(cublasSgemm(cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    B, N,
                    A, K,
                    &beta,
                    C, N));
}

void gemm_backward_dA(float *dA, const float *dout, const float *B,
                       int M, int N, int K, float alpha, float beta) {
    // dA[M,K] = dout[M,N] * B[K,N]^T
    CB(cublasSgemm(cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    K, M, N,
                    &alpha,
                    B, N,
                    dout, N,
                    &beta,
                    dA, K));
}

void gemm_backward_dB(float *dB, const float *A, const float *dout,
                       int M, int N, int K, float alpha, float beta) {
    // dB[K,N] = A[M,K]^T * dout[M,N]
    CB(cublasSgemm(cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    N, K, M,
                    &alpha,
                    dout, N,
                    A, K,
                    &beta,
                    dB, N));
}

// ============================================================
// LAYER NORM FORWARD
// ============================================================
__global__ void layer_norm_fwd_kernel(float *out, float *mean, float *inv_std,
                                       const float *x, const float *w, const float *b,
                                       int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float sh[];
    float *s_sum = sh;
    float *s_sqsum = sh + blockDim.x;

    const float *xr = x + row * D;
    float *out_ptr = out + row * D;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        local_sum += xr[i];
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
        float d = xr[i] - m;
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
        out_ptr[i] = w[i] * (xr[i] - m) * inv + b[i];
}

void layer_norm_forward(float *out, float *mean, float *inv_std,
                         const float *x, const float *w, const float *b,
                         int N, int D, float eps) {
    int threads = 256;
    if (D < 256) threads = D;
    size_t shmem = 2 * threads * sizeof(float);
    layer_norm_fwd_kernel<<<N, threads, shmem>>>(out, mean, inv_std, x, w, b, N, D, eps);
}

// ============================================================
// LAYER NORM BACKWARD
// ============================================================
__global__ void layer_norm_bwd_kernel(float *dx, float *dw, float *db,
                                       const float *dout, const float *x,
                                       const float *w, const float *mean,
                                       const float *inv_std, int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float sh[];
    float *s1 = sh;
    float *s2 = sh + blockDim.x;

    const float *dxr = dout + row * D;
    const float *xr = x + row * D;
    float *dxor = dx + row * D;
    float m = mean[row];
    float inv = inv_std[row];

    float local_s1 = 0.0f, local_s2 = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float dw_i = dxr[i] * w[i];
        local_s1 += dw_i;
        local_s2 += dw_i * (xr[i] - m);
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
        float dw_i = dxr[i] * w[i];
        dxor[i] = inv * (dw_i - (ts1 + (xr[i] - m) * inv * ts2) / D);
    }

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        atomicAdd(&dw[i], dxr[i] * (xr[i] - m) * inv);
        atomicAdd(&db[i], dxr[i]);
    }
}

void layer_norm_backward(float *dx, float *dw, float *db,
                          const float *dout, const float *x,
                          const float *w, const float *mean,
                          const float *inv_std, int N, int D) {
    int threads = 256;
    if (D < 256) threads = D;
    size_t shmem = 2 * threads * sizeof(float);
    layer_norm_bwd_kernel<<<N, threads, shmem>>>(dx, dw, db, dout, x, w, mean, inv_std, N, D);
}

// ============================================================
// SOFTMAX FORWARD
// ============================================================
__global__ void softmax_fwd_kernel(float *out, const float *x, int N, int C) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float sh[];
    float *s_max = sh;
    float *s_sum = sh + blockDim.x;

    const float *xr = x + row * C;
    float *out_ptr = out + row * C;

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

void softmax_forward(float *out, const float *x, int N, int C) {
    int threads = 256;
    if (C < 256) threads = C;
    size_t shmem = 2 * threads * sizeof(float);
    softmax_fwd_kernel<<<N, threads, shmem>>>(out, x, N, C);
}

// ============================================================
// SOFTMAX BACKWARD
// ============================================================
__global__ void softmax_bwd_kernel(float *dx, const float *dout, const float *out,
                                    int N, int C) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float sh[];

    const float *doutr = dout + row * C;
    const float *outr = out + row * C;
    float *dxr = dx + row * C;

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

void softmax_backward(float *dx, const float *dout, const float *out, int N, int C) {
    int threads = 256;
    if (C < 256) threads = C;
    size_t shmem = threads * sizeof(float);
    softmax_bwd_kernel<<<N, threads, shmem>>>(dx, dout, out, N, C);
}

// ============================================================
// RELU
// ============================================================
__global__ void relu_fwd_kernel(float *out, const float *x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = fmaxf(0.0f, x[i]);
}

__global__ void relu_bwd_kernel(float *dx, const float *dout, const float *x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dx[i] = (x[i] > 0.0f) ? dout[i] : 0.0f;
}

void relu_forward(float *out, const float *x, int N) {
    relu_fwd_kernel<<<(N + 255) / 256, 256>>>(out, x, N);
}

void relu_backward(float *dx, const float *dout, const float *x, int N) {
    relu_bwd_kernel<<<(N + 255) / 256, 256>>>(dx, dout, x, N);
}

// ============================================================
// CROSS ENTROPY LOSS
// ============================================================
__global__ void cross_entropy_fwd_kernel(float *out, const float *logits,
                                          const int *targets, int N, int V) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float sh[];
    float *s_max = sh;
    float *s_sum = sh + blockDim.x;

    const float *lr = logits + row * V;
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

float cross_entropy_forward(const float *logits, const int *targets, int N, int V) {
    float h_loss = 0.0f;
    float *d_loss;
    CE(cudaMalloc(&d_loss, sizeof(float)));
    CE(cudaMemset(d_loss, 0, sizeof(float)));

    int threads = 256;
    if (V < 256) threads = V;
    size_t shmem = 2 * threads * sizeof(float);
    cross_entropy_fwd_kernel<<<N, threads, shmem>>>(d_loss, logits, targets, N, V);

    CE(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CE(cudaFree(d_loss));
    return h_loss / N;
}

// ============================================================
// CROSS ENTROPY BACKWARD
// ============================================================
__global__ void cross_entropy_bwd_kernel(float *d_logits, const float *logits,
                                          const int *targets, int N, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * V;
    if (idx >= total) return;

    int row = idx / V;
    int col = idx % V;
    int tgt = targets[row];

    const float *lr = logits + row * V;
    float mx = -1e30f;
    for (int j = 0; j < V; j++) mx = fmaxf(mx, lr[j]);
    float s = 0.0f;
    for (int j = 0; j < V; j++) s += expf(lr[j] - mx);
    float prob = expf(lr[col] - mx) / s;

    d_logits[idx] = (prob - (col == tgt ? 1.0f : 0.0f)) / (float)N;
}

void cross_entropy_backward(float *d_logits, const float *logits,
                             const int *targets, int N, int V) {
    int total = N * V;
    cross_entropy_bwd_kernel<<<(total + 255) / 256, 256>>>(d_logits, logits, targets, N, V);
}

// ============================================================
// DROPOUT
// ============================================================
__global__ void dropout_fwd_kernel(float *out, const float *x, uint8_t *mask,
                                    int N, float p, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float r = (float)rand() / RAND_MAX;
        mask[i] = (r < p) ? 0 : 1;
        out[i] = mask[i] ? x[i] * scale : 0.0f;
    }
}

__global__ void dropout_bwd_kernel(float *dx, const float *dout, const uint8_t *mask,
                                    int N, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dx[i] = mask[i] ? dout[i] * scale : 0.0f;
}

void dropout_forward(float *out, const float *x, uint8_t *mask, int N, float p) {
    float scale = 1.0f / (1.0f - p);
    dropout_fwd_kernel<<<(N + 255) / 256, 256>>>(out, x, mask, N, p, scale);
}

void dropout_backward(float *dx, const float *dout, const uint8_t *mask, int N, float p) {
    float scale = 1.0f / (1.0f - p);
    dropout_bwd_kernel<<<(N + 255) / 256, 256>>>(dx, dout, mask, N, scale);
}

// ============================================================
// EMBEDDING
// ============================================================
__global__ void embedding_fwd_kernel(float *out, const float *weight,
                                      const int *indices, int B, int T, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * D;
    if (idx >= total) return;
    int b = idx / (T * D);
    int t = (idx / D) % T;
    int d = idx % D;
    int token = indices[b * T + t];
    out[idx] = weight[token * D + d];
}

__global__ void embedding_bwd_kernel(float *d_weight, const float *d_out,
                                      const int *indices, int B, int T, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * D;
    if (idx >= total) return;
    int b = idx / (T * D);
    int t = (idx / D) % T;
    int d = idx % D;
    int token = indices[b * T + t];
    atomicAdd(&d_weight[token * D + d], d_out[idx]);
}

void embedding_forward(float *out, const float *weight, const int *indices, int B, int T, int D) {
    int total = B * T * D;
    embedding_fwd_kernel<<<(total + 255) / 256, 256>>>(out, weight, indices, B, T, D);
}

void embedding_backward(float *d_weight, const float *d_out,
                         const int *indices, int B, int T, int D, int V) {
    int total = B * T * D;
    embedding_bwd_kernel<<<(total + 255) / 256, 256>>>(d_weight, d_out, indices, B, T, D);
}

// ============================================================
// ADAMW
// ============================================================
__global__ void adamw_step_kernel(float *p, float *g, float *m, float *v,
                                   int N, float lr, float b1, float b2,
                                   float eps, float wd, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float gi = g[i] + wd * p[i];
    m[i] = b1 * m[i] + (1.0f - b1) * gi;
    v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;
    float mh = m[i] / (1.0f - powf(b1, (float)t));
    float vh = v[i] / (1.0f - powf(b2, (float)t));
    p[i] -= lr * mh / (sqrtf(vh) + eps);
    g[i] = 0.0f;
}

void adamw_step(float *p, float *g, float *m, float *v,
                 int N, float lr, float b1, float b2,
                 float eps, float wd, int t) {
    adamw_step_kernel<<<(N + 255) / 256, 256>>>(p, g, m, v, N, lr, b1, b2, eps, wd, t);
}

// ============================================================
// UTILITY KERNELS
// ============================================================
__global__ void residual_add_kernel(float *out, const float *a, const float *b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[i] + b[i];
}

void residual_add(float *out, const float *a, const float *b, int N) {
    residual_add_kernel<<<(N + 255) / 256, 256>>>(out, a, b, N);
}

__global__ void copy_kernel(float *dst, const float *src, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[i];
}

void gpu_copy(float *dst, const float *src, int N) {
    copy_kernel<<<(N + 255) / 256, 256>>>(dst, src, N);
}

__global__ void zero_fill_kernel(float *x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = 0.0f;
}

void gpu_zero(float *x, int N) {
    zero_fill_kernel<<<(N + 255) / 256, 256>>>(x, N);
}

__global__ void scale_add_kernel(float *dst, const float *src, float alpha, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] += alpha * src[i];
}

void scale_add(float *dst, const float *src, float alpha, int N) {
    scale_add_kernel<<<(N + 255) / 256, 256>>>(dst, src, alpha, N);
}

__global__ void causal_mask_kernel(float *att, int B, int H, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * T * T;
    if (idx >= total) return;
    int t1 = (idx / T) % T;
    int t2 = idx % T;
    if (t2 > t1) att[idx] = -1e9f;
}

void apply_causal_mask(float *att, int B, int H, int T) {
    int total = B * H * T * T;
    causal_mask_kernel<<<(total + 255) / 256, 256>>>(att, B, H, T);
}

// ============================================================
// LINEAR LAYER HELPERS
// ============================================================
void linear_forward(float *out, const float *inp, const float *W, const float *b,
                    int M, int Di, int Do) {
    // out = inp @ W^T + b
    // inp: [M, Di], W: [Do, Di], out: [M, Do]
    gemm_forward(out, inp, W, M, Do, Di);
    if (b) {
        for (int i = 0; i < M; i++)
            scale_add(out + i * Do, b, 1.0f, Do);
    }
}

void linear_backward(float *dinp, float *dW, float *db,
                     const float *dout, const float *inp,
                     const float *W, int M, int Di, int Do) {
    // dinp = dout @ W
    gemm_backward_dA(dinp, dout, W, M, Do, Di);
    // dW += dout^T @ inp
    gemm_backward_dB(dW, inp, dout, M, Do, Di);
    // db += sum(dout, axis=0)
    if (db) {
        // Use cuBLAS to sum columns: db += ones^T @ dout
        float *d_sum;
        CE(cudaMalloc(&d_sum, Do * sizeof(float)));
        CE(cudaMemset(d_sum, 0, Do * sizeof(float)));
        float one = 1.0f;
        // db[1,Do] = ones[1,M] @ dout[M,Do]
        CB(cublasSgemm(cublas_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        Do, 1, M,
                        &one,
                        dout, Do,
                        nullptr, M, // placeholder
                        &one,
                        d_sum, Do));
        scale_add(db, d_sum, 1.0f, Do);
        CE(cudaFree(d_sum));
    }
}
