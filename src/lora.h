// ============================================================
// RUBIDIUM TRANSFORMER - LoRA LAYERS
// Low-Rank Adaptation for Fine-tuning
// ============================================================
#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

// ============================================================
// LoRA CONFIG
// ============================================================
struct LoRAConfig {
    int rank = 16;        // LoRA rank (r)
    float alpha = 32.0f;  // Scaling factor
    float dropout = 0.1f; // Dropout probability
    
    // Derived scaling
    float scaling() const { return alpha / rank; }
};

// ============================================================
// LoRA WEIGHTS
// For each target module: W = W_base + A @ B * scaling
// A: [d, r], B: [r, d] -> A@B: [d, d]
// ============================================================
struct LoRAWeights {
    // Q projection
    float *lora_A_q, *lora_B_q;
    float *grad_A_q, *grad_B_q;
    float *m_A_q, *m_B_q, *v_A_q, *v_B_q;
    
    // K projection
    float *lora_A_k, *lora_B_k;
    float *grad_A_k, *grad_B_k;
    float *m_A_k, *m_B_k, *v_A_k, *v_B_k;
    
    // V projection
    float *lora_A_v, *lora_B_v;
    float *grad_A_v, *grad_B_v;
    float *m_A_v, *m_B_v, *v_A_v, *v_B_v;
    
    // O projection
    float *lora_A_o, *lora_B_o;
    float *grad_A_o, *grad_B_o;
    float *m_A_o, *m_B_o, *v_A_o, *v_B_o;
    
    // FFN W1
    float *lora_A_w1, *lora_B_w1;
    float *grad_A_w1, *grad_B_w1;
    float *m_A_w1, *m_B_w1, *v_A_w1, *v_B_w1;
    
    // FFN W2
    float *lora_A_w2, *lora_B_w2;
    float *grad_A_w2, *grad_B_w2;
    float *m_A_w2, *m_B_w2, *v_A_w2, *v_B_w2;
    
    LoRAConfig config;
    int d_model;
    int d_ff;
    bool initialized = false;
    
    // ============================================================
    // INITIALIZE
    // ============================================================
    void init(int d, int ff, const LoRAConfig& cfg) {
        d_model = d;
        d_ff = ff;
        config = cfg;
        
        auto alloc_pair = [](float** A, float** B, int rows, int cols, int r) {
            cudaMalloc(A, rows * r * sizeof(float));
            cudaMalloc(B, r * cols * sizeof(float));
        };
        
        // Q: [d, d] -> A:[d,r], B:[r,d]
        alloc_pair(&lora_A_q, &lora_B_q, d_model, d_model, config.rank);
        cudaMalloc(&grad_A_q, d_model * config.rank * sizeof(float));
        cudaMalloc(&grad_B_q, config.rank * d_model * sizeof(float));
        cudaMalloc(&m_A_q, d_model * config.rank * sizeof(float));
        cudaMalloc(&v_A_q, d_model * config.rank * sizeof(float));
        cudaMalloc(&m_B_q, config.rank * d_model * sizeof(float));
        cudaMalloc(&v_B_q, config.rank * d_model * sizeof(float));
        
        // K
        alloc_pair(&lora_A_k, &lora_B_k, d_model, d_model, config.rank);
        cudaMalloc(&grad_A_k, d_model * config.rank * sizeof(float));
        cudaMalloc(&grad_B_k, config.rank * d_model * sizeof(float));
        cudaMalloc(&m_A_k, d_model * config.rank * sizeof(float));
        cudaMalloc(&v_A_k, d_model * config.rank * sizeof(float));
        cudaMalloc(&m_B_k, config.rank * d_model * sizeof(float));
        cudaMalloc(&v_B_k, config.rank * d_model * sizeof(float));
        
        // V
        alloc_pair(&lora_A_v, &lora_B_v, d_model, d_model, config.rank);
        cudaMalloc(&grad_A_v, d_model * config.rank * sizeof(float));
        cudaMalloc(&grad_B_v, config.rank * d_model * sizeof(float));
        cudaMalloc(&m_A_v, d_model * config.rank * sizeof(float));
        cudaMalloc(&v_A_v, d_model * config.rank * sizeof(float));
        cudaMalloc(&m_B_v, config.rank * d_model * sizeof(float));
        cudaMalloc(&v_B_v, config.rank * d_model * sizeof(float));
        
        // O
        alloc_pair(&lora_A_o, &lora_B_o, d_model, d_model, config.rank);
        cudaMalloc(&grad_A_o, d_model * config.rank * sizeof(float));
        cudaMalloc(&grad_B_o, config.rank * d_model * sizeof(float));
        cudaMalloc(&m_A_o, d_model * config.rank * sizeof(float));
        cudaMalloc(&v_A_o, d_model * config.rank * sizeof(float));
        cudaMalloc(&m_B_o, config.rank * d_model * sizeof(float));
        cudaMalloc(&v_B_o, config.rank * d_model * sizeof(float));
        
        // FFN W1: [d, ff] -> A:[d,r], B:[r,ff]
        alloc_pair(&lora_A_w1, &lora_B_w1, d_model, d_ff, config.rank);
        cudaMalloc(&grad_A_w1, d_model * config.rank * sizeof(float));
        cudaMalloc(&grad_B_w1, config.rank * d_ff * sizeof(float));
        cudaMalloc(&m_A_w1, d_model * config.rank * sizeof(float));
        cudaMalloc(&v_A_w1, d_model * config.rank * sizeof(float));
        cudaMalloc(&m_B_w1, config.rank * d_ff * sizeof(float));
        cudaMalloc(&v_B_w1, config.rank * d_ff * sizeof(float));
        
        // FFN W2: [ff, d] -> A:[ff,r], B:[r,d]
        alloc_pair(&lora_A_w2, &lora_B_w2, d_ff, d_model, config.rank);
        cudaMalloc(&grad_A_w2, d_ff * config.rank * sizeof(float));
        cudaMalloc(&grad_B_w2, config.rank * d_model * sizeof(float));
        cudaMalloc(&m_A_w2, d_ff * config.rank * sizeof(float));
        cudaMalloc(&v_A_w2, d_ff * config.rank * sizeof(float));
        cudaMalloc(&m_B_w2, config.rank * d_model * sizeof(float));
        cudaMalloc(&v_B_w2, config.rank * d_model * sizeof(float));
        
        // Initialize A with small random, B with zeros
        init_weights();
        initialized = true;
        
        printf("LoRA initialized: rank=%d, alpha=%.1f, params=~%.1fM\n",
               config.rank, config.alpha, count_params() / 1e6);
    }
    
    void init_weights() {
        // A: normal(0, 0.02), B: zeros
        // Implement in host code
    }
    
    int count_params() {
        // Q,K,V,O: 4 * 2 * d * r
        // W1: 2 * d * r + 2 * r * ff
        // W2: 2 * ff * r + 2 * r * d
        int attn = 4 * 2 * d_model * config.rank;
        int ffn = 2 * d_model * config.rank + 2 * config.rank * d_ff + 
                  2 * d_ff * config.rank + 2 * config.rank * d_model;
        return attn + ffn;
    }
    
    // ============================================================
    // FORWARD: LoRA output = (x @ A @ B) * scaling
    // x: [BT, d], A: [d, r], B: [r, d] -> out: [BT, d]
    // ============================================================
    void forward_q(const float* x, float* out, int BT, cublasHandle_t handle) {
        float scaling = config.scaling();
        float one = 1.0f, zero = 0.0f;
        
        // temp = x @ A  [BT, r]
        float* temp;
        cudaMalloc(&temp, BT * config.rank * sizeof(float));
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    config.rank, BT, d_model,
                    &one, lora_A_q, config.rank,
                    x, d_model,
                    &zero, temp, config.rank);
        
        // out = temp @ B * scaling  [BT, d]
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    d_model, BT, config.rank,
                    &scaling, lora_B_q, d_model,
                    temp, config.rank,
                    &zero, out, d_model);
        cudaFree(temp);
    }
    
    // Similar for k, v, o, w1, w2...
    
    // ============================================================
    // BACKWARD: Compute gradients for A and B
    // ============================================================
    void backward_q(const float* x, const float* grad_out, int BT, cublasHandle_t handle) {
        // grad_B = A^T @ x^T @ grad_out
        // grad_A = x^T @ grad_out @ B^T
    }
    
    // ============================================================
    // OPTIMIZER STEP
    // ============================================================
    void optimizer_step(int t, float lr, float b1, float b2, float eps, float wd) {
        // AdamW on A and B matrices
    }
    
    // ============================================================
    // FREE
    // ============================================================
    void free() {
        if (!initialized) return;
        auto free_pair = [](float* A, float* B) { cudaFree(A); cudaFree(B); };
        
        free_pair(lora_A_q, lora_B_q);
        free_pair(lora_A_k, lora_B_k);
        free_pair(lora_A_v, lora_B_v);
        free_pair(lora_A_o, lora_B_o);
        free_pair(lora_A_w1, lora_B_w1);
        free_pair(lora_A_w2, lora_B_w2);
        
        // Free grads and moments...
        initialized = false;
    }
};

// ============================================================
// GLOBAL LoRA
// ============================================================
extern LoRAWeights g_lora;

// ============================================================
// LoRA FUNCTIONS
// ============================================================
void init_lora(int d_model, int d_ff, const LoRAConfig& config);
void lora_forward_all(const float* x, float* out_q, float* out_k, 
                      float* out_v, float* out_o, float* out_w1, float* out_w2,
                      int BT, cublasHandle_t handle);
void lora_backward_all(const float* x, const float* grad_q, const float* grad_k,
                       const float* grad_v, const float* grad_o, const float* grad_w1,
                       const float* grad_w2, int BT, cublasHandle_t handle);
void lora_optimizer_step(int t, float lr, float b1, float b2, float eps, float wd);
void lora_free();