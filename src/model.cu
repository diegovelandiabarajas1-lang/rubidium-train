// ============================================================
// RUBIDIUM TRANSFORMER - MODEL IMPLEMENTATION (Mixed Precision FP16)
// ============================================================
#include "model.h"
#include "cuda_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

// ============================================================
// INIT
// ============================================================
void RubidiumTransformer::init(const ModelConfig &config) {
    cfg = config;
    allocate_weights();
    init_weights();
    if (cfg.use_fp16) {
        convert_weights_fp32_to_fp16();
    }
}

// ============================================================
// ALLOCATE WEIGHTS (FP16 + FP32 master)
// ============================================================
void RubidiumTransformer::allocate_weights() {
    auto alloc_pair = [](half *&fp16, float *&fp32, int n) {
        CE(cudaMalloc(&fp16, n * sizeof(half)));
        CE(cudaMemset(fp16, 0, n * sizeof(half)));
        CE(cudaMalloc(&fp32, n * sizeof(float)));
        CE(cudaMemset(fp32, 0, n * sizeof(float)));
    };
    auto alloc_grad = [](float *&g, float *&m, float *&v, int n) {
        CE(cudaMalloc(&g, n * sizeof(float))); CE(cudaMemset(g, 0, n * sizeof(float)));
        CE(cudaMalloc(&m, n * sizeof(float))); CE(cudaMemset(m, 0, n * sizeof(float)));
        CE(cudaMalloc(&v, n * sizeof(float))); CE(cudaMemset(v, 0, n * sizeof(float)));
    };

    // Embeddings
    alloc_pair(w.token_emb, w.token_emb_fp32, cfg.V * cfg.D);
    alloc_pair(w.pos_emb, w.pos_emb_fp32, cfg.T * cfg.D);
    alloc_pair(w.ln_f_w, w.ln_f_w_fp32, cfg.D);
    alloc_pair(w.ln_f_b, w.ln_f_b_fp32, cfg.D);
    alloc_pair(w.lm_w, w.lm_w_fp32, cfg.V * cfg.D);
    alloc_pair(w.lm_b, w.lm_b_fp32, cfg.V);

    alloc_grad(w.g_token_emb, w.m_token_emb, w.v_token_emb, cfg.V * cfg.D);
    alloc_grad(w.g_pos_emb, w.m_pos_emb, w.v_pos_emb, cfg.T * cfg.D);
    alloc_grad(w.g_ln_f_w, w.m_ln_f_w, w.v_ln_f_w, cfg.D);
    alloc_grad(w.g_ln_f_b, w.m_ln_f_b, w.v_ln_f_b, cfg.D);
    alloc_grad(w.g_lm_w, w.m_lm_w, w.v_lm_w, cfg.V * cfg.D);
    alloc_grad(w.g_lm_b, w.m_lm_b, w.v_lm_b, cfg.V);

    w.layers.resize(cfg.L);
    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        alloc_pair(ly.ln1_w, ly.ln1_w_fp32, cfg.D);
        alloc_pair(ly.ln1_b, ly.ln1_b_fp32, cfg.D);
        alloc_pair(ly.ln2_w, ly.ln2_w_fp32, cfg.D);
        alloc_pair(ly.ln2_b, ly.ln2_b_fp32, cfg.D);
        alloc_pair(ly.wq, ly.wq_fp32, cfg.D * cfg.D); alloc_pair(ly.bq, ly.bq_fp32, cfg.D);
        alloc_pair(ly.wk, ly.wk_fp32, cfg.D * cfg.D); alloc_pair(ly.bk, ly.bk_fp32, cfg.D);
        alloc_pair(ly.wv, ly.wv_fp32, cfg.D * cfg.D); alloc_pair(ly.bv, ly.bv_fp32, cfg.D);
        alloc_pair(ly.wo, ly.wo_fp32, cfg.D * cfg.D); alloc_pair(ly.bo, ly.bo_fp32, cfg.D);
        alloc_pair(ly.w1, ly.w1_fp32, cfg.D * cfg.FF); alloc_pair(ly.b1, ly.b1_fp32, cfg.FF);
        alloc_pair(ly.w2, ly.w2_fp32, cfg.FF * cfg.D); alloc_pair(ly.b2, ly.b2_fp32, cfg.D);

        alloc_grad(ly.g_ln1_w, ly.m_ln1_w, ly.v_ln1_w, cfg.D);
        alloc_grad(ly.g_ln1_b, ly.m_ln1_b, ly.v_ln1_b, cfg.D);
        alloc_grad(ly.g_ln2_w, ly.m_ln2_w, ly.v_ln2_w, cfg.D);
        alloc_grad(ly.g_ln2_b, ly.m_ln2_b, ly.v_ln2_b, cfg.D);
        alloc_grad(ly.g_wq, ly.m_wq, ly.v_wq, cfg.D * cfg.D); alloc_grad(ly.g_bq, ly.m_bq, ly.v_bq, cfg.D);
        alloc_grad(ly.g_wk, ly.m_wk, ly.v_wk, cfg.D * cfg.D); alloc_grad(ly.g_bk, ly.m_bk, ly.v_bk, cfg.D);
        alloc_grad(ly.g_wv, ly.m_wv, ly.v_wv, cfg.D * cfg.D); alloc_grad(ly.g_bv, ly.m_bv, ly.v_bv, cfg.D);
        alloc_grad(ly.g_wo, ly.m_wo, ly.v_wo, cfg.D * cfg.D); alloc_grad(ly.g_bo, ly.m_bo, ly.v_bo, cfg.D);
        alloc_grad(ly.g_w1, ly.m_w1, ly.v_w1, cfg.D * cfg.FF); alloc_grad(ly.g_b1, ly.m_b1, ly.v_b1, cfg.FF);
        alloc_grad(ly.g_w2, ly.m_w2, ly.v_w2, cfg.FF * cfg.D); alloc_grad(ly.g_b2, ly.m_b2, ly.v_b2, cfg.D);
    }
}

// ============================================================
// FP32 <-> FP16 CONVERSION
// ============================================================
void RubidiumTransformer::convert_weights_fp32_to_fp16() {
    convert_fp32_to_fp16(w.token_emb, w.token_emb_fp32, cfg.V * cfg.D);
    convert_fp32_to_fp16(w.pos_emb, w.pos_emb_fp32, cfg.T * cfg.D);
    convert_fp32_to_fp16(w.ln_f_w, w.ln_f_w_fp32, cfg.D);
    convert_fp32_to_fp16(w.ln_f_b, w.ln_f_b_fp32, cfg.D);
    convert_fp32_to_fp16(w.lm_w, w.lm_w_fp32, cfg.V * cfg.D);
    convert_fp32_to_fp16(w.lm_b, w.lm_b_fp32, cfg.V);
    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        convert_fp32_to_fp16(ly.ln1_w, ly.ln1_w_fp32, cfg.D);
        convert_fp32_to_fp16(ly.ln1_b, ly.ln1_b_fp32, cfg.D);
        convert_fp32_to_fp16(ly.ln2_w, ly.ln2_w_fp32, cfg.D);
        convert_fp32_to_fp16(ly.ln2_b, ly.ln2_b_fp32, cfg.D);
        convert_fp32_to_fp16(ly.wq, ly.wq_fp32, cfg.D * cfg.D);
        convert_fp32_to_fp16(ly.bq, ly.bq_fp32, cfg.D);
        convert_fp32_to_fp16(ly.wk, ly.wk_fp32, cfg.D * cfg.D);
        convert_fp32_to_fp16(ly.bk, ly.bk_fp32, cfg.D);
        convert_fp32_to_fp16(ly.wv, ly.wv_fp32, cfg.D * cfg.D);
        convert_fp32_to_fp16(ly.bv, ly.bv_fp32, cfg.D);
        convert_fp32_to_fp16(ly.wo, ly.wo_fp32, cfg.D * cfg.D);
        convert_fp32_to_fp16(ly.bo, ly.bo_fp32, cfg.D);
        convert_fp32_to_fp16(ly.w1, ly.w1_fp32, cfg.D * cfg.FF);
        convert_fp32_to_fp16(ly.b1, ly.b1_fp32, cfg.FF);
        convert_fp32_to_fp16(ly.w2, ly.w2_fp32, cfg.FF * cfg.D);
        convert_fp32_to_fp16(ly.b2, ly.b2_fp32, cfg.D);
    }
}

void RubidiumTransformer::convert_weights_fp16_to_fp32() {
    convert_fp16_to_fp32(w.token_emb_fp32, w.token_emb, cfg.V * cfg.D);
    convert_fp16_to_fp32(w.pos_emb_fp32, w.pos_emb, cfg.T * cfg.D);
    convert_fp16_to_fp32(w.ln_f_w_fp32, w.ln_f_w, cfg.D);
    convert_fp16_to_fp32(w.ln_f_b_fp32, w.ln_f_b, cfg.D);
    convert_fp16_to_fp32(w.lm_w_fp32, w.lm_w, cfg.V * cfg.D);
    convert_fp16_to_fp32(w.lm_b_fp32, w.lm_b, cfg.V);
    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        convert_fp16_to_fp32(ly.ln1_w_fp32, ly.ln1_w, cfg.D);
        convert_fp16_to_fp32(ly.ln1_b_fp32, ly.ln1_b, cfg.D);
        convert_fp16_to_fp32(ly.ln2_w_fp32, ly.ln2_w, cfg.D);
        convert_fp16_to_fp32(ly.ln2_b_fp32, ly.ln2_b, cfg.D);
        convert_fp16_to_fp32(ly.wq_fp32, ly.wq, cfg.D * cfg.D);
        convert_fp16_to_fp32(ly.bq_fp32, ly.bq, cfg.D);
        convert_fp16_to_fp32(ly.wk_fp32, ly.wk, cfg.D * cfg.D);
        convert_fp16_to_fp32(ly.bk_fp32, ly.bk, cfg.D);
        convert_fp16_to_fp32(ly.wv_fp32, ly.wv, cfg.D * cfg.D);
        convert_fp16_to_fp32(ly.bv_fp32, ly.bv, cfg.D);
        convert_fp16_to_fp32(ly.wo_fp32, ly.wo, cfg.D * cfg.D);
        convert_fp16_to_fp32(ly.bo_fp32, ly.bo, cfg.D);
        convert_fp16_to_fp32(ly.w1_fp32, ly.w1, cfg.D * cfg.FF);
        convert_fp16_to_fp32(ly.b1_fp32, ly.b1, cfg.FF);
        convert_fp16_to_fp32(ly.w2_fp32, ly.w2, cfg.FF * cfg.D);
        convert_fp16_to_fp32(ly.b2_fp32, ly.b2, cfg.D);
    }
}

// ============================================================
// INIT WEIGHTS (FP32 master)
// ============================================================
void RubidiumTransformer::init_weights() {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    auto randn = [&](float *p, int n, float std) {
        std::vector<float> h(n);
        for (auto &v : h) v = dist(gen) * std;
        CE(cudaMemcpy(p, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    };

    randn(w.token_emb_fp32, cfg.V * cfg.D, 0.02f);
    randn(w.pos_emb_fp32, cfg.T * cfg.D, 0.02f);

    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        std::vector<float> ones(cfg.D, 1.0f);
        std::vector<float> zeros_d(cfg.D, 0.0f);
        std::vector<float> zeros_ff(cfg.FF, 0.0f);
        CE(cudaMemcpy(ly.ln1_w_fp32, ones.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
        CE(cudaMemcpy(ly.ln1_b_fp32, zeros_d.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
        CE(cudaMemcpy(ly.ln2_w_fp32, ones.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
        CE(cudaMemcpy(ly.ln2_b_fp32, zeros_d.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));

        randn(ly.wq_fp32, cfg.D*cfg.D, 0.02f); CE(cudaMemcpy(ly.bq_fp32, zeros_d.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
        randn(ly.wk_fp32, cfg.D*cfg.D, 0.02f); CE(cudaMemcpy(ly.bk_fp32, zeros_d.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
        randn(ly.wv_fp32, cfg.D*cfg.D, 0.02f); CE(cudaMemcpy(ly.bv_fp32, zeros_d.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
        randn(ly.wo_fp32, cfg.D*cfg.D, 0.02f); CE(cudaMemcpy(ly.bo_fp32, zeros_d.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
        randn(ly.w1_fp32, cfg.D*cfg.FF, 0.02f); CE(cudaMemcpy(ly.b1_fp32, zeros_ff.data(), cfg.FF*sizeof(float), cudaMemcpyHostToDevice));
        randn(ly.w2_fp32, cfg.FF*cfg.D, 0.02f); CE(cudaMemcpy(ly.b2_fp32, zeros_d.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
    }

    std::vector<float> ones(cfg.D, 1.0f);
    std::vector<float> zeros(cfg.D, 0.0f);
    CE(cudaMemcpy(w.ln_f_w_fp32, ones.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
    CE(cudaMemcpy(w.ln_f_b_fp32, zeros.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
    randn(w.lm_w_fp32, cfg.V*cfg.D, 0.02f);
    std::vector<float> zeros_v(cfg.V, 0.0f);
    CE(cudaMemcpy(w.lm_b_fp32, zeros_v.data(), cfg.V*sizeof(float), cudaMemcpyHostToDevice));
}

// ============================================================
// ALLOCATE ACTIVATIONS (FP16 with checkpointing)
// ============================================================
void RubidiumTransformer::allocate_activations(int max_BT) {
    auto alloc_fp16 = [](half *&p, int n) { CE(cudaMalloc(&p, n*sizeof(half))); };
    auto alloc_fp32 = [](float *&p, int n) { CE(cudaMalloc(&p, n*sizeof(float))); };

    alloc_fp16(act.x_emb, max_BT * cfg.D);
    act.layers.resize(cfg.L);
    for (int l = 0; l < cfg.L; l++) {
        auto &la = act.layers[l];
        alloc_fp16(la.h1, max_BT*cfg.D);
        alloc_fp16(la.q, max_BT*cfg.D); alloc_fp16(la.k, max_BT*cfg.D); alloc_fp16(la.v, max_BT*cfg.D);
        alloc_fp32(la.att, 8*cfg.H*cfg.T*cfg.T);
        alloc_fp32(la.att_p, 8*cfg.H*cfg.T*cfg.T);
        alloc_fp16(la.ao, max_BT*cfg.D);
        alloc_fp16(la.x1, max_BT*cfg.D);
        alloc_fp16(la.h2, max_BT*cfg.D);
        alloc_fp16(la.fi, max_BT*cfg.FF);
        alloc_fp32(la.ln1_mean, max_BT); alloc_fp32(la.ln1_inv_std, max_BT);
        alloc_fp32(la.ln2_mean, max_BT); alloc_fp32(la.ln2_inv_std, max_BT);

        if (cfg.use_activation_checkpointing) {
            alloc_fp16(la.x1_checkpoint, max_BT*cfg.D);
            alloc_fp16(la.h2_checkpoint, max_BT*cfg.D);
        }
    }
    alloc_fp16(act.hf, max_BT*cfg.D);
    alloc_fp32(act.logits, max_BT*cfg.V);
    alloc_fp32(act.ln_f_mean, max_BT); alloc_fp32(act.ln_f_inv_std, max_BT);
}

// ============================================================
// FREE ACTIVATIONS
// ============================================================
void RubidiumTransformer::free_activations() {
    auto fg = [](half *&p) { if(p){cudaFree(p);p=nullptr;} };
    auto fg32 = [](float *&p) { if(p){cudaFree(p);p=nullptr;} };
    fg(act.x_emb);
    for (auto &la : act.layers) {
        fg(la.h1); fg(la.q); fg(la.k); fg(la.v);
        fg32(la.att); fg32(la.att_p);
        fg(la.ao); fg(la.x1); fg(la.h2); fg(la.fi);
        fg32(la.ln1_mean); fg32(la.ln1_inv_std);
        fg32(la.ln2_mean); fg32(la.ln2_inv_std);
        if (cfg.use_activation_checkpointing) {
            fg(la.x1_checkpoint); fg(la.h2_checkpoint);
        }
    }
    fg(act.hf);
    fg32(act.logits);
    fg32(act.ln_f_mean); fg32(act.ln_f_inv_std);
}

// ============================================================
// FORWARD (Mixed Precision FP16 + Activation Checkpointing)
// ============================================================
float RubidiumTransformer::forward(const int *d_tokens, const int *d_targets, int B, int T) {
    int BT = B * T;

    // 1. Embedding (FP16)
    embedding_fp16_forward(act.x_emb, w.token_emb, d_tokens, B, T, cfg.D);
    for (int b = 0; b < B; b++)
        scale_add_fp16(act.x_emb + b*T*cfg.D, w.pos_emb, 1.0f, T*cfg.D);

    // 2. Transformer layers
    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        auto &la = act.layers[l];
        half *x0 = (l == 0) ? act.x_emb : act.layers[l-1].x1;

        // LayerNorm 1
        layer_norm_fp16_forward(la.h1, la.ln1_mean, la.ln1_inv_std, x0, ly.ln1_w, ly.ln1_b, BT, cfg.D);

        // Fused QKV projection
        fused_qkv_projection(la.q, la.k, la.v, la.h1, ly.wq, ly.bq, ly.wk, ly.bk, ly.wv, ly.bv, BT, cfg.D, cfg.H, cfg.hd);

        // Attention scores (FP32 for stability)
        float scale = 1.0f / sqrtf((float)cfg.hd);
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < cfg.H; h++) {
                half *q_h = la.q + (b*T + 0)*H*hd + h*hd;  // stride per token = H*hd = D
                half *k_h = la.k + (b*T + 0)*H*hd + h*hd;
                float *att_h = la.att + (b*cfg.H+h)*T*T;
                // att_h[T,T] = q_h[T,hd] * k_h[T,hd]^T * scale
                // Use cuBLAS: C(T,T) = A(T,hd) * B(hd,T) -> B(hd,T) transposed
                gemm_fp16_accum_fp32(att_h, q_h, k_h, T, T, cfg.hd, scale, 0.0f);
            }
        }
        apply_causal_mask_fp32(la.att, B, cfg.H, T);
        softmax_fp32_forward(la.att_p, la.att, B*cfg.H*T, T);

        // Weighted sum
        float one = 1.0f;
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < cfg.H; h++) {
                half *v_h = la.v + (b*T + 0)*cfg.D + h*cfg.hd;
                float *att_h = la.att_p + (b*cfg.H+h)*T*T;
                half *ao_h = la.ao + (b*T + 0)*cfg.D + h*cfg.hd;
                gemm_fp16_accum_fp32(ao_h, att_h, v_h, T, cfg.hd, T, one, 0.0f);
            }
        }

        // Output projection (FP16)
        linear_forward_fp16(la.x1, la.ao, ly.wo, ly.bo, BT, cfg.D, cfg.D);
        residual_add_fp16(la.x1, x0, la.x1, BT*cfg.D);

        if (cfg.use_activation_checkpointing)
            CE(cudaMemcpy(la.x1_checkpoint, la.x1, BT*cfg.D*sizeof(half), cudaMemcpyDeviceToDevice));

        // LayerNorm 2
        layer_norm_fp16_forward(la.h2, la.ln2_mean, la.ln2_inv_std, la.x1, ly.ln2_w, ly.ln2_b, BT, cfg.D);

        if (cfg.use_activation_checkpointing)
            CE(cudaMemcpy(la.h2_checkpoint, la.h2, BT*cfg.D*sizeof(half), cudaMemcpyDeviceToDevice));

        // FFN (fused: Linear + ReLU + Linear)
        fused_ffn_forward(la.fi, nullptr, la.h2, ly.w1, ly.b1, ly.w2, ly.b2, BT, cfg.D, cfg.FF);
        residual_add_fp16(la.x1, la.x1, la.fi, BT*cfg.D);
    }

    // 3. Final LN
    layer_norm_fp16_forward(act.hf, act.ln_f_mean, act.ln_f_inv_std,
                           act.layers[cfg.L-1].x1, w.ln_f_w, w.ln_f_b, BT, cfg.D);

    // 4. LM head (FP16 -> FP32 logits)
    linear_forward_fp16_to_fp32(act.logits, act.hf, w.lm_w, w.lm_b, BT, cfg.D, cfg.V);

    // 5. Loss (FP32)
    if (d_targets) return cross_entropy_fp32_forward(act.logits, d_targets, BT, cfg.V);
    return 0.0f;
}

// ============================================================
// BACKWARD (Mixed Precision FP16 + Loss Scaling)
// ============================================================
void RubidiumTransformer::backward(const int *d_tokens, const int *d_targets,
                                    int B, int T, float loss_scale_val) {
    int BT = B * T;

    // Zero gradients (FP32)
    gpu_zero(w.g_token_emb, cfg.V*cfg.D);
    gpu_zero(w.g_pos_emb, cfg.T*cfg.D);
    gpu_zero(w.g_ln_f_w, cfg.D); gpu_zero(w.g_ln_f_b, cfg.D);
    gpu_zero(w.g_lm_w, cfg.V*cfg.D); gpu_zero(w.g_lm_b, cfg.V);
    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        gpu_zero(ly.g_ln1_w, cfg.D); gpu_zero(ly.g_ln1_b, cfg.D);
        gpu_zero(ly.g_ln2_w, cfg.D); gpu_zero(ly.g_ln2_b, cfg.D);
        gpu_zero(ly.g_wq, cfg.D*cfg.D); gpu_zero(ly.g_bq, cfg.D);
        gpu_zero(ly.g_wk, cfg.D*cfg.D); gpu_zero(ly.g_bk, cfg.D);
        gpu_zero(ly.g_wv, cfg.D*cfg.D); gpu_zero(ly.g_bv, cfg.D);
        gpu_zero(ly.g_wo, cfg.D*cfg.D); gpu_zero(ly.g_bo, cfg.D);
        gpu_zero(ly.g_w1, cfg.D*cfg.FF); gpu_zero(ly.g_b1, cfg.FF);
        gpu_zero(ly.g_w2, cfg.FF*cfg.D); gpu_zero(ly.g_b2, cfg.D);
    }

    // Temp buffers
    float *d_dlogits, *d_dx;
    CE(cudaMalloc(&d_dlogits, BT*cfg.V*sizeof(float)));
    CE(cudaMalloc(&d_dx, BT*cfg.D*sizeof(float)));

    // CE backward (FP32)
    cross_entropy_fp32_backward(d_dlogits, act.logits, d_targets, BT, cfg.V);
    if (loss_scale_val != 1.0f)
        scale_add(d_dlogits, d_dlogits, loss_scale_val - 1.0f, BT*cfg.V);

    // LM head backward
    gemm_fp16_accum_fp32_backward_dA(d_dx, d_dlogits, w.lm_w, BT, cfg.V, cfg.D);
    gemm_fp16_accum_fp32_backward_dB(w.g_lm_w, act.hf, d_dlogits, BT, cfg.V, cfg.D);

    // Final LN backward
    float *d_ln_f;
    CE(cudaMalloc(&d_ln_f, BT*cfg.D*sizeof(float)));
    layer_norm_fp16_backward((half*)d_ln_f, w.g_ln_f_w, w.g_ln_f_b, (half*)d_dx,
                           act.layers[cfg.L-1].x1, w.ln_f_w,
                           act.ln_f_mean, act.ln_f_inv_std, BT, cfg.D);

    // Backward through layers (reverse)
    float *d_res = d_ln_f;
    float *d_fi, *d_h2, *d_ln2, *d_ao, *d_h1, *d_ln1;
    CE(cudaMalloc(&d_fi, BT*cfg.FF*sizeof(float)));
    CE(cudaMalloc(&d_h2, BT*cfg.D*sizeof(float)));
    CE(cudaMalloc(&d_ln2, BT*cfg.D*sizeof(float)));
    CE(cudaMalloc(&d_ao, BT*cfg.D*sizeof(float)));
    CE(cudaMalloc(&d_h1, BT*cfg.D*sizeof(float)));
    CE(cudaMalloc(&d_ln1, BT*cfg.D*sizeof(float)));

    for (int l = cfg.L-1; l >= 0; l--) {
        auto &ly = w.layers[l];
        auto &la = act.layers[l];

        // FFN backward
        fused_ffn_backward((half*)d_h2, ly.g_w1, ly.g_b1, ly.g_w2, ly.g_b2,
                          (half*)d_res, la.h2, ly.w1, ly.w2, ly.w1, la.fi,
                          BT, cfg.D, cfg.FF);

        // LN2 backward
        half *x1_src = cfg.use_activation_checkpointing ? la.x1_checkpoint : la.x1;
        layer_norm_fp16_backward((half*)d_ln2, ly.g_ln2_w, ly.g_ln2_b, (half*)d_h2,
                                x1_src, ly.ln2_w, la.ln2_mean, la.ln2_inv_std, BT, cfg.D);

        // Attention backward
        gemm_fp16_accum_fp32_backward_dA(d_ao, d_ln2, ly.wo, BT, cfg.D, cfg.D);
        gemm_fp16_accum_fp32_backward_dB(ly.g_wo, la.ao, d_ln2, BT, cfg.D, cfg.D);

        // QKV backward
        gemm_fp16_accum_fp32_backward_dA(d_h1, d_ao, ly.wq, BT, cfg.D, cfg.D);
        gemm_fp16_accum_fp32_backward_dB(ly.g_wq, la.h1, d_ao, BT, cfg.D, cfg.D);
        gemm_fp16_accum_fp32_backward_dA(d_h1, d_ao, ly.wk, BT, cfg.D, cfg.D);
        gemm_fp16_accum_fp32_backward_dB(ly.g_wk, la.h1, d_ao, BT, cfg.D, cfg.D);
        gemm_fp16_accum_fp32_backward_dA(d_h1, d_ao, ly.wv, BT, cfg.D, cfg.D);
        gemm_fp16_accum_fp32_backward_dB(ly.g_wv, la.h1, d_ao, BT, cfg.D, cfg.D);

        // LN1 backward
        half *x0 = (l == 0) ? act.x_emb : (cfg.use_activation_checkpointing ?
                                           act.layers[l-1].x1_checkpoint : act.layers[l-1].x1);
        layer_norm_fp16_backward((half*)d_ln1, ly.g_ln1_w, ly.g_ln1_b, (half*)d_h1,
                                x0, ly.ln1_w, la.ln1_mean, la.ln1_inv_std, BT, cfg.D);

        scale_add(d_res, d_ln1, 1.0f, BT*cfg.D);
    }

    // Embedding backward
    embedding_fp16_backward(w.g_token_emb, d_res, d_tokens, B, T, cfg.D, cfg.V);

    // Check for overflow
    std::vector<float*> grads = {w.g_token_emb, w.g_pos_emb, w.g_ln_f_w, w.g_ln_f_b, w.g_lm_w, w.g_lm_b};
    std::vector<int> sizes = {cfg.V*cfg.D, cfg.T*cfg.D, cfg.D, cfg.D, cfg.V*cfg.D, cfg.V};
    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        grads.insert(grads.end(), {ly.g_ln1_w, ly.g_ln1_b, ly.g_ln2_w, ly.g_ln2_b,
                                   ly.g_wq, ly.g_bq, ly.g_wk, ly.g_bk, ly.g_wv, ly.g_bv,
                                   ly.g_wo, ly.g_bo, ly.g_w1, ly.g_b1, ly.g_w2, ly.g_b2});
        sizes.insert(sizes.end(), {cfg.D, cfg.D, cfg.D, cfg.D,
                                   cfg.D*cfg.D, cfg.D, cfg.D*cfg.D, cfg.D,
                                   cfg.D*cfg.D, cfg.D, cfg.D*cfg.D, cfg.D,
                                   cfg.D*cfg.FF, cfg.FF, cfg.FF*cfg.D, cfg.D});
    }
    overflow = check_overflow_all(grads.data(), grads.size(), sizes.data());
    update_loss_scale(overflow);

    // Cleanup
    CE(cudaFree(d_dlogits)); CE(cudaFree(d_dx)); CE(cudaFree(d_ln_f));
    CE(cudaFree(d_fi)); CE(cudaFree(d_h2)); CE(cudaFree(d_ln2));
    CE(cudaFree(d_ao)); CE(cudaFree(d_h1)); CE(cudaFree(d_ln1));
}

// ============================================================
// LOSS SCALE UPDATE
// ============================================================
void RubidiumTransformer::update_loss_scale(bool overflow) {
    if (overflow) {
        loss_scale *= 0.5f;
        if (loss_scale < 1.0f) loss_scale = 1.0f;
    } else {
        static int unskipped = 0;
        unskipped++;
        if (unskipped >= 2000) {
            loss_scale *= 2.0f;
            unskipped = 0;
            if (loss_scale > 65536.0f) loss_scale = 65536.0f;
        }
    }
}

// ============================================================
// OPTIMIZER STEP (FP32 master weights + loss scaling)
// ============================================================
void RubidiumTransformer::optimizer_step(int t, float lr, float b1, float b2, float eps, float wd) {
    float scaled_lr = lr;

    auto step = [&](float *p_fp32, half *p_fp16, float *g, float *m, float *v, int n) {
        adamw_step(p_fp32, g, m, v, n, scaled_lr, b1, b2, eps, wd, t);
        convert_fp32_to_fp16(p_fp16, p_fp32, n);
    };

    step(w.token_emb_fp32, w.token_emb, w.g_token_emb, w.m_token_emb, w.v_token_emb, cfg.V*cfg.D);
    step(w.pos_emb_fp32, w.pos_emb, w.g_pos_emb, w.m_pos_emb, w.v_pos_emb, cfg.T*cfg.D);
    step(w.ln_f_w_fp32, w.ln_f_w, w.g_ln_f_w, w.m_ln_f_w, w.v_ln_f_w, cfg.D);
    step(w.ln_f_b_fp32, w.ln_f_b, w.g_ln_f_b, w.m_ln_f_b, w.v_ln_f_b, cfg.D);
    step(w.lm_w_fp32, w.lm_w, w.g_lm_w, w.m_lm_w, w.v_lm_w, cfg.V*cfg.D);
    step(w.lm_b_fp32, w.lm_b, w.g_lm_b, w.m_lm_b, w.v_lm_b, cfg.V);

    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        step(ly.ln1_w_fp32, ly.ln1_w, ly.g_ln1_w, ly.m_ln1_w, ly.v_ln1_w, cfg.D);
        step(ly.ln1_b_fp32, ly.ln1_b, ly.g_ln1_b, ly.m_ln1_b, ly.v_ln1_b, cfg.D);
        step(ly.ln2_w_fp32, ly.ln2_w, ly.g_ln2_w, ly.m_ln2_w, ly.v_ln2_w, cfg.D);
        step(ly.ln2_b_fp32, ly.ln2_b, ly.g_ln2_b, ly.m_ln2_b, ly.v_ln2_b, cfg.D);
        step(ly.wq_fp32, ly.wq, ly.g_wq, ly.m_wq, ly.v_wq, cfg.D*cfg.D);
        step(ly.bq_fp32, ly.bq, ly.g_bq, ly.m_bq, ly.v_bq, cfg.D);
        step(ly.wk_fp32, ly.wk, ly.g_wk, ly.m_wk, ly.v_wk, cfg.D*cfg.D);
        step(ly.bk_fp32, ly.bk, ly.g_bk, ly.m_bk, ly.v_bk, cfg.D);
        step(ly.wv_fp32, ly.wv, ly.g_wv, ly.m_wv, ly.v_wv, cfg.D*cfg.D);
        step(ly.bv_fp32, ly.bv, ly.g_bv, ly.m_bv, ly.v_bv, cfg.D);
        step(ly.wo_fp32, ly.wo, ly.g_wo, ly.m_wo, ly.v_wo, cfg.D*cfg.D);
        step(ly.bo_fp32, ly.bo, ly.g_bo, ly.m_bo, ly.v_bo, cfg.D);
        step(ly.w1_fp32, ly.w1, ly.g_w1, ly.m_w1, ly.v_w1, cfg.D*cfg.FF);
        step(ly.b1_fp32, ly.b1, ly.g_b1, ly.m_b1, ly.v_b1, cfg.FF);
        step(ly.w2_fp32, ly.w2, ly.g_w2, ly.m_w2, ly.v_w2, cfg.FF*cfg.D);
        step(ly.b2_fp32, ly.b2, ly.g_b2, ly.m_b2, ly.v_b2, cfg.D);
    }
}

// ============================================================
// GRADIENT CLIPPING
// ============================================================
void RubidiumTransformer::clip_gradients(float max_norm) {
    auto clip = [](float *g, int n, float max_norm) {
        std::vector<float> h(n);
        CE(cudaMemcpy(h.data(), g, n*sizeof(float), cudaMemcpyDeviceToHost));
        float norm = 0.0f;
        for (float v : h) norm += v*v;
        norm = sqrtf(norm);
        if (norm > max_norm) {
            float scale = max_norm / norm;
            for (int i = 0; i < n; i++) h[i] *= scale;
            CE(cudaMemcpy(g, h.data(), n*sizeof(float), cudaMemcpyHostToDevice));
        }
    };
    clip(w.g_token_emb, cfg.V*cfg.D, max_norm);
    clip(w.g_pos_emb, cfg.T*cfg.D, max_norm);
    clip(w.g_ln_f_w, cfg.D, max_norm);
    clip(w.g_ln_f_b, cfg.D, max_norm);
    clip(w.g_lm_w, cfg.V*cfg.D, max_norm);
    clip(w.g_lm_b, cfg.V, max_norm);
    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        clip(ly.g_ln1_w, cfg.D, max_norm); clip(ly.g_ln1_b, cfg.D, max_norm);
        clip(ly.g_ln2_w, cfg.D, max_norm); clip(ly.g_ln2_b, cfg.D, max_norm);
        clip(ly.g_wq, cfg.D*cfg.D, max_norm); clip(ly.g_bq, cfg.D, max_norm);
        clip(ly.g_wk, cfg.D*cfg.D, max_norm); clip(ly.g_bk, cfg.D, max_norm);
        clip(ly.g_wv, cfg.D*cfg.D, max_norm); clip(ly.g_bv, cfg.D, max_norm);
        clip(ly.g_wo, cfg.D*cfg.D, max_norm); clip(ly.g_bo, cfg.D, max_norm);
        clip(ly.g_w1, cfg.D*cfg.FF, max_norm); clip(ly.g_b1, cfg.FF, max_norm);
        clip(ly.g_w2, cfg.FF*cfg.D, max_norm); clip(ly.g_b2, cfg.D, max_norm);
    }
}

// ============================================================
// GENERATE
// ============================================================
std::string RubidiumTransformer::generate(const std::string &seed, int max_chars,
                                           float temperature, int top_k) {
    std::vector<int> tokens;
    for (char c : seed) {
        auto it = char_to_id.find((unsigned char)c);
        tokens.push_back(it != char_to_id.end() ? it->second : 0);
    }
    float temp = fmaxf(temperature, 0.05f);

    for (int i = 0; i < max_chars; i++) {
        int start = std::max(0, (int)tokens.size() - cfg.T);
        int seq_len = tokens.size() - start;

        std::vector<int> h_tokens(tokens.begin()+start, tokens.end());
        int *d_tok;
        CE(cudaMalloc(&d_tok, seq_len*sizeof(int)));
        CE(cudaMemcpy(d_tok, h_tokens.data(), seq_len*sizeof(int), cudaMemcpyHostToDevice));

        forward(d_tok, nullptr, 1, seq_len);

        std::vector<float> h_logits(cfg.V);
        CE(cudaMemcpy(h_logits.data(), act.logits + (seq_len-1)*cfg.V,
                       cfg.V*sizeof(float), cudaMemcpyDeviceToHost));

        for (auto &v : h_logits) v /= temp;
        if (top_k > 0 && top_k < cfg.V) {
            std::vector<int> idx(cfg.V);
            for (int j = 0; j < cfg.V; j++) idx[j] = j;
            std::partial_sort(idx.begin(), idx.begin()+top_k, idx.end(),
                [&](int a, int b){ return h_logits[a] > h_logits[b]; });
            float thresh = h_logits[idx[top_k-1]];
            for (auto &v : h_logits) if (v < thresh) v = -1e9f;
        }
        float mx = *std::max_element(h_logits.begin(), h_logits.end());
        float s = 0.0f;
        for (auto &v : h_logits) { v = expf(v-mx); s += v; }
        for (auto &v : h_logits) v /= s;

        float r = (float)rand()/RAND_MAX;
        float cum = 0.0f;
        int next = 0;
        for (int j = 0; j < cfg.V; j++) { cum += h_logits[j]; if (r <= cum) { next = j; break; } }
        tokens.push_back(next);
        CE(cudaFree(d_tok));
    }

    std::string result;
    for (size_t i = seed.size(); i < tokens.size(); i++) {
        auto it = id_to_char.find(tokens[i]);
        result += (it != id_to_char.end()) ? (char)it->second : '?';
    }
    return result;
}

// ============================================================
// SAVE
// ============================================================
void RubidiumTransformer::save(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return; }
    fwrite("RBN2", 1, 4, f);  // v2 format: FP16 weights
    fwrite(&cfg.V, sizeof(int), 1, f);
    fwrite(&cfg.T, sizeof(int), 1, f);
    fwrite(&cfg.D, sizeof(int), 1, f);
    fwrite(&cfg.H, sizeof(int), 1, f);
    fwrite(&cfg.L, sizeof(int), 1, f);
    fwrite(&cfg.FF, sizeof(int), 1, f);
    int map[256] = {};
    for (auto &p : char_to_id) map[(unsigned char)p.first] = p.second;
    fwrite(map, sizeof(int), 256, f);

    auto wg16 = [&](half *p, int n) {
        fwrite(p, sizeof(half), n, f);
    };
    auto wg32 = [&](float *p, int n) {
        std::vector<float> h(n);
        CE(cudaMemcpy(h.data(), p, n*sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(h.data(), sizeof(float), n, f);
    };

    wg16(w.token_emb, cfg.V*cfg.D);
    wg16(w.pos_emb, cfg.T*cfg.D);
    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        wg16(ly.ln1_w, cfg.D); wg16(ly.ln1_b, cfg.D);
        wg16(ly.ln2_w, cfg.D); wg16(ly.ln2_b, cfg.D);
        wg16(ly.wq, cfg.D*cfg.D); wg16(ly.bq, cfg.D);
        wg16(ly.wk, cfg.D*cfg.D); wg16(ly.bk, cfg.D);
        wg16(ly.wv, cfg.D*cfg.D); wg16(ly.bv, cfg.D);
        wg16(ly.wo, cfg.D*cfg.D); wg16(ly.bo, cfg.D);
        wg16(ly.w1, cfg.D*cfg.FF); wg16(ly.b1, cfg.FF);
        wg16(ly.w2, cfg.FF*cfg.D); wg16(ly.b2, cfg.D);
    }
    wg16(w.ln_f_w, cfg.D); wg16(w.ln_f_b, cfg.D);
    wg16(w.lm_w, cfg.V*cfg.D); wg16(w.lm_b, cfg.V);
    fclose(f);
    printf("Saved: %s (%.1f MB)\n", path, ftell(f)/1e6);
}

// ============================================================
// LOAD
// ============================================================
void RubidiumTransformer::load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return; }
    char magic[4];
    fread(magic, 1, 4, f);
    fread(&cfg.V, sizeof(int), 1, f);
    fread(&cfg.T, sizeof(int), 1, f);
    fread(&cfg.D, sizeof(int), 1, f);
    fread(&cfg.H, sizeof(int), 1, f);
    fread(&cfg.L, sizeof(int), 1, f);
    fread(&cfg.FF, sizeof(int), 1, f);
    cfg.hd = cfg.D / cfg.H;

    int map[256];
    fread(map, sizeof(int), 256, f);
    char_to_id.clear(); id_to_char.clear();
    for (int i = 0; i < 256; i++) {
        if (map[i] != 0 || i == 0) {
            char_to_id[(unsigned char)i] = map[i];
            id_to_char[map[i]] = (unsigned char)i;
        }
    }

    allocate_weights();

    auto rg16 = [&](half *p, int n) {
        fread(p, sizeof(half), n, f);
    };

    rg16(w.token_emb, cfg.V*cfg.D);
    rg16(w.pos_emb, cfg.T*cfg.D);
    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        rg16(ly.ln1_w, cfg.D); rg16(ly.ln1_b, cfg.D);
        rg16(ly.ln2_w, cfg.D); rg16(ly.ln2_b, cfg.D);
        rg16(ly.wq, cfg.D*cfg.D); rg16(ly.bq, cfg.D);
        rg16(ly.wk, cfg.D*cfg.D); rg16(ly.bk, cfg.D);
        rg16(ly.wv, cfg.D*cfg.D); rg16(ly.bv, cfg.D);
        rg16(ly.wo, cfg.D*cfg.D); rg16(ly.bo, cfg.D);
        rg16(ly.w1, cfg.D*cfg.FF); rg16(ly.b1, cfg.FF);
        rg16(ly.w2, cfg.FF*cfg.D); rg16(ly.b2, cfg.D);
    }
    rg16(w.ln_f_w, cfg.D); rg16(w.ln_f_b, cfg.D);
    rg16(w.lm_w, cfg.V*cfg.D); rg16(w.lm_b, cfg.V);

    // Copy FP16 -> FP32 master
    convert_weights_fp16_to_fp32();
    fclose(f);
    printf("Loaded: %s (V=%d T=%d D=%d H=%d L=%d FF=%d)\n", path, cfg.V, cfg.T, cfg.D, cfg.H, cfg.L, cfg.FF);
}

// ============================================================
// FREE ALL
// ============================================================
void RubidiumTransformer::free_all() {
    auto fg = [](half *&p) { if(p){cudaFree(p);p=nullptr;} };
    auto fg32 = [](float *&p) { if(p){cudaFree(p);p=nullptr;} };
    fg(w.token_emb); fg(w.pos_emb);
    fg(w.ln_f_w); fg(w.ln_f_b);
    fg(w.lm_w); fg(w.lm_b);
    fg32(w.token_emb_fp32); fg32(w.pos_emb_fp32);
    fg32(w.ln_f_w_fp32); fg32(w.ln_f_b_fp32);
    fg32(w.lm_w_fp32); fg32(w.lm_b_fp32);
    fg32(w.g_token_emb); fg32(w.g_pos_emb);
    fg32(w.g_ln_f_w); fg32(w.g_ln_f_b);
    fg32(w.g_lm_w); fg32(w.g_lm_b);
    fg32(w.m_token_emb); fg32(w.v_token_emb);
    fg32(w.m_pos_emb); fg32(w.v_pos_emb);
    fg32(w.m_ln_f_w); fg32(w.v_ln_f_w);
    fg32(w.m_ln_f_b); fg32(w.v_ln_f_b);
    fg32(w.m_lm_w); fg32(w.v_lm_w);
    fg32(w.m_lm_b); fg32(w.v_lm_b);
    for (auto &ly : w.layers) {
        fg(ly.ln1_w);fg(ly.ln1_b);fg(ly.ln2_w);fg(ly.ln2_b);
        fg(ly.wq);fg(ly.bq);fg(ly.wk);fg(ly.bk);fg(ly.wv);fg(ly.bv);fg(ly.wo);fg(ly.bo);
        fg(ly.w1);fg(ly.b1);fg(ly.w2);fg(ly.b2);
        fg32(ly.ln1_w_fp32);fg32(ly.ln1_b_fp32);fg32(ly.ln2_w_fp32);fg32(ly.ln2_b_fp32);
        fg32(ly.wq_fp32);fg32(ly.bq_fp32);fg32(ly.wk_fp32);fg32(ly.bk_fp32);
        fg32(ly.wv_fp32);fg32(ly.bv_fp32);fg32(ly.wo_fp32);fg32(ly.bo_fp32);
        fg32(ly.w1_fp32);fg32(ly.b1_fp32);fg32(ly.w2_fp32);fg32(ly.b2_fp32);
        fg32(ly.g_ln1_w);fg32(ly.g_ln1_b);fg32(ly.g_ln2_w);fg32(ly.g_ln2_b);
        fg32(ly.g_wq);fg32(ly.g_bq);fg32(ly.g_wk);fg32(ly.g_bk);
        fg32(ly.g_wv);fg32(ly.g_bv);fg32(ly.g_wo);fg32(ly.g_bo);
        fg32(ly.g_w1);fg32(ly.g_b1);fg32(ly.g_w2);fg32(ly.g_b2);
        fg32(ly.m_ln1_w);fg32(ly.m_ln1_b);fg32(ly.m_ln2_w);fg32(ly.m_ln2_b);
        fg32(ly.m_wq);fg32(ly.m_bq);fg32(ly.m_wk);fg32(ly.m_bk);
        fg32(ly.m_wv);fg32(ly.m_bv);fg32(ly.m_wo);fg32(ly.m_bo);
        fg32(ly.m_w1);fg32(ly.m_b1);fg32(ly.m_w2);fg32(ly.m_b2);
        fg32(ly.v_ln1_w);fg32(ly.v_ln1_b);fg32(ly.v_ln2_w);fg32(ly.v_ln2_b);
        fg32(ly.v_wq);fg32(ly.v_bq);fg32(ly.v_wk);fg32(ly.v_bk);
        fg32(ly.v_wv);fg32(ly.v_bv);fg32(ly.v_wo);fg32(ly.v_bo);
        fg32(ly.v_w1);fg32(ly.v_b1);fg32(ly.v_w2);fg32(ly.v_b2);
    }
    free_activations();
    destroy_handles();
}
