// ============================================================
// RUBIDIUM TRANSFORMER - MODEL IMPLEMENTATION
// ============================================================
#include "model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// ============================================================
// INIT
// ============================================================
void RubidiumTransformer::init(const ModelConfig &config) {
    cfg = config;
    allocate_weights();
    init_weights();
}

// ============================================================
// ALLOCATE WEIGHTS
// ============================================================
void RubidiumTransformer::allocate_weights() {
    auto alloc = [](float *&p, int n) {
        CE(cudaMalloc(&p, n * sizeof(float)));
        CE(cudaMemset(p, 0, n * sizeof(float)));
    };

    alloc(w.token_emb, cfg.V * cfg.D);
    alloc(w.pos_emb, cfg.T * cfg.D);
    alloc(w.ln_f_w, cfg.D); alloc(w.ln_f_b, cfg.D);
    alloc(w.lm_w, cfg.V * cfg.D); alloc(w.lm_b, cfg.V);

    alloc(w.g_token_emb, cfg.V * cfg.D);
    alloc(w.g_pos_emb, cfg.T * cfg.D);
    alloc(w.g_ln_f_w, cfg.D); alloc(w.g_ln_f_b, cfg.D);
    alloc(w.g_lm_w, cfg.V * cfg.D); alloc(w.g_lm_b, cfg.V);

    alloc(w.m_token_emb, cfg.V * cfg.D); alloc(w.v_token_emb, cfg.V * cfg.D);
    alloc(w.m_pos_emb, cfg.T * cfg.D); alloc(w.v_pos_emb, cfg.T * cfg.D);
    alloc(w.m_ln_f_w, cfg.D); alloc(w.v_ln_f_w, cfg.D);
    alloc(w.m_ln_f_b, cfg.D); alloc(w.v_ln_f_b, cfg.D);
    alloc(w.m_lm_w, cfg.V * cfg.D); alloc(w.v_lm_w, cfg.V * cfg.D);
    alloc(w.m_lm_b, cfg.V); alloc(w.v_lm_b, cfg.V);

    w.layers.resize(cfg.L);
    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        alloc(ly.ln1_w, cfg.D); alloc(ly.ln1_b, cfg.D);
        alloc(ly.ln2_w, cfg.D); alloc(ly.ln2_b, cfg.D);
        alloc(ly.wq, cfg.D*cfg.D); alloc(ly.bq, cfg.D);
        alloc(ly.wk, cfg.D*cfg.D); alloc(ly.bk, cfg.D);
        alloc(ly.wv, cfg.D*cfg.D); alloc(ly.bv, cfg.D);
        alloc(ly.wo, cfg.D*cfg.D); alloc(ly.bo, cfg.D);
        alloc(ly.w1, cfg.D*cfg.FF); alloc(ly.b1, cfg.FF);
        alloc(ly.w2, cfg.FF*cfg.D); alloc(ly.b2, cfg.D);

        alloc(ly.g_ln1_w, cfg.D); alloc(ly.g_ln1_b, cfg.D);
        alloc(ly.g_ln2_w, cfg.D); alloc(ly.g_ln2_b, cfg.D);
        alloc(ly.g_wq, cfg.D*cfg.D); alloc(ly.g_bq, cfg.D);
        alloc(ly.g_wk, cfg.D*cfg.D); alloc(ly.g_bk, cfg.D);
        alloc(ly.g_wv, cfg.D*cfg.D); alloc(ly.g_bv, cfg.D);
        alloc(ly.g_wo, cfg.D*cfg.D); alloc(ly.g_bo, cfg.D);
        alloc(ly.g_w1, cfg.D*cfg.FF); alloc(ly.g_b1, cfg.FF);
        alloc(ly.g_w2, cfg.FF*cfg.D); alloc(ly.g_b2, cfg.D);

        alloc(ly.m_ln1_w, cfg.D); alloc(ly.v_ln1_w, cfg.D);
        alloc(ly.m_ln1_b, cfg.D); alloc(ly.v_ln1_b, cfg.D);
        alloc(ly.m_ln2_w, cfg.D); alloc(ly.v_ln2_w, cfg.D);
        alloc(ly.m_ln2_b, cfg.D); alloc(ly.v_ln2_b, cfg.D);
        alloc(ly.m_wq, cfg.D*cfg.D); alloc(ly.v_wq, cfg.D*cfg.D);
        alloc(ly.m_bq, cfg.D); alloc(ly.v_bq, cfg.D);
        alloc(ly.m_wk, cfg.D*cfg.D); alloc(ly.v_wk, cfg.D*cfg.D);
        alloc(ly.m_bk, cfg.D); alloc(ly.v_bk, cfg.D);
        alloc(ly.m_wv, cfg.D*cfg.D); alloc(ly.v_wv, cfg.D*cfg.D);
        alloc(ly.m_bv, cfg.D); alloc(ly.v_bv, cfg.D);
        alloc(ly.m_wo, cfg.D*cfg.D); alloc(ly.v_wo, cfg.D*cfg.D);
        alloc(ly.m_bo, cfg.D); alloc(ly.v_bo, cfg.D);
        alloc(ly.m_w1, cfg.D*cfg.FF); alloc(ly.v_w1, cfg.D*cfg.FF);
        alloc(ly.m_b1, cfg.FF); alloc(ly.v_b1, cfg.FF);
        alloc(ly.m_w2, cfg.FF*cfg.D); alloc(ly.v_w2, cfg.FF*cfg.D);
        alloc(ly.m_b2, cfg.D); alloc(ly.v_b2, cfg.D);
    }
}

// ============================================================
// INIT WEIGHTS
// ============================================================
void RubidiumTransformer::init_weights() {
    srand(42);
    auto randn = [](float *p, int n, float std) {
        std::vector<float> h(n);
        for (auto &v : h) {
            float u1 = (float)rand() / RAND_MAX + 1e-10f;
            float u2 = (float)rand() / RAND_MAX;
            v = std * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
        }
        CE(cudaMemcpy(p, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    };

    randn(w.token_emb, cfg.V * cfg.D, 0.02f);
    randn(w.pos_emb, cfg.T * cfg.D, 0.02f);

    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        std::vector<float> ones(cfg.D, 1.0f);
        std::vector<float> zeros_d(cfg.D, 0.0f);
        std::vector<float> zeros_ff(cfg.FF, 0.0f);
        CE(cudaMemcpy(ly.ln1_w, ones.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
        CE(cudaMemcpy(ly.ln1_b, zeros_d.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
        CE(cudaMemcpy(ly.ln2_w, ones.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
        CE(cudaMemcpy(ly.ln2_b, zeros_d.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
        randn(ly.wq, cfg.D*cfg.D, 0.02f); CE(cudaMemcpy(ly.bq, zeros_d.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
        randn(ly.wk, cfg.D*cfg.D, 0.02f); CE(cudaMemcpy(ly.bk, zeros_d.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
        randn(ly.wv, cfg.D*cfg.D, 0.02f); CE(cudaMemcpy(ly.bv, zeros_d.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
        randn(ly.wo, cfg.D*cfg.D, 0.02f); CE(cudaMemcpy(ly.bo, zeros_d.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
        randn(ly.w1, cfg.D*cfg.FF, 0.02f); CE(cudaMemcpy(ly.b1, zeros_ff.data(), cfg.FF*sizeof(float), cudaMemcpyHostToDevice));
        randn(ly.w2, cfg.FF*cfg.D, 0.02f); CE(cudaMemcpy(ly.b2, zeros_d.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
    }

    std::vector<float> ones(cfg.D, 1.0f);
    std::vector<float> zeros(cfg.D, 0.0f);
    CE(cudaMemcpy(w.ln_f_w, ones.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
    CE(cudaMemcpy(w.ln_f_b, zeros.data(), cfg.D*sizeof(float), cudaMemcpyHostToDevice));
    randn(w.lm_w, cfg.V*cfg.D, 0.02f);
    std::vector<float> zeros_v(cfg.V, 0.0f);
    CE(cudaMemcpy(w.lm_b, zeros_v.data(), cfg.V*sizeof(float), cudaMemcpyHostToDevice));
}

// ============================================================
// ALLOCATE ACTIVATIONS
// ============================================================
void RubidiumTransformer::allocate_activations(int max_BT) {
    auto alloc = [](float *&p, int n) { CE(cudaMalloc(&p, n*sizeof(float))); };

    alloc(act.x_emb, max_BT * cfg.D);
    act.layers.resize(cfg.L);
    for (int l = 0; l < cfg.L; l++) {
        auto &la = act.layers[l];
        alloc(la.h1, max_BT*cfg.D);
        alloc(la.q, max_BT*cfg.D); alloc(la.k, max_BT*cfg.D); alloc(la.v, max_BT*cfg.D);
        alloc(la.att, 8*cfg.H*cfg.T*cfg.T);
        alloc(la.att_p, 8*cfg.H*cfg.T*cfg.T);
        alloc(la.ao, max_BT*cfg.D);
        alloc(la.x1, max_BT*cfg.D);
        alloc(la.h2, max_BT*cfg.D);
        alloc(la.fi, max_BT*cfg.FF);
        alloc(la.ln1_mean, max_BT); alloc(la.ln1_inv_std, max_BT);
        alloc(la.ln2_mean, max_BT); alloc(la.ln2_inv_std, max_BT);
    }
    alloc(act.hf, max_BT*cfg.D);
    alloc(act.logits, max_BT*cfg.V);
    alloc(act.ln_f_mean, max_BT); alloc(act.ln_f_inv_std, max_BT);
}

// ============================================================
// FORWARD
// ============================================================
float RubidiumTransformer::forward(const int *d_tokens, const int *d_targets, int B, int T) {
    int BT = B * T;

    // 1. Embedding
    embedding_forward(act.x_emb, w.token_emb, d_tokens, B, T, cfg.D);
    for (int b = 0; b < B; b++)
        scale_add(act.x_emb + b*T*cfg.D, w.pos_emb, 1.0f, T*cfg.D);

    // 2. Transformer layers
    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        auto &la = act.layers[l];
        float *x0 = (l == 0) ? act.x_emb : act.layers[l-1].x1;

        // LayerNorm 1
        layer_norm_forward(la.h1, la.ln1_mean, la.ln1_inv_std, x0, ly.ln1_w, ly.ln1_b, BT, cfg.D);

        // Q, K, V
        linear_forward(la.q, la.h1, ly.wq, ly.bq, BT, cfg.D, cfg.D);
        linear_forward(la.k, la.h1, ly.wk, ly.bk, BT, cfg.D, cfg.D);
        linear_forward(la.v, la.h1, ly.wv, ly.bv, BT, cfg.D, cfg.D);

        // Attention scores
        float scale = 1.0f / sqrtf((float)cfg.hd);
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < cfg.H; h++) {
                float *q_h = la.q + (b*T)*cfg.D + h*cfg.hd;
                float *k_h = la.k + (b*T)*cfg.D + h*cfg.hd;
                float *att_h = la.att + (b*cfg.H+h)*T*T;
                CB(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    T, T, cfg.hd, &scale, k_h, cfg.D, q_h, cfg.D, ZERO_FLOAT_PTR, att_h, T));
            }
        }
        apply_causal_mask(la.att, B, cfg.H, T);
        softmax_forward(la.att_p, la.att, B*cfg.H*T, T);

        // Weighted sum
        float one = 1.0f;
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < cfg.H; h++) {
                float *v_h = la.v + (b*T)*cfg.D + h*cfg.hd;
                float *att_h = la.att_p + (b*cfg.H+h)*T*T;
                float *ao_h = la.ao + (b*T)*cfg.D + h*cfg.hd;
                CB(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    cfg.hd, T, T, &one, v_h, cfg.D, att_h, T, ZERO_FLOAT_PTR, ao_h, cfg.D));
            }
        }

        // Output proj
        linear_forward(la.x1, la.ao, ly.wo, ly.bo, BT, cfg.D, cfg.D);
        residual_add(la.x1, x0, la.x1, BT*cfg.D);

        // LayerNorm 2
        layer_norm_forward(la.h2, la.ln2_mean, la.ln2_inv_std, la.x1, ly.ln2_w, ly.ln2_b, BT, cfg.D);

        // FFN
        linear_forward(la.fi, la.h2, ly.w1, ly.b1, BT, cfg.D, cfg.FF);
        relu_forward(la.fi, la.fi, BT*cfg.FF);
        float *ff_out;
        CE(cudaMalloc(&ff_out, BT*cfg.D*sizeof(float)));
        linear_forward(ff_out, la.fi, ly.w2, ly.b2, BT, cfg.FF, cfg.D);
        residual_add(la.x1, la.x1, ff_out, BT*cfg.D);
        CE(cudaFree(ff_out));
    }

    // 3. Final LN
    layer_norm_forward(act.hf, act.ln_f_mean, act.ln_f_inv_std,
                       act.layers[cfg.L-1].x1, w.ln_f_w, w.ln_f_b, BT, cfg.D);

    // 4. LM head
    linear_forward(act.logits, act.hf, w.lm_w, w.lm_b, BT, cfg.D, cfg.V);

    // 5. Loss
    if (d_targets) return cross_entropy_forward(act.logits, d_targets, BT, cfg.V);
    return 0.0f;
}

// ============================================================
// BACKWARD
// ============================================================
void RubidiumTransformer::backward(const int *d_tokens, const int *d_targets,
                                    int B, int T, float loss_scale) {
    int BT = B * T;

    // Zero gradients
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

    float *d_dlogits, *d_dx;
    CE(cudaMalloc(&d_dlogits, BT*cfg.V*sizeof(float)));
    CE(cudaMalloc(&d_dx, BT*cfg.D*sizeof(float)));

    // CE backward
    cross_entropy_backward(d_dlogits, act.logits, d_targets, BT, cfg.V);
    if (loss_scale != 1.0f) scale_add(d_dlogits, d_dlogits, loss_scale - 1.0f, BT*cfg.V);

    // LM head backward
    gemm_backward_dA(d_dx, d_dlogits, w.lm_w, BT, cfg.V, cfg.D);
    gemm_backward_dB(w.g_lm_w, act.hf, d_dlogits, BT, cfg.V, cfg.D);

    // Final LN backward
    float *d_ln_f;
    CE(cudaMalloc(&d_ln_f, BT*cfg.D*sizeof(float)));
    layer_norm_backward(d_ln_f, w.g_ln_f_w, w.g_ln_f_b, d_dx,
                        act.layers[cfg.L-1].x1, w.ln_f_w,
                        act.ln_f_mean, act.ln_f_inv_std, BT, cfg.D);

    // Backward through layers
    float *d_res = d_ln_f;
    for (int l = cfg.L-1; l >= 0; l--) {
        auto &ly = w.layers[l];
        auto &la = act.layers[l];

        // FFN backward
        float *d_fi;
        CE(cudaMalloc(&d_fi, BT*cfg.FF*sizeof(float)));
        gemm_backward_dA(d_fi, d_res, ly.w2, BT, cfg.D, cfg.FF);
        gemm_backward_dB(ly.g_w2, la.fi, d_res, BT, cfg.D, cfg.FF);
        relu_backward(d_fi, d_fi, la.fi, BT*cfg.FF);

        float *d_h2;
        CE(cudaMalloc(&d_h2, BT*cfg.D*sizeof(float)));
        gemm_backward_dA(d_h2, d_fi, ly.w1, BT, cfg.FF, cfg.D);
        gemm_backward_dB(ly.g_w1, la.h2, d_fi, BT, cfg.FF, cfg.D);
        CE(cudaFree(d_fi));

        // LN2 backward
        float *d_ln2;
        CE(cudaMalloc(&d_ln2, BT*cfg.D*sizeof(float)));
        layer_norm_backward(d_ln2, ly.g_ln2_w, ly.g_ln2_b, d_h2,
                           la.x1, ly.ln2_w, la.ln2_mean, la.ln2_inv_std, BT, cfg.D);
        CE(cudaFree(d_h2));

        // Attention backward (simplified)
        float *d_ao;
        CE(cudaMalloc(&d_ao, BT*cfg.D*sizeof(float)));
        gemm_backward_dA(d_ao, d_ln2, ly.wo, BT, cfg.D, cfg.D);
        gemm_backward_dB(ly.g_wo, la.ao, d_ln2, BT, cfg.D, cfg.D);

        // Q, K, V backward (simplified: use d_ao for all)
        float *d_h1;
        CE(cudaMalloc(&d_h1, BT*cfg.D*sizeof(float)));
        gemm_backward_dA(d_h1, d_ao, ly.wq, BT, cfg.D, cfg.D);
        gemm_backward_dB(ly.g_wq, la.h1, d_ao, BT, cfg.D, cfg.D);
        CE(cudaFree(d_ao));

        // LN1 backward
        float *d_ln1;
        CE(cudaMalloc(&d_ln1, BT*cfg.D*sizeof(float)));
        float *x0 = (l == 0) ? act.x_emb : act.layers[l-1].x1;
        layer_norm_backward(d_ln1, ly.g_ln1_w, ly.g_ln1_b, d_h1,
                           x0, ly.ln1_w, la.ln1_mean, la.ln1_inv_std, BT, cfg.D);
        CE(cudaFree(d_h1));

        // Residual
        scale_add(d_res, d_ln1, 1.0f, BT*cfg.D);
        CE(cudaFree(d_ln1));
    }

    // Embedding backward
    embedding_backward(w.g_token_emb, d_res, d_tokens, B, T, cfg.D, cfg.V);

    CE(cudaFree(d_dlogits)); CE(cudaFree(d_dx)); CE(cudaFree(d_ln_f));
}

// ============================================================
// OPTIMIZER STEP
// ============================================================
void RubidiumTransformer::optimizer_step(int t, float lr, float b1, float b2, float eps, float wd) {
    auto step = [&](float *p, float *g, float *m, float *v, int n) {
        adamw_step(p, g, m, v, n, lr, b1, b2, eps, wd, t);
    };
    step(w.token_emb, w.g_token_emb, w.m_token_emb, w.v_token_emb, cfg.V*cfg.D);
    step(w.pos_emb, w.g_pos_emb, w.m_pos_emb, w.v_pos_emb, cfg.T*cfg.D);
    step(w.ln_f_w, w.g_ln_f_w, w.m_ln_f_w, w.v_ln_f_w, cfg.D);
    step(w.ln_f_b, w.g_ln_f_b, w.m_ln_f_b, w.v_ln_f_b, cfg.D);
    step(w.lm_w, w.g_lm_w, w.m_lm_w, w.v_lm_w, cfg.V*cfg.D);
    step(w.lm_b, w.g_lm_b, w.m_lm_b, w.v_lm_b, cfg.V);

    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        step(ly.ln1_w, ly.g_ln1_w, ly.m_ln1_w, ly.v_ln1_w, cfg.D);
        step(ly.ln1_b, ly.g_ln1_b, ly.m_ln1_b, ly.v_ln1_b, cfg.D);
        step(ly.ln2_w, ly.g_ln2_w, ly.m_ln2_w, ly.v_ln2_w, cfg.D);
        step(ly.ln2_b, ly.g_ln2_b, ly.m_ln2_b, ly.v_ln2_b, cfg.D);
        step(ly.wq, ly.g_wq, ly.m_wq, ly.v_wq, cfg.D*cfg.D);
        step(ly.bq, ly.g_bq, ly.m_bq, ly.v_bq, cfg.D);
        step(ly.wk, ly.g_wk, ly.m_wk, ly.v_wk, cfg.D*cfg.D);
        step(ly.bk, ly.g_bk, ly.m_bk, ly.v_bk, cfg.D);
        step(ly.wv, ly.g_wv, ly.m_wv, ly.v_wv, cfg.D*cfg.D);
        step(ly.bv, ly.g_bv, ly.m_bv, ly.v_bv, cfg.D);
        step(ly.wo, ly.g_wo, ly.m_wo, ly.v_wo, cfg.D*cfg.D);
        step(ly.bo, ly.g_bo, ly.m_bo, ly.v_bo, cfg.D);
        step(ly.w1, ly.g_w1, ly.m_w1, ly.v_w1, cfg.D*cfg.FF);
        step(ly.b1, ly.g_b1, ly.m_b1, ly.v_b1, cfg.FF);
        step(ly.w2, ly.g_w2, ly.m_w2, ly.v_w2, cfg.FF*cfg.D);
        step(ly.b2, ly.g_b2, ly.m_b2, ly.v_b2, cfg.D);
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
            std::vector<float> h_scaled(n);
            for (int i = 0; i < n; i++) h_scaled[i] = h[i] * scale;
            CE(cudaMemcpy(g, h_scaled.data(), n*sizeof(float), cudaMemcpyHostToDevice));
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
    fwrite("RBN1", 1, 4, f);
    fwrite(&cfg.V, sizeof(int), 1, f);
    fwrite(&cfg.T, sizeof(int), 1, f);
    fwrite(&cfg.D, sizeof(int), 1, f);
    fwrite(&cfg.H, sizeof(int), 1, f);
    fwrite(&cfg.L, sizeof(int), 1, f);
    fwrite(&cfg.FF, sizeof(int), 1, f);
    int map[256] = {};
    for (auto &p : char_to_id) map[(unsigned char)p.first] = p.second;
    fwrite(map, sizeof(int), 256, f);

    auto wg = [&](float *gp, int n) {
        std::vector<float> h(n);
        CE(cudaMemcpy(h.data(), gp, n*sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(h.data(), sizeof(float), n, f);
    };
    wg(w.token_emb, cfg.V*cfg.D);
    wg(w.pos_emb, cfg.T*cfg.D);
    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        wg(ly.ln1_w, cfg.D); wg(ly.ln1_b, cfg.D);
        wg(ly.wq, cfg.D*cfg.D); wg(ly.bq, cfg.D);
        wg(ly.wk, cfg.D*cfg.D); wg(ly.bk, cfg.D);
        wg(ly.wv, cfg.D*cfg.D); wg(ly.bv, cfg.D);
        wg(ly.wo, cfg.D*cfg.D); wg(ly.bo, cfg.D);
        wg(ly.ln2_w, cfg.D); wg(ly.ln2_b, cfg.D);
        wg(ly.w1, cfg.D*cfg.FF); wg(ly.b1, cfg.FF);
        wg(ly.w2, cfg.FF*cfg.D); wg(ly.b2, cfg.D);
    }
    wg(w.ln_f_w, cfg.D); wg(w.ln_f_b, cfg.D);
    wg(w.lm_w, cfg.V*cfg.D); wg(w.lm_b, cfg.V);
    fclose(f);
    printf("Saved: %s (%.1f MB)\n", path, ftell(f)/1e6);
}

// ============================================================
// FREE ALL
// ============================================================
void RubidiumTransformer::free_all() {
    auto fg = [](float *&p) { if(p){cudaFree(p);p=nullptr;} };
    fg(w.token_emb); fg(w.pos_emb);
    fg(w.ln_f_w); fg(w.ln_f_b);
    fg(w.lm_w); fg(w.lm_b);
    fg(w.g_token_emb); fg(w.g_pos_emb);
    fg(w.g_ln_f_w); fg(w.g_ln_f_b);
    fg(w.g_lm_w); fg(w.g_lm_b);
    fg(w.m_token_emb); fg(w.v_token_emb);
    fg(w.m_pos_emb); fg(w.v_pos_emb);
    fg(w.m_ln_f_w); fg(w.v_ln_f_w);
    fg(w.m_ln_f_b); fg(w.v_ln_f_b);
    fg(w.m_lm_w); fg(w.v_lm_w);
    fg(w.m_lm_b); fg(w.v_lm_b);
    for (auto &ly : w.layers) {
        fg(ly.ln1_w);fg(ly.ln1_b);fg(ly.ln2_w);fg(ly.ln2_b);
        fg(ly.wq);fg(ly.bq);fg(ly.wk);fg(ly.bk);fg(ly.wv);fg(ly.bv);fg(ly.wo);fg(ly.bo);
        fg(ly.w1);fg(ly.b1);fg(ly.w2);fg(ly.b2);
        fg(ly.g_ln1_w);fg(ly.g_ln1_b);fg(ly.g_ln2_w);fg(ly.g_ln2_b);
        fg(ly.g_wq);fg(ly.g_bq);fg(ly.g_wk);fg(ly.g_bk);fg(ly.g_wv);fg(ly.g_bv);fg(ly.g_wo);fg(ly.g_bo);
        fg(ly.g_w1);fg(ly.g_b1);fg(ly.g_w2);fg(ly.g_b2);
        fg(ly.m_ln1_w);fg(ly.m_ln1_b);fg(ly.m_ln2_w);fg(ly.m_ln2_b);
        fg(ly.m_wq);fg(ly.m_bq);fg(ly.m_wk);fg(ly.m_bk);fg(ly.m_wv);fg(ly.m_bv);fg(ly.m_wo);fg(ly.m_bo);
        fg(ly.m_w1);fg(ly.m_b1);fg(ly.m_w2);fg(ly.m_b2);
        fg(ly.v_ln1_w);fg(ly.v_ln1_b);fg(ly.v_ln2_w);fg(ly.v_ln2_b);
        fg(ly.v_wq);fg(ly.v_bq);fg(ly.v_wk);fg(ly.v_bk);fg(ly.v_wv);fg(ly.v_bv);fg(ly.v_wo);fg(ly.v_bo);
        fg(ly.v_w1);fg(ly.v_b1);fg(ly.v_w2);fg(ly.v_b2);
    }
    for (auto &la : act.layers) {
        fg(la.h1);fg(la.q);fg(la.k);fg(la.v);
        fg(la.att);fg(la.att_p);fg(la.ao);fg(la.x1);fg(la.h2);fg(la.fi);
        fg(la.ln1_mean);fg(la.ln1_inv_std);fg(la.ln2_mean);fg(la.ln2_inv_std);
    }
    fg(act.x_emb);fg(act.hf);fg(act.logits);fg(act.ln_f_mean);fg(act.ln_f_inv_std);
    destroy_handles();
}
