// ============================================================
// RUBIDIUM TRANSFORMER - MODEL HEADER
// ============================================================
#pragma once
#include "cuda_kernels.cuh"
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <cstring>
#include <algorithm>

// ============================================================
// MODEL CONFIG
// ============================================================
struct ModelConfig {
    int V, T, D, H, L, FF, hd;
    bool use_fp16 = true;
    bool use_activation_checkpointing = true;
    int grad_accum_steps = 16;
    void init(int v=32000, int t=512, int d=2048, int h=32, int l=10, int ff=8192) {
        V=v; T=t; D=d; H=h; L=l; FF=ff; hd=d/h;
    }
};

// ============================================================
// LAYER WEIGHTS
// ============================================================
struct LayerWeights {
    // FP16 weights (for forward)
    half *ln1_w, *ln1_b;
    half *ln2_w, *ln2_b;
    half *wq, *bq, *wk, *bk, *wv, *bv, *wo, *bo;
    half *w1, *b1, *w2, *b2;
    
    // FP32 master weights (for optimizer)
    float *ln1_w_fp32, *ln1_b_fp32;
    float *ln2_w_fp32, *ln2_b_fp32;
    float *wq_fp32, *bq_fp32, *wk_fp32, *bk_fp32;
    float *wv_fp32, *bv_fp32, *wo_fp32, *bo_fp32;
    float *w1_fp32, *b1_fp32, *w2_fp32, *b2_fp32;
    
    // Gradients (FP32 for accumulation)
    float *g_ln1_w, *g_ln1_b, *g_ln2_w, *g_ln2_b;
    float *g_wq, *g_bq, *g_wk, *g_bk, *g_wv, *g_bv, *g_wo, *g_bo;
    float *g_w1, *g_b1, *g_w2, *g_b2;
    
    // Adam moments (FP32)
    float *m_ln1_w, *m_ln1_b, *m_ln2_w, *m_ln2_b;
    float *m_wq, *m_bq, *m_wk, *m_bk, *m_wv, *m_bv, *m_wo, *m_bo;
    float *m_w1, *m_b1, *m_w2, *m_b2;
    float *v_ln1_w, *v_ln1_b, *v_ln2_w, *v_ln2_b;
    float *v_wq, *v_bq, *v_wk, *v_bk, *v_wv, *v_bv, *v_wo, *v_bo;
    float *v_w1, *v_b1, *v_w2, *v_b2;
};

// ============================================================
// MODEL WEIGHTS
// ============================================================
struct ModelWeights {
    // FP16 weights
    half *token_emb, *pos_emb;
    half *ln_f_w, *ln_f_b;
    half *lm_w, *lm_b;
    
    // FP32 master weights
    float *token_emb_fp32, *pos_emb_fp32;
    float *ln_f_w_fp32, *ln_f_b_fp32;
    float *lm_w_fp32, *lm_b_fp32;
    
    // Gradients (FP32)
    float *g_token_emb, *g_pos_emb;
    float *g_ln_f_w, *g_ln_f_b;
    float *g_lm_w, *g_lm_b;
    
    // Adam moments (FP32)
    float *m_token_emb, *m_pos_emb, *v_token_emb, *v_pos_emb;
    float *m_ln_f_w, *m_ln_f_b, *v_ln_f_w, *v_ln_f_b;
    float *m_lm_w, *m_lm_b, *v_lm_w, *v_lm_b;
    
    std::vector<LayerWeights> layers;
};

// ============================================================
// ACTIVATIONS (with checkpointing support)
// ============================================================
struct Activations {
    half *x_emb;
    struct LayerActs {
        // Stored activations (for backward)
        half *h1, *q, *k, *v;
        half *att, *att_p;
        half *ao, *x1, *h2, *fi;
        float *ln1_mean, *ln1_inv_std;
        float *ln2_mean, *ln2_inv_std;
        
        // Checkpointed: only store these, recompute rest
        half *x1_checkpoint;  // Output of attention block
        half *h2_checkpoint;  // Output of FFN block
    };
    std::vector<LayerActs> layers;
    half *hf, *logits;
    float *ln_f_mean, *ln_f_inv_std;
    
    // Dropout masks (pre-generated)
    uint8_t **dropout_masks;
};

// ============================================================
// RUBIDIUM TRANSFORMER
// ============================================================
struct RubidiumTransformer {
    ModelConfig cfg;
    ModelWeights w;
    Activations act;
    std::map<unsigned char, int> char_to_id;
    std::map<int, unsigned char> id_to_char;
    
    // Mixed precision
    float loss_scale = 65536.0f;
    bool overflow = false;
    
    void init(const ModelConfig &config);
    void allocate_weights();
    void init_weights();
    void allocate_activations(int max_BT);
    void free_activations();
    void convert_weights_fp16_to_fp32();  // Copy FP16 -> FP32 master
    void convert_weights_fp32_to_fp16();  // Copy FP32 master -> FP16
    float forward(const int *d_tokens, const int *d_targets, int B, int T);
    void backward(const int *d_tokens, const int *d_targets, int B, int T, float loss_scale);
    void optimizer_step(int t, float lr, float b1, float b2, float eps, float wd);
    void clip_gradients(float max_norm);
    void update_loss_scale(bool overflow);
    std::string generate(const std::string &seed, int max_chars, float temp=0.7f, int topk=40);
    void save(const char *path);
    void load(const char *path);
    void free_all();
};
