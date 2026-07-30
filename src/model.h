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
    void init(int v=256, int t=256, int d=2048, int h=32, int l=10, int ff=8192) {
        V=v; T=t; D=d; H=h; L=l; FF=ff; hd=d/h;
    }
};

// ============================================================
// LAYER WEIGHTS
// ============================================================
struct LayerWeights {
    float *ln1_w, *ln1_b;
    float *ln2_w, *ln2_b;
    float *wq, *bq, *wk, *bk, *wv, *bv, *wo, *bo;
    float *w1, *b1, *w2, *b2;
    // Gradients
    float *g_ln1_w, *g_ln1_b, *g_ln2_w, *g_ln2_b;
    float *g_wq, *g_bq, *g_wk, *g_bk, *g_wv, *g_bv, *g_wo, *g_bo;
    float *g_w1, *g_b1, *g_w2, *g_b2;
    // Adam moments
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
    float *token_emb, *pos_emb;
    float *ln_f_w, *ln_f_b;
    float *lm_w, *lm_b;
    // Gradients
    float *g_token_emb, *g_pos_emb;
    float *g_ln_f_w, *g_ln_f_b;
    float *g_lm_w, *g_lm_b;
    // Adam moments
    float *m_token_emb, *m_pos_emb, *v_token_emb, *v_pos_emb;
    float *m_ln_f_w, *m_ln_f_b, *v_ln_f_w, *v_ln_f_b;
    float *m_lm_w, *m_lm_b, *v_lm_w, *v_lm_b;
    std::vector<LayerWeights> layers;
};

// ============================================================
// ACTIVATIONS
// ============================================================
struct Activations {
    float *x_emb;
    struct LayerActs {
        float *h1, *q, *k, *v;
        float *att, *att_p;
        float *ao, *x1, *h2, *fi;
        float *ln1_mean, *ln1_inv_std;
        float *ln2_mean, *ln2_inv_std;
    };
    std::vector<LayerActs> layers;
    float *hf, *logits;
    float *ln_f_mean, *ln_f_inv_std;
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

    void init(const ModelConfig &config);
    void allocate_weights();
    void init_weights();
    void allocate_activations(int max_BT);
    float forward(const int *d_tokens, const int *d_targets, int B, int T);
    void backward(const int *d_tokens, const int *d_targets, int B, int T, float loss_scale);
    void optimizer_step(int t, float lr, float b1, float b2, float eps, float wd);
    void clip_gradients(float max_norm);
    std::string generate(const std::string &seed, int max_chars, float temp=0.7f, int topk=40);
    void save(const char *path);
    void free_all();
};
