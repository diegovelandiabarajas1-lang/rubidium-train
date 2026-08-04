// ============================================================
// RUBIDIUM - CPU Transformer Model (250M params, parallel)
// ============================================================
#pragma once
#include "cpu_mat.h"
#include <vector>
#include <map>
#include <string>

// ============================================================
// MODEL CONFIG
// ============================================================
struct CPUConfig {
    int V = 32000, T = 512, D = 1536, H = 24, L = 10, FF = 6144, hd = 64;
    void init(int v=32000, int t=512, int d=1536, int h=24, int l=10, int ff=6144) {
        V=v; T=t; D=d; H=h; L=l; FF=ff; hd=d/h;
    }
};

// ============================================================
// LAYER WEIGHTS
// ============================================================
struct CPULayerWeights {
    Mat ln1_w, ln1_b, ln2_w, ln2_b;
    Mat wq, bq, wk, bk, wv, bv, wo, bo;
    Mat w1, b1, w2, b2;
    // Gradients
    Mat g_ln1_w, g_ln1_b, g_ln2_w, g_ln2_b;
    Mat g_wq, g_bq, g_wk, g_bk, g_wv, g_bv, g_wo, g_bo;
    Mat g_w1, g_b1, g_w2, g_b2;
    // Adam moments
    Mat m_ln1_w, m_ln1_b, m_ln2_w, m_ln2_b;
    Mat m_wq, m_bq, m_wk, m_bk, m_wv, m_bv, m_wo, m_bo;
    Mat m_w1, m_b1, m_w2, m_b2;
    Mat v_ln1_w, v_ln1_b, v_ln2_w, v_ln2_b;
    Mat v_wq, v_bq, v_wk, v_bk, v_wv, v_bv, v_wo, v_bo;
    Mat v_w1, v_b1, v_w2, v_b2;
};

// ============================================================
// MODEL WEIGHTS
// ============================================================
struct CPUModelWeights {
    Mat token_emb, pos_emb;
    Mat ln_f_w, ln_f_b;
    Mat lm_w, lm_b;
    // Gradients
    Mat g_token_emb, g_pos_emb;
    Mat g_ln_f_w, g_ln_f_b;
    Mat g_lm_w, g_lm_b;
    // Adam moments
    Mat m_token_emb, m_pos_emb, v_token_emb, v_pos_emb;
    Mat m_ln_f_w, m_ln_f_b, v_ln_f_w, v_ln_f_b;
    Mat m_lm_w, m_lm_b, v_lm_w, v_lm_b;
    std::vector<CPULayerWeights> layers;
};

// ============================================================
// ACTIVATIONS
// ============================================================
struct CPUActivations {
    Mat x_emb;
    struct LayerActs {
        Mat h1, q, k, v, att, att_p, ao, x1, h2, fi;
        Mat ln1_mean, ln1_inv_std;
        Mat ln2_mean, ln2_inv_std;
    };
    std::vector<LayerActs> layers;
    Mat hf, logits;
    Mat ln_f_mean, ln_f_inv_std;
};

// ============================================================
// CPU TRANSFORMER
// ============================================================
struct CPUTransformer {
    CPUConfig cfg;
    CPUModelWeights w;
    CPUActivations act;
    std::map<unsigned char, int> char_to_id;
    std::map<int, unsigned char> id_to_char;

    void init(const CPUConfig &config);
    void allocate_weights();
    void init_weights();
    void allocate_activations(int max_BT);

    float forward(const std::vector<int> &tokens, const std::vector<int> *targets = nullptr);
    void backward(const std::vector<int> &tokens, const std::vector<int> &targets, float loss_scale);
    void optimizer_step(int t, float lr, float b1, float b2, float eps, float wd);
    void clip_gradients(float max_norm);

    std::string generate(const std::string &seed, int max_chars, float temp = 0.7f, int topk = 40);
    void save(const char *path);
    void load(const char *path);
};
