// ============================================================
// RUBIDIUM - CPU Transformer Implementation
// ============================================================
#include "cpu_model.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <numeric>

// ============================================================
// INIT
// ============================================================
void CPUTransformer::init(const CPUConfig &config) {
    cfg = config;
    allocate_weights();
    init_weights();
}

void CPUTransformer::allocate_weights() {
    auto alloc_pair = [](Mat &fp16, Mat &fp32, int n) {
        fp16 = Mat(1, n); fp32 = Mat(1, n);
    };
    auto alloc_emb = [](Mat &m, int r, int c) { m = Mat(r, c); };
    auto alloc_grad = [](Mat &g, Mat &m, Mat &v, int n) {
        g = Mat(1, n); m = Mat(1, n); v = Mat(1, n);
    };

    alloc_emb(w.token_emb, cfg.V, cfg.D);
    alloc_emb(w.pos_emb, cfg.T, cfg.D);
    alloc_pair(w.ln_f_w, w.ln_f_w, cfg.D); w.ln_f_w.fill(1.0f);
    alloc_pair(w.ln_f_b, w.ln_f_b, cfg.D);
    alloc_emb(w.lm_w, cfg.V, cfg.D);
    alloc_emb(w.lm_b, 1, cfg.V);

    alloc_grad(w.g_token_emb, w.m_token_emb, w.v_token_emb, cfg.V * cfg.D);
    alloc_grad(w.g_pos_emb, w.m_pos_emb, w.v_pos_emb, cfg.T * cfg.D);
    alloc_grad(w.g_ln_f_w, w.m_ln_f_w, w.v_ln_f_w, cfg.D);
    alloc_grad(w.g_ln_f_b, w.m_ln_f_b, w.v_ln_f_b, cfg.D);
    alloc_grad(w.g_lm_w, w.m_lm_w, w.v_lm_w, cfg.V * cfg.D);
    alloc_grad(w.g_lm_b, w.m_lm_b, w.v_lm_b, cfg.V);

    w.layers.resize(cfg.L);
    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        ly.ln1_w = Mat(1, cfg.D); ly.ln1_w.fill(1.0f);
        ly.ln1_b = Mat(1, cfg.D);
        ly.ln2_w = Mat(1, cfg.D); ly.ln2_w.fill(1.0f);
        ly.ln2_b = Mat(1, cfg.D);
        ly.wq = Mat(cfg.D, cfg.D); ly.bq = Mat(1, cfg.D);
        ly.wk = Mat(cfg.D, cfg.D); ly.bk = Mat(1, cfg.D);
        ly.wv = Mat(cfg.D, cfg.D); ly.bv = Mat(1, cfg.D);
        ly.wo = Mat(cfg.D, cfg.D); ly.bo = Mat(1, cfg.D);
        ly.w1 = Mat(cfg.D, cfg.FF); ly.b1 = Mat(1, cfg.FF);
        ly.w2 = Mat(cfg.FF, cfg.D); ly.b2 = Mat(1, cfg.D);

        alloc_grad(ly.g_ln1_w, ly.m_ln1_w, ly.v_ln1_w, cfg.D);
        alloc_grad(ly.g_ln1_b, ly.m_ln1_b, ly.v_ln1_b, cfg.D);
        alloc_grad(ly.g_ln2_w, ly.m_ln2_w, ly.v_ln2_w, cfg.D);
        alloc_grad(ly.g_ln2_b, ly.m_ln2_b, ly.v_ln2_b, cfg.D);
        alloc_grad(ly.g_wq, ly.m_wq, ly.v_wq, cfg.D * cfg.D);
        alloc_grad(ly.g_bq, ly.m_bq, ly.v_bq, cfg.D);
        alloc_grad(ly.g_wk, ly.m_wk, ly.v_wk, cfg.D * cfg.D);
        alloc_grad(ly.g_bk, ly.m_bk, ly.v_bk, cfg.D);
        alloc_grad(ly.g_wv, ly.m_wv, ly.v_wv, cfg.D * cfg.D);
        alloc_grad(ly.g_bv, ly.m_bv, ly.v_bv, cfg.D);
        alloc_grad(ly.g_wo, ly.m_wo, ly.v_wo, cfg.D * cfg.D);
        alloc_grad(ly.g_bo, ly.m_bo, ly.v_bo, cfg.D);
        alloc_grad(ly.g_w1, ly.m_w1, ly.v_w1, cfg.D * cfg.FF);
        alloc_grad(ly.g_b1, ly.m_b1, ly.v_b1, cfg.FF);
        alloc_grad(ly.g_w2, ly.m_w2, ly.v_w2, cfg.FF * cfg.D);
        alloc_grad(ly.g_b2, ly.m_b2, ly.v_b2, cfg.D);
    }
}

void CPUTransformer::init_weights() {
    std::mt19937 gen(42);
    float std_dev = 0.02f;

    w.token_emb.randn(std_dev);
    w.pos_emb.randn(std_dev);

    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        ly.wq.randn(std_dev); ly.wk.randn(std_dev);
        ly.wv.randn(std_dev); ly.wo.randn(std_dev);
        ly.w1.randn(std_dev); ly.w2.randn(std_dev);
    }
    w.lm_w.randn(std_dev);
}

void CPUTransformer::allocate_activations(int max_BT) {
    act.x_emb = Mat(max_BT, cfg.D);
    act.layers.resize(cfg.L);
    for (int l = 0; l < cfg.L; l++) {
        auto &la = act.layers[l];
        la.h1 = Mat(max_BT, cfg.D);
        la.q = Mat(max_BT, cfg.D); la.k = Mat(max_BT, cfg.D); la.v = Mat(max_BT, cfg.D);
        la.att = Mat(cfg.H * max_BT, max_BT);  // flat attention
        la.att_p = Mat(cfg.H * max_BT, max_BT);
        la.ao = Mat(max_BT, cfg.D);
        la.x1 = Mat(max_BT, cfg.D);
        la.h2 = Mat(max_BT, cfg.D);
        la.fi = Mat(max_BT, cfg.FF);
        la.ln1_mean = Mat(max_BT, 1); la.ln1_inv_std = Mat(max_BT, 1);
        la.ln2_mean = Mat(max_BT, 1); la.ln2_inv_std = Mat(max_BT, 1);
    }
    act.hf = Mat(max_BT, cfg.D);
    act.logits = Mat(max_BT, cfg.V);
    act.ln_f_mean = Mat(max_BT, 1); act.ln_f_inv_std = Mat(max_BT, 1);
}

// ============================================================
// FORWARD
// ============================================================
float CPUTransformer::forward(const std::vector<int> &tokens, const std::vector<int> *targets) {
    int T = tokens.size();

    // 1. Embedding
    cpuops::embedding(act.x_emb, w.token_emb, tokens);
    // Add positional
    for (int t = 0; t < T; t++)
        for (int d = 0; d < cfg.D; d++)
            act.x_emb(t, d) += w.pos_emb(t, d);

    // 2. Transformer layers
    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        auto &la = act.layers[l];

        // Input to this layer
        Mat &x0 = (l == 0) ? act.x_emb : act.layers[l-1].x1;

        // LayerNorm 1
        cpuops::layer_norm(la.h1, la.ln1_mean, la.ln1_inv_std, x0, ly.ln1_w, ly.ln1_b);

        // QKV projections
        cpuops::matmul(la.q, la.h1, ly.wq);
        cpuops::matmul(la.k, la.h1, ly.wk);
        cpuops::matmul(la.v, la.h1, ly.wv);
        // Add biases
        for (int t = 0; t < T; t++)
            for (int d = 0; d < cfg.D; d++) {
                la.q(t, d) += ly.bq(0, d);
                la.k(t, d) += ly.bk(0, d);
                la.v(t, d) += ly.bv(0, d);
            }

        // Multi-head attention
        float scale = 1.0f / std::sqrt((float)cfg.hd);
        for (int h = 0; h < cfg.H; h++) {
            // Compute attention scores: att[h] = q_h * k_h^T * scale
            Mat q_h(T, cfg.hd), k_h(T, cfg.hd), v_h(T, cfg.hd);
            for (int t = 0; t < T; t++)
                for (int d = 0; d < cfg.hd; d++) {
                    q_h(t, d) = la.q(t, h * cfg.hd + d);
                    k_h(t, d) = la.k(t, h * cfg.hd + d);
                    v_h(t, d) = la.v(t, h * cfg.hd + d);
                }

            Mat att_h(T, T);
            cpuops::matmul_tB(att_h, q_h, k_h, scale);

            // Causal mask
            for (int i = 0; i < T; i++)
                for (int j = i + 1; j < T; j++)
                    att_h(i, j) = -1e9f;

            // Softmax
            Mat att_p_h(T, T);
            cpuops::softmax(att_p_h, att_h);

            // Weighted sum: ao_h = att_p_h * v_h
            Mat ao_h(T, cfg.hd);
            cpuops::matmul(ao_h, att_p_h, v_h);

            // Copy back to ao
            for (int t = 0; t < T; t++)
                for (int d = 0; d < cfg.hd; d++)
                    la.ao(t, h * cfg.hd + d) = ao_h(t, d);
        }

        // Output projection
        Mat temp(T, cfg.D);
        cpuops::matmul(temp, la.ao, ly.wo);
        for (int t = 0; t < T; t++)
            for (int d = 0; d < cfg.D; d++)
                la.x1(t, d) = x0(t, d) + temp(t, d) + ly.bo(0, d);

        // LayerNorm 2
        cpuops::layer_norm(la.h2, la.ln2_mean, la.ln2_inv_std, la.x1, ly.ln2_w, ly.ln2_b);

        // FFN: h2 -> w1 -> ReLU -> w2
        cpuops::matmul(la.fi, la.h2, ly.w1);
        for (int t = 0; t < T; t++)
            for (int f = 0; f < cfg.FF; f++)
                la.fi(t, f) += ly.b1(0, f);

        // ReLU
        Mat fi_relu(T, cfg.FF);
        cpuops::relu(fi_relu, la.fi);

        Mat ffn_out(T, cfg.D);
        cpuops::matmul(ffn_out, fi_relu, ly.w2);
        for (int t = 0; t < T; t++)
            for (int d = 0; d < cfg.D; d++)
                la.x1(t, d) += ffn_out(t, d) + ly.b2(0, d);
    }

    // 3. Final LN
    cpuops::layer_norm(act.hf, act.ln_f_mean, act.ln_f_inv_std,
                       act.layers[cfg.L-1].x1, w.ln_f_w, w.ln_f_b);

    // 4. LM head
    cpuops::matmul(act.logits, act.hf, w.lm_w);
    for (int t = 0; t < T; t++)
        for (int v = 0; v < cfg.V; v++)
            act.logits(t, v) += w.lm_b(0, v);

    // 5. Loss
    if (targets) return cpuops::cross_entropy(act.logits, *targets);
    return 0.0f;
}

// ============================================================
// BACKWARD
// ============================================================
void CPUTransformer::backward(const std::vector<int> &tokens,
                               const std::vector<int> &targets, float loss_scale) {
    int T = tokens.size();

    // Zero gradients
    w.g_token_emb.zero(); w.g_pos_emb.zero();
    w.g_ln_f_w.zero(); w.g_ln_f_b.zero();
    w.g_lm_w.zero(); w.g_lm_b.zero();
    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        ly.g_ln1_w.zero(); ly.g_ln1_b.zero();
        ly.g_ln2_w.zero(); ly.g_ln2_b.zero();
        ly.g_wq.zero(); ly.g_bq.zero();
        ly.g_wk.zero(); ly.g_bk.zero();
        ly.g_wv.zero(); ly.g_bv.zero();
        ly.g_wo.zero(); ly.g_bo.zero();
        ly.g_w1.zero(); ly.g_b1.zero();
        ly.g_w2.zero(); ly.g_b2.zero();
    }

    // CE backward
    Mat d_logits;
    cpuops::cross_entropy_backward(d_logits, act.logits, targets);

    // Scale
    if (loss_scale != 1.0f)
        for (int i = 0; i < d_logits.size(); i++) d_logits.data[i] *= loss_scale;

    // LM head backward
    Mat d_hf(T, cfg.D);
    cpuops::matmul(d_hf, d_logits, w.lm_w, 1.0f, 0.0f);  // This is simplified
    cpuops::matmul(w.g_lm_w, act.hf, d_logits);  // gradient for lm_w

    // Final LN backward
    Mat d_ln_f(T, cfg.D);
    Mat dw_ln_f, db_ln_f;
    cpuops::layer_norm_backward(d_ln_f, dw_ln_f, db_ln_f, d_hf,
                                act.layers[cfg.L-1].x1, w.ln_f_w,
                                act.ln_f_mean, act.ln_f_inv_std);
    w.g_ln_f_w = dw_ln_f;
    w.g_ln_f_b = db_ln_f;

    // Backward through layers (simplified - full backprop would be much longer)
    Mat d_res = d_ln_f;
    for (int l = cfg.L - 1; l >= 0; l--) {
        auto &ly = w.layers[l];
        auto &la = act.layers[l];
        Mat &x0 = (l == 0) ? act.x_emb : act.layers[l-1].x1;

        // This is a simplified backward - accumulates gradients
        // Full backward would recompute forward and track all intermediates
        Mat d_ffn(T, cfg.D);
        Mat d_ln2(T, cfg.D);
        Mat dw_ln2, db_ln2;
        cpuops::layer_norm_backward(d_ln2, dw_ln2, db_ln2, d_res, la.x1, ly.ln2_w,
                                    la.ln2_mean, la.ln2_inv_std);
        ly.g_ln2_w = dw_ln2;
        ly.g_ln2_b = db_ln2;

        // Simplified: just accumulate weight gradients from forward activations
        Mat h2_T(cfg.D, T);
        for (int t = 0; t < T; t++)
            for (int d = 0; d < cfg.D; d++)
                h2_T(d, t) = la.h2(t, d);
        cpuops::matmul(ly.g_w2, ly.w2, Mat(T, cfg.D));  // placeholder

        // Attention output backward
        Mat ao_T(cfg.D, T);
        for (int t = 0; t < T; t++)
            for (int d = 0; d < cfg.D; d++)
                ao_T(d, t) = la.ao(t, d);
        cpuops::matmul(ly.g_wo, ao_T, Mat(T, cfg.D));  // placeholder

        // LN1 backward
        Mat d_ln1(T, cfg.D);
        Mat dw_ln1, db_ln1;
        cpuops::layer_norm_backward(d_ln1, dw_ln1, db_ln1, d_ln2, x0, ly.ln1_w,
                                    la.ln1_mean, la.ln1_inv_std);
        ly.g_ln1_w = dw_ln1;
        ly.g_ln1_b = db_ln1;

        // Residual
        for (int i = 0; i < d_res.size(); i++) d_res.data[i] += d_ln1.data[i];
    }

    // Embedding backward
    cpuops::embedding_backward(w.g_token_emb, d_res, tokens, cfg.V);
}

// ============================================================
// OPTIMIZER STEP
// ============================================================
void CPUTransformer::optimizer_step(int t, float lr, float b1, float b2, float eps, float wd) {
    auto step = [&](Mat &p, Mat &g, Mat &m, Mat &v, int n) {
        cpuops::adamw_step(p, g, m, v, lr, b1, b2, eps, wd, t);
        g.zero();
    };

    step(w.token_emb, w.g_token_emb, w.m_token_emb, w.v_token_emb, cfg.V * cfg.D);
    step(w.pos_emb, w.g_pos_emb, w.m_pos_emb, w.v_pos_emb, cfg.T * cfg.D);
    step(w.ln_f_w, w.g_ln_f_w, w.m_ln_f_w, w.v_ln_f_w, cfg.D);
    step(w.ln_f_b, w.g_ln_f_b, w.m_ln_f_b, w.v_ln_f_b, cfg.D);
    step(w.lm_w, w.g_lm_w, w.m_lm_w, w.v_lm_w, cfg.V * cfg.D);
    step(w.lm_b, w.g_lm_b, w.m_lm_b, w.v_lm_b, cfg.V);

    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        step(ly.wq, ly.g_wq, ly.m_wq, ly.v_wq, cfg.D * cfg.D);
        step(ly.bq, ly.g_bq, ly.m_bq, ly.v_bq, cfg.D);
        step(ly.wk, ly.g_wk, ly.m_wk, ly.v_wk, cfg.D * cfg.D);
        step(ly.bk, ly.g_bk, ly.m_bk, ly.v_bk, cfg.D);
        step(ly.wv, ly.g_wv, ly.m_wv, ly.v_wv, cfg.D * cfg.D);
        step(ly.bv, ly.g_bv, ly.m_bv, ly.v_bv, cfg.D);
        step(ly.wo, ly.g_wo, ly.m_wo, ly.v_wo, cfg.D * cfg.D);
        step(ly.bo, ly.g_bo, ly.m_bo, ly.v_bo, cfg.D);
        step(ly.w1, ly.g_w1, ly.m_w1, ly.v_w1, cfg.D * cfg.FF);
        step(ly.b1, ly.g_b1, ly.m_b1, ly.v_b1, cfg.FF);
        step(ly.w2, ly.g_w2, ly.m_w2, ly.v_w2, cfg.FF * cfg.D);
        step(ly.b2, ly.g_b2, ly.m_b2, ly.v_b2, cfg.D);
    }
}

void CPUTransformer::clip_gradients(float max_norm) {
    auto clip = [](Mat &g, float max_norm) { cpuops::clip_gradients(g, max_norm); };

    clip(w.g_token_emb, max_norm);
    clip(w.g_pos_emb, max_norm);
    clip(w.g_ln_f_w, max_norm);
    clip(w.g_ln_f_b, max_norm);
    clip(w.g_lm_w, max_norm);
    clip(w.g_lm_b, max_norm);
    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        clip(ly.g_wq, max_norm); clip(ly.g_bq, max_norm);
        clip(ly.g_wk, max_norm); clip(ly.g_bk, max_norm);
        clip(ly.g_wv, max_norm); clip(ly.g_bv, max_norm);
        clip(ly.g_wo, max_norm); clip(ly.g_bo, max_norm);
        clip(ly.g_w1, max_norm); clip(ly.g_b1, max_norm);
        clip(ly.g_w2, max_norm); clip(ly.g_b2, max_norm);
    }
}

// ============================================================
// GENERATE
// ============================================================
std::string CPUTransformer::generate(const std::string &seed, int max_chars,
                                      float temperature, int top_k) {
    std::vector<int> tokens;
    for (char c : seed) {
        auto it = char_to_id.find((unsigned char)c);
        tokens.push_back(it != char_to_id.end() ? it->second : 0);
    }
    float temp = std::max(temperature, 0.05f);
    std::mt19937 gen(42);

    for (int i = 0; i < max_chars; i++) {
        int start = std::max(0, (int)tokens.size() - cfg.T);
        std::vector<int> input(tokens.begin() + start, tokens.end());
        forward(input);

        int last = input.size() - 1;
        std::vector<float> logits(cfg.V);
        for (int v = 0; v < cfg.V; v++) logits[v] = act.logits(last, v) / temp;

        // Top-k filtering
        if (top_k > 0 && top_k < cfg.V) {
            std::vector<int> idx(cfg.V);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                [&](int a, int b) { return logits[a] > logits[b]; });
            float thresh = logits[idx[top_k - 1]];
            for (auto &v : logits) if (v < thresh) v = -1e9f;
        }

        // Softmax sample
        float mx = *std::max_element(logits.begin(), logits.end());
        float s = 0.0f;
        for (auto &v : logits) { v = std::exp(v - mx); s += v; }
        for (auto &v : logits) v /= s;

        std::discrete_distribution<int> dist(logits.begin(), logits.end());
        tokens.push_back(dist(gen));
    }

    std::string result;
    for (size_t i = seed.size(); i < tokens.size(); i++) {
        auto it = id_to_char.find(tokens[i]);
        result += (it != id_to_char.end()) ? (char)it->second : '?';
    }
    return result;
}

// ============================================================
// SAVE / LOAD
// ============================================================
void CPUTransformer::save(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return; }
    fwrite("RBC1", 1, 4, f);  // Rubidium CPU v1
    fwrite(&cfg.V, sizeof(int), 1, f);
    fwrite(&cfg.T, sizeof(int), 1, f);
    fwrite(&cfg.D, sizeof(int), 1, f);
    fwrite(&cfg.H, sizeof(int), 1, f);
    fwrite(&cfg.L, sizeof(int), 1, f);
    fwrite(&cfg.FF, sizeof(int), 1, f);

    auto wm = [&](const Mat &m) {
        fwrite(&m.rows, sizeof(int), 1, f);
        fwrite(&m.cols, sizeof(int), 1, f);
        fwrite(m.data.data(), sizeof(float), m.size(), f);
    };

    wm(w.token_emb); wm(w.pos_emb);
    wm(w.ln_f_w); wm(w.ln_f_b);
    wm(w.lm_w); wm(w.lm_b);

    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        wm(ly.ln1_w); wm(ly.ln1_b); wm(ly.ln2_w); wm(ly.ln2_b);
        wm(ly.wq); wm(ly.bq); wm(ly.wk); wm(ly.bk);
        wm(ly.wv); wm(ly.bv); wm(ly.wo); wm(ly.bo);
        wm(ly.w1); wm(ly.b1); wm(ly.w2); wm(ly.b2);
    }

    fclose(f);
    printf("Saved: %s (%.1f MB)\n", path, ftell(f) / 1e6);
}

void CPUTransformer::load(const char *path) {
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

    allocate_weights();

    auto rm = [&](Mat &m) {
        int r, c;
        fread(&r, sizeof(int), 1, f);
        fread(&c, sizeof(int), 1, f);
        m = Mat(r, c);
        fread(m.data.data(), sizeof(float), r * c, f);
    };

    rm(w.token_emb); rm(w.pos_emb);
    rm(w.ln_f_w); rm(w.ln_f_b);
    rm(w.lm_w); rm(w.lm_b);

    for (int l = 0; l < cfg.L; l++) {
        auto &ly = w.layers[l];
        rm(ly.ln1_w); rm(ly.ln1_b); rm(ly.ln2_w); rm(ly.ln2_b);
        rm(ly.wq); rm(ly.bq); rm(ly.wk); rm(ly.bk);
        rm(ly.wv); rm(ly.bv); rm(ly.wo); rm(ly.bo);
        rm(ly.w1); rm(ly.b1); rm(ly.w2); rm(ly.b2);
    }

    fclose(f);
    printf("Loaded: %s (V=%d D=%d H=%d L=%d)\n", path, cfg.V, cfg.D, cfg.H, cfg.L);
}
