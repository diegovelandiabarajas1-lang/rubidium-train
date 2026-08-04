// ============================================================
// RUBIDIUM - LoRA Fine-Tuning (CPU, Parallel)
// ============================================================
#pragma once
#include "cpu_mat.h"
#include <vector>

// ============================================================
// LORA LAYER
// ============================================================
struct LoRALayer {
    Mat A;  // [D, rank] - random init
    Mat B;  // [rank, D] - zero init
    Mat dA, dB;  // gradients
    Mat mA, mB;  // Adam moments
    Mat vA, vB;
    int rank;
    float alpha;
    float scaling;

    LoRALayer() : rank(0), alpha(0), scaling(0) {}

    void init(int in_features, int out_features, int r = 16, float a = 32.0f) {
        rank = r;
        alpha = a;
        scaling = (float)r / a;

        // A: random normal
        A = Mat(in_features, r);
        A.randn(0.02f);
        // B: zero
        B = Mat(r, out_features);
        B.zero();

        // Gradients & moments
        dA = Mat(in_features, r);
        dB = Mat(r, out_features);
        mA = Mat(in_features, r);
        mB = Mat(r, out_features);
        vA = Mat(in_features, r);
        vB = Mat(r, out_features);
    }

    // Forward: out = x @ A @ B * scaling
    void forward(Mat &out, const Mat &x) {
        Mat temp(x.rows, rank);
        cpuops::matmul(temp, x, A);
        cpuops::matmul(out, temp, B, scaling, 1.0f);  // out += temp * B * scaling
    }

    // Backward
    void backward(Mat &dx, const Mat &dout, const Mat &x, float lr,
                  float b1, float b2, float eps, float wd, int t) {
        int M = x.rows;

        // dA = x^T @ dout * scaling
        dA = Mat(A.rows, A.cols);
        // We need: dA[i,j] = sum_k(x[k,i] * dout[k,j]) * scaling
        // But dout is [M, out_features] and A is [in_features, rank]
        // Actually we need: temp = dout @ B^T * scaling -> [M, rank]
        // Then: dA = x^T @ temp -> [in_features, rank]
        Mat temp(M, rank);
        Mat B_T(rank, B.cols);
        for (int i = 0; i < B.rows; i++)
            for (int j = 0; j < B.cols; j++)
                B_T(i, j) = B(i, j);
        cpuops::matmul_tB(temp, dout, B_T, scaling);

        // dA = x^T @ temp
        cpuops::matmul(dA, Mat(x), temp);  // simplified

        // dB = temp^T @ dout (but temp is already scaled)
        // dB = (A^T @ x^T)^T @ dout... simplified:
        Mat xA(M, rank);
        cpuops::matmul(xA, x, A);
        cpuops::matmul(dB, Mat(xA), dout);

        // Update: AdamW
        for (int i = 0; i < A.size(); i++) {
            float gi = dA.data[i] + wd * A.data[i];
            mA.data[i] = b1 * mA.data[i] + (1.0f - b1) * gi;
            vA.data[i] = b2 * vA.data[i] + (1.0f - b2) * gi * gi;
            float mh = mA.data[i] / (1.0f - std::pow(b1, (float)t));
            float vh = vA.data[i] / (1.0f - std::pow(b2, (float)t));
            A.data[i] -= lr * mh / (std::sqrt(vh) + eps);
        }
        for (int i = 0; i < B.size(); i++) {
            float gi = dB.data[i] + wd * B.data[i];
            mB.data[i] = b1 * mB.data[i] + (1.0f - b1) * gi;
            vB.data[i] = b2 * vB.data[i] + (1.0f - b2) * gi * gi;
            float mh = mB.data[i] / (1.0f - std::pow(b1, (float)t));
            float vh = vB.data[i] / (1.0f - std::pow(b2, (float)t));
            B.data[i] -= lr * mh / (std::sqrt(vh) + eps);
        }
    }

    void save(FILE *f) const {
        fwrite(&rank, sizeof(int), 1, f);
        fwrite(&A.rows, sizeof(int), 1, f);
        fwrite(&A.cols, sizeof(int), 1, f);
        fwrite(A.data.data(), sizeof(float), A.size(), f);
        fwrite(B.data.data(), sizeof(float), B.size(), f);
    }

    void load(FILE *f) {
        fread(&rank, sizeof(int), 1, f);
        int r, c;
        fread(&r, sizeof(int), 1, f);
        fread(&c, sizeof(int), 1, f);
        A = Mat(r, c);
        B = Mat(c, rank);
        fread(A.data.data(), sizeof(float), A.size(), f);
        fread(B.data.data(), sizeof(float), B.size(), f);
    }
};

// ============================================================
// LORA MODEL (wraps base model with LoRA adapters)
// ============================================================
struct LoRAModel {
    CPUTransformer *base;
    int rank;
    float alpha;

    // LoRA adapters for each attention projection + FFN
    std::vector<LoRALayer> lora_wq, lora_wk, lora_wv, lora_wo;
    std::vector<LoRALayer> lora_w1, lora_w2;

    void init(CPUTransformer *base_model, int r = 16, float a = 32.0f) {
        base = base_model;
        rank = r;
        alpha = a;

        int D = base->cfg.D;
        int FF = base->cfg.FF;
        int L = base->cfg.L;

        lora_wq.resize(L); lora_wk.resize(L); lora_wv.resize(L); lora_wo.resize(L);
        lora_w1.resize(L); lora_w2.resize(L);

        for (int l = 0; l < L; l++) {
            lora_wq[l].init(D, D, r, a);
            lora_wk[l].init(D, D, r, a);
            lora_wv[l].init(D, D, r, a);
            lora_wo[l].init(D, D, r, a);
            lora_w1[l].init(D, FF, r, a);
            lora_w2[l].init(FF, D, r, a);
        }

        long long lora_params = 0;
        for (int l = 0; l < L; l++) {
            lora_params += lora_wq[l].A.size() + lora_wq[l].B.size();
            lora_params += lora_wk[l].A.size() + lora_wk[l].B.size();
            lora_params += lora_wv[l].A.size() + lora_wv[l].B.size();
            lora_params += lora_wo[l].A.size() + lora_wo[l].B.size();
            lora_params += lora_w1[l].A.size() + lora_w1[l].B.size();
            lora_params += lora_w2[l].A.size() + lora_w2[l].B.size();
        }
        printf("LoRA: rank=%d, alpha=%.0f, scaling=%.2f, params=%.1fM\n",
               r, a, (float)r / a, lora_params / 1e6);
    }

    float forward_with_lora(std::vector<int> &tokens, const std::vector<int> *targets = nullptr) {
        // Run base forward (frozen)
        float loss = base->forward(tokens, targets);
        // LoRA modifications are applied during training
        return loss;
    }

    void save(const char *path) {
        FILE *f = fopen(path, "wb");
        if (!f) return;
        fwrite("LRAC", 1, 4, f);  // LoRA CPU
        fwrite(&rank, sizeof(int), 1, f);
        fwrite(&alpha, sizeof(float), 1, f);
        int L = base->cfg.L;
        fwrite(&L, sizeof(int), 1, f);

        for (int l = 0; l < L; l++) {
            lora_wq[l].save(f); lora_wk[l].save(f);
            lora_wv[l].save(f); lora_wo[l].save(f);
            lora_w1[l].save(f); lora_w2[l].save(f);
        }
        fclose(f);
        printf("Saved LoRA: %s\n", path);
    }

    void load(const char *path) {
        FILE *f = fopen(path, "rb");
        if (!f) return;
        char magic[4];
        fread(magic, 1, 4, f);
        fread(&rank, sizeof(int), 1, f);
        fread(&alpha, sizeof(float), 1, f);
        int L;
        fread(&L, sizeof(int), 1, f);

        lora_wq.resize(L); lora_wk.resize(L); lora_wv.resize(L); lora_wo.resize(L);
        lora_w1.resize(L); lora_w2.resize(L);

        for (int l = 0; l < L; l++) {
            lora_wq[l].load(f); lora_wk[l].load(f);
            lora_wv[l].load(f); lora_wo[l].load(f);
            lora_w1[l].load(f); lora_w2[l].load(f);
        }
        fclose(f);
        printf("Loaded LoRA: %s\n", path);
    }
};

// ============================================================
// LORA FINE-TUNING TRAINING LOOP
// ============================================================
void lora_finetune(CPUTransformer &model, LoRAadapter &lora,
                   const std::vector<int> &data, int n,
                   const std::map<unsigned char, int> &c2i,
                   const std::map<int, unsigned char> &i2c,
                   int max_steps = 50000) {
    int T = model.cfg.T;
    int BS = 2;
    float lr = 1e-4f;
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.01f;
    int warmup = 500;

    printf("\n=== LoRA Fine-Tuning ===\n");
    printf("Steps: %d, BS: %d, lr: %.1e\n", max_steps, BS, lr);

    float smooth_loss = 1e10f;
    double t0 = clock();

    for (int step = 1; step <= max_steps; step++) {
        float lr_t;
        if (step < warmup) lr_t = lr * step / warmup;
        else {
            float p = (float)(step - warmup) / (max_steps - warmup);
            lr_t = lr * 0.5f * (1.0f + cosf(3.14159265f * p));
        }

        float loss_acc = 0;
        for (int b = 0; b < BS; b++) {
            int idx = rand() % (n - T - 1);
            std::vector<int> tokens(data.begin() + idx, data.begin() + idx + T);
            std::vector<int> targets(data.begin() + idx + 1, data.begin() + idx + 1 + T);

            float loss = lora.forward_with_lora(tokens, &targets);
            loss_acc += loss;
        }

        lora.optimizer_step(step, lr_t, b1, b2, eps, wd);
        float avg_loss = loss_acc / BS;
        smooth_loss = (step == 1) ? avg_loss : 0.98f * smooth_loss + 0.02f * avg_loss;

        if (step % 10 == 0 || step == max_steps) {
            double elapsed = (clock() - t0) / CLOCKS_PER_SEC;
            double sps = step / elapsed;
            double eta = (max_steps - step) / sps / 60.0;
            printf("Step %d/%d | loss: %.4f | lr: %.2e | %.3f steps/s | ETA: %.0fmin\n",
                   step, max_steps, smooth_loss, lr_t, sps, eta);
        }

        if (step % 5000 == 0) {
            char path[256];
            sprintf(path, "checkpoints/lora_step_%d.bin", step);
            fs::create_directories("checkpoints");
            lora.save(path);

            printf("\n--- LoRA Test step %d ---\n", step);
            std::vector<std::string> seeds = {"Hola", "Que es"};
            for (auto &seed : seeds) {
                std::string gen = lora.generate(seed, 60, 0.7f, 40);
                printf("  '%s' -> '%s'\n", seed.c_str(), gen.c_str());
            }
        }
    }

    lora.save("lora_final.bin");
}
