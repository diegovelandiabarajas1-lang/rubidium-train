// ============================================================
// RUBIDIUM TRANSFORMER - CUDA/C++ TRAINING ENGINE
// Main entry point
// ============================================================
#include "model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <glob.h>

// ============================================================
// LOAD CORPUS
// ============================================================
std::string load_corpus(const std::string &path) {
    std::string corpus;
    // Use glob to find .txt files
    glob_t glob_result;
    glob((path + "/*.txt").c_str(), GLOB_NOSORT, nullptr, &glob_result);
    for (size_t i = 0; i < glob_result.gl_pathc; i++) {
        FILE *f = fopen(glob_result.gl_pathv[i], "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::string s(sz, 0);
        fread(&s[0], 1, sz, f);
        fclose(f);
        if (!corpus.empty()) corpus += "\n";
        corpus += s;
        printf("Loaded: %s (%ld chars)\n", glob_result.gl_pathv[i], sz);
    }
    globfree(&glob_result);
    printf("Corpus: %zu chars\n", corpus.size());
    return corpus;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char **argv) {
    printf("============================================================\n");
    printf("RUBIDIUM TRANSFORMER - CUDA/C++ Training Engine\n");
    printf("cuBLAS + cuDNN + Custom CUDA kernels\n");
    printf("============================================================\n");

    // GPU info
    cudaDeviceProp prop;
    CE(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("VRAM: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("SMs: %d\n", prop.multiProcessorCount);

    // Init handles
    init_handles();

    // Config
    ModelConfig cfg;
    cfg.init(32000, 512, 2048, 32, 10, 8192);  // V=32K, T=512

    // Hyperparams
    int BS = 2, GA = 16, max_steps = 200000;
    float lr = 3e-4f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.1f;
    int warmup = 6000;
    float gc = 1.0f;

    // Load corpus
    std::string corpus_path = "data";
    if (argc > 1) corpus_path = argv[1];
    std::string full_text = load_corpus(corpus_path);
    if (full_text.empty()) {
        fprintf(stderr, "No corpus found in %s\n", corpus_path.c_str());
        return 1;
    }

    // Build vocab
    std::map<unsigned char, int> c2i;
    std::map<int, unsigned char> i2c;
    for (unsigned char c : full_text) {
        if (c2i.find(c) == c2i.end()) {
            int id = c2i.size();
            c2i[c] = id;
            i2c[id] = c;
        }
    }
    int V = c2i.size();
    if (V > cfg.V) cfg.V = V;
    printf("Vocab: %d\n", V);

    // Encode
    std::vector<int> data(full_text.size());
    for (size_t i = 0; i < full_text.size(); i++)
        data[i] = c2i[(unsigned char)full_text[i]];
    int n = data.size();
    printf("Tokens: %d\n", n);

    // Init model
    RubidiumTransformer model;
    model.init(cfg);
    model.allocate_activations(BS * cfg.T);
    model.char_to_id = c2i;
    model.id_to_char = i2c;

    // Count params
    long long tp = (long long)cfg.V*cfg.D*2 + cfg.T*cfg.D;
    for (int l = 0; l < cfg.L; l++)
        tp += 4*(long long)cfg.D*cfg.D + 8*cfg.D + 2*(long long)cfg.D*cfg.FF + 2*cfg.FF;
    tp += 2*cfg.D + (long long)cfg.V*cfg.D + cfg.V;
    printf("Parameters: %.1fM\n", tp/1e6);
    printf("Training: %d steps, BS=%d, GA=%d, Eff=%d\n", max_steps, BS, GA, BS*GA);
    printf("------------------------------------------------------------\n");

    // Training loop
    float smooth_loss = 1e10f;
    double t0 = clock();

    for (int step = 1; step <= max_steps; step++) {
        // LR schedule
        float lr_t;
        if (step < warmup) lr_t = lr * step / warmup;
        else {
            float p = (float)(step - warmup) / (max_steps - warmup);
            lr_t = lr * 0.5f * (1.0f + cosf(3.14159265f * p));
        }

        float loss_acc = 0;

        for (int ga = 0; ga < GA; ga++) {
            // Sample batch
            std::vector<int> idx(BS);
            for (auto &i : idx) i = rand() % (n - cfg.T - 1);

            std::vector<int> h_tok(BS * cfg.T), h_tgt(BS * cfg.T);
            for (int b = 0; b < BS; b++)
                for (int t = 0; t < cfg.T; t++) {
                    h_tok[b*cfg.T+t] = data[idx[b]+t];
                    h_tgt[b*cfg.T+t] = data[idx[b]+t+1];
                }

            int *d_tok, *d_tgt;
            CE(cudaMalloc(&d_tok, BS*cfg.T*sizeof(int)));
            CE(cudaMalloc(&d_tgt, BS*cfg.T*sizeof(int)));
            CE(cudaMemcpy(d_tok, h_tok.data(), BS*cfg.T*sizeof(int), cudaMemcpyHostToDevice));
            CE(cudaMemcpy(d_tgt, h_tgt.data(), BS*cfg.T*sizeof(int), cudaMemcpyHostToDevice));

            // Forward
            float loss = model.forward(d_tok, d_tgt, BS, cfg.T);

            // Backward
            model.backward(d_tok, d_tgt, BS, cfg.T, 1.0f / (BS * GA));

            loss_acc += loss;

            CE(cudaFree(d_tok));
            CE(cudaFree(d_tgt));
        }

        // Gradient clipping
        model.clip_gradients(gc);

        // Optimizer step
        model.optimizer_step(step, lr_t, b1, b2, eps, wd);

        cudaDeviceSynchronize();

        float avg_loss = loss_acc / GA;
        smooth_loss = (step == 1) ? avg_loss : 0.98f * smooth_loss + 0.02f * avg_loss;

        if (step % 100 == 0 || step == max_steps) {
            double elapsed = (clock() - t0) / CLOCKS_PER_SEC;
            double sps = step / elapsed;
            double eta = (max_steps - step) / sps / 60.0;
            printf("Step %d/%d | loss: %.4f | lr: %.2e | %.1f steps/s | ETA: %.0fmin\n",
                   step, max_steps, smooth_loss, lr_t, sps, eta);
        }

        // Checkpoint
        if (step % 5000 == 0) {
            char path[256];
            sprintf(path, "checkpoints/model_step_%d.bin", step);
            model.save(path);
        }
    }

    double total_time = (clock() - t0) / CLOCKS_PER_SEC;
    printf("\nTraining complete: %.1f min (%.1f steps/s)\n", total_time/60, max_steps/total_time);

    // Save final model
    model.save("model_final.bin");

    // Quick test
    printf("\n--- Quick Test ---\n");
    std::vector<std::string> seeds = {"Hola", "Buenos dias", "Quien eres", "Que puedes hacer"};
    for (auto &seed : seeds) {
        std::string gen = model.generate(seed, 120, 0.7f, 40);
        printf("%s -> %s\n", seed.c_str(), gen.c_str());
    }

    model.free_all();
    return 0;
}
