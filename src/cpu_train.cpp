// ============================================================
// RUBIDIUM - CPU Training Engine + LoRA Fine-Tuning
// Main entry point
// ============================================================
#include "cpu_model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

// ============================================================
// LOAD CORPUS
// ============================================================
std::string load_corpus(const std::string &path) {
    std::string corpus;
    for (const auto &entry : fs::directory_iterator(path)) {
        if (entry.path().extension() == ".txt") {
            FILE *f = fopen(entry.path().string().c_str(), "rb");
            if (!f) continue;
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            std::string s(sz, 0);
            fread(&s[0], 1, sz, f);
            fclose(f);
            if (!corpus.empty()) corpus += "\n";
            corpus += s;
            printf("Loaded: %s (%ld chars)\n", entry.path().filename().string().c_str(), sz);
        }
    }
    printf("Corpus: %zu chars (~%d tokens)\n", corpus.size(), (int)(corpus.size() / 4));
    return corpus;
}

// ============================================================
// BUILD VOCAB
// ============================================================
void build_vocab(const std::string &text,
                 std::map<unsigned char, int> &c2i,
                 std::map<int, unsigned char> &i2c) {
    for (unsigned char c : text) {
        if (c2i.find(c) == c2i.end()) {
            int id = c2i.size();
            c2i[c] = id;
            i2c[id] = c;
        }
    }
}

// ============================================================
// MAIN - TRAINING
// ============================================================
int main(int argc, char **argv) {
    printf("============================================================\n");
    printf("RUBIDIUM CPU - Parallel Training Engine (OpenMP)\n");
    printf("============================================================\n");

    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    printf("Threads: %d\n", num_threads);

    // Config
    CPUConfig cfg;
    cfg.init(32000, 512, 1536, 24, 10, 6144);

    int BS = 4, max_steps = 200000;
    float lr = 3e-4f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.1f;
    int warmup = 6000;
    float gc = 1.0f;

    // Mode: train or finetune
    std::string mode = "train";
    std::string corpus_path = "data";
    std::string load_path = "";
    if (argc > 1) mode = argv[1];
    if (argc > 2) corpus_path = argv[2];
    if (argc > 3) load_path = argv[3];

    // Load corpus
    printf("\n--- Loading corpus ---\n");
    std::string full_text = load_corpus(corpus_path);
    if (full_text.empty()) {
        fprintf(stderr, "No corpus found in %s\n", corpus_path.c_str());
        return 1;
    }

    // Build vocab
    std::map<unsigned char, int> c2i;
    std::map<int, unsigned char> i2c;
    build_vocab(full_text, c2i, i2c);
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
    CPUTransformer model;
    if (!load_path.empty()) {
        printf("\n--- Loading pre-trained model ---\n");
        model.load(load_path.c_str());
        // Override vocab if needed
        if (V > model.cfg.V) model.cfg.V = V;
    } else {
        model.init(cfg);
    }
    model.allocate_activations(BS * cfg.T);
    model.char_to_id = c2i;
    model.id_to_char = i2c;

    // Count params
    long long tp = (long long)cfg.V * cfg.D + cfg.T * cfg.D;
    for (int l = 0; l < cfg.L; l++)
        tp += 4 * (long long)cfg.D * cfg.D + 8 * cfg.D + 2 * (long long)cfg.D * cfg.FF + 2 * cfg.FF;
    tp += 2 * cfg.D + (long long)cfg.V * cfg.D + cfg.V;
    printf("Parameters: %.1fM\n", tp / 1e6);
    printf("Training: %d steps, BS=%d\n", max_steps, BS);
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

        // Micro-batch (simplified: 1 sample at a time for CPU memory)
        for (int b = 0; b < BS; b++) {
            int idx = rand() % (n - cfg.T - 1);
            std::vector<int> tokens(data.begin() + idx, data.begin() + idx + cfg.T);
            std::vector<int> targets(data.begin() + idx + 1, data.begin() + idx + 1 + cfg.T);

            float loss = model.forward(tokens, &targets);
            model.backward(tokens, targets, 1.0f / BS);
            loss_acc += loss;
        }

        // Gradient clipping
        model.clip_gradients(gc);

        // Optimizer step
        model.optimizer_step(step, lr_t, b1, b2, eps, wd);

        float avg_loss = loss_acc / BS;
        smooth_loss = (step == 1) ? avg_loss : 0.98f * smooth_loss + 0.02f * avg_loss;

        if (step % 10 == 0 || step == max_steps) {
            double elapsed = (clock() - t0) / CLOCKS_PER_SEC;
            double sps = step / elapsed;
            double eta = (max_steps - step) / sps / 60.0;
            printf("Step %d/%d | loss: %.4f | lr: %.2e | %.3f steps/s | ETA: %.0fmin\n",
                   step, max_steps, smooth_loss, lr_t, sps, eta);
        }

        // Checkpoint
        if (step % 5000 == 0) {
            char path[256];
            sprintf(path, "checkpoints/model_step_%d.bin", step);
            fs::create_directories("checkpoints");
            model.save(path);
        }

        // Quick test
        if (step % 1000 == 0) {
            printf("\n--- Test step %d ---\n", step);
            std::vector<std::string> seeds = {"Hola", "Que es"};
            for (auto &seed : seeds) {
                std::string gen = model.generate(seed, 60, 0.7f, 40);
                printf("  '%s' -> '%s'\n", seed.c_str(), gen.c_str());
            }
            printf("\n");
        }
    }

    double total_time = (clock() - t0) / CLOCKS_PER_SEC;
    printf("\nTraining complete: %.1f min (%.3f steps/s)\n", total_time / 60, max_steps / total_time);

    model.save("model_final.bin");

    printf("\n--- Final Test ---\n");
    std::vector<std::string> seeds = {"Hola", "Buenos dias", "Que es", "Como"};
    for (auto &seed : seeds) {
        std::string gen = model.generate(seed, 120, 0.7f, 40);
        printf("'%s' -> '%s'\n", seed.c_str(), gen.c_str());
    }

    return 0;
}
