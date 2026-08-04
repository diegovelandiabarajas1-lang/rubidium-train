// ============================================================
// RUBIDIUM - LoRA Fine-Tuning CPU (Main Entry)
// ============================================================
#include "cpu_model.h"
#include "cpu_lora.h"
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
    printf("Corpus: %zu chars\n", corpus.size());
    return corpus;
}

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
// MAIN - LORA FINE-TUNING
// ============================================================
int main(int argc, char **argv) {
    printf("============================================================\n");
    printf("RUBIDIUM CPU - LoRA Fine-Tuning\n");
    printf("============================================================\n");

    int num_threads = omp_get_max_threads();
    printf("Threads: %d\n", num_threads);

    // Args
    if (argc < 3) {
        printf("Usage: %s <base_model.bin> <corpus_dir> [lora_rank] [steps]\n", argv[0]);
        printf("  base_model.bin  - Pre-trained model from cpu_train\n");
        printf("  corpus_dir      - Directory with .txt files\n");
        printf("  lora_rank       - LoRA rank (default: 16)\n");
        printf("  steps           - Fine-tuning steps (default: 50000)\n");
        return 1;
    }

    std::string base_path = argv[1];
    std::string corpus_path = argv[2];
    int lora_rank = argc > 3 ? atoi(argv[3]) : 16;
    int max_steps = argc > 4 ? atoi(argv[4]) : 50000;

    // Load base model
    printf("\n--- Loading base model ---\n");
    CPUTransformer model;
    model.load(base_path.c_str());
    model.allocate_activations(4 * model.cfg.T);

    // Load corpus
    printf("\n--- Loading corpus ---\n");
    std::string full_text = load_corpus(corpus_path);
    if (full_text.empty()) {
        fprintf(stderr, "No corpus found\n");
        return 1;
    }

    // Build vocab
    std::map<unsigned char, int> c2i;
    std::map<int, unsigned char> i2c;
    build_vocab(full_text, c2i, i2c);
    model.char_to_id = c2i;
    model.id_to_char = i2c;
    printf("Vocab: %d\n", (int)c2i.size());

    // Encode
    std::vector<int> data(full_text.size());
    for (size_t i = 0; i < full_text.size(); i++)
        data[i] = c2i[(unsigned char)full_text[i]];
    int n = data.size();
    printf("Tokens: %d\n", n);

    // Create LoRA adapter
    printf("\n--- Creating LoRA adapter ---\n");
    LoRAModel lora;
    lora.init(&model, lora_rank, 32.0f);

    // Fine-tune
    lora_finetune(model, lora, data, n, c2i, i2c, max_steps);

    // Final test
    printf("\n--- Final LoRA Test ---\n");
    std::vector<std::string> seeds = {"Hola", "Buenos dias", "Que es", "Como"};
    for (auto &seed : seeds) {
        std::string gen = lora.generate(seed, 120, 0.7f, 40);
        printf("'%s' -> '%s'\n", seed.c_str(), gen.c_str());
    }

    return 0;
}
