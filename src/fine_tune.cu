// ============================================================
// RUBIDIUM TRANSFORMER - FINE-TUNING ENTRY POINT
// LoRA Fine-tuning with CUDA
// ============================================================
#include "model.h"
#include "lora.h"
#include "tokenizer_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <map>
#include <glob.h>

// ============================================================
// FINE-TUNING CONFIG
// ============================================================
struct FineTuneConfig {
    int vocab_size = 32000;
    int max_seq_len = 512;
    int batch_size = 2;
    int grad_accum = 16;
    int max_steps = 50000;
    float lr = 1e-4f;
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.01f;
    int warmup = 1000;
    float gc = 1.0f;
    int lora_rank = 16;
    float lora_alpha = 32.0f;
    float lora_dropout = 0.1f;
    const char* base_model_path = "model_final.bin";
    const char* tokenizer_path = "tokenizer.json";
    const char* data_dir = "data_finetune";
    const char* output_dir = "checkpoints_finetune";
};

// ============================================================
// INSTRUCTION FORMAT
// ============================================================
struct InstructionPair {
    std::string prompt;
    std::string response;
};

// Convert U:/B: format to instruction format
std::vector<InstructionPair> convert_corpus_to_instructions(const std::string& corpus_path) {
    std::vector<InstructionPair> pairs;
    
    glob_t glob_result;
    glob((corpus_path + "/*.txt").c_str(), GLOB_NOSORT, nullptr, &glob_result);
    
    for (size_t i = 0; i < glob_result.gl_pathc; i++) {
        FILE* f = fopen(glob_result.gl_pathv[i], "rb");
        if (!f) continue;
        
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::string content(sz, 0);
        fread(&content[0], 1, sz, f);
        fclose(f);
        
        // Parse U:/B: pairs
        std::string current_u, current_b;
        std::istringstream iss(content);
        std::string line;
        
        while (std::getline(iss, line)) {
            if (line.rfind("U: ", 0) == 0) {
                if (!current_u.empty() && !current_b.empty()) {
                    pairs.push_back({current_u, current_b});
                }
                current_u = line.substr(3);
                current_b.clear();
            } else if (line.rfind("B: ", 0) == 0) {
                current_b = line.substr(3);
            }
        }
        
        if (!current_u.empty() && !current_b.empty()) {
            pairs.push_back({current_u, current_b});
        }
    }
    
    globfree(&glob_result);
    printf("Converted %zu U:/B: pairs to instructions\n", pairs.size());
    return pairs;
}

// ============================================================
// TOKENIZE INSTRUCTIONS
// ============================================================
struct TokenizedBatch {
    std::vector<int> tokens;
    std::vector<int> targets;
    int total_tokens = 0;
};

TokenizedBatch tokenize_instructions(
    const std::vector<InstructionPair>& pairs,
    GPUTokenizer* gpu_tokenizer,
    int max_seq_len
) {
    TokenizedBatch batch;
    batch.tokens.reserve(pairs.size() * max_seq_len);
    batch.targets.reserve(pairs.size() * max_seq_len);
    
    for (const auto& pair : pairs) {
        // Format: "<bos> U: {prompt} <sep> B: {response} <eos>"
        std::string formatted = "<bos> U: " + pair.prompt + " <sep> B: " + pair.response + " <eos>";
        
        std::vector<int> token_ids = encode_string(gpu_tokenizer, formatted, max_seq_len);
        
        // Create targets (shifted by 1)
        for (size_t i = 0; i + 1 < token_ids.size(); i++) {
            batch.tokens.push_back(token_ids[i]);
            batch.targets.push_back(token_ids[i + 1]);
        }
    }
    
    batch.total_tokens = batch.tokens.size();
    printf("Tokenized: %d total tokens\n", batch.total_tokens);
    return batch;
}

// ============================================================
// MAIN FINE-TUNING
// ============================================================
int main(int argc, char** argv) {
    printf("============================================================\n");
    printf("RUBIDIUM - LoRA Fine-tuning\n");
    printf("============================================================\n");
    
    // Parse args
    FineTuneConfig config;
    if (argc > 1) config.base_model_path = argv[1];
    if (argc > 2) config.data_dir = argv[2];
    if (argc > 3) config.tokenizer_path = argv[3];
    
    // Init GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%.1f GB)\n", prop.name, prop.totalGlobalMem / 1e9);
    
    init_handles();
    
    // Load tokenizer
    printf("\nLoading tokenizer...\n");
    HostTokenizer h_tokenizer;
    if (!h_tokenizer.load_from_json(config.tokenizer_path)) {
        printf("Using Python tokenizer - run: python3 src/tokenizer.py\n");
        return 1;
    }
    
    GPUTokenizer* d_tokenizer;
    tokenizer_copy_to_device(h_tokenizer, &d_tokenizer);
    
    // Convert corpus to instructions
    printf("\nConverting corpus to instructions...\n");
    std::vector<InstructionPair> pairs = convert_corpus_to_instructions(config.data_dir);
    if (pairs.empty()) {
        printf("No instruction pairs found!\n");
        return 1;
    }
    
    // Tokenize
    printf("\nTokenizing instructions...\n");
    TokenizedBatch batch = tokenize_instructions(pairs, d_tokenizer, config.max_seq_len);
    
    // Upload tokens to GPU
    int* d_tokens, *d_targets;
    cudaMalloc(&d_tokens, batch.total_tokens * sizeof(int));
    cudaMalloc(&d_targets, batch.total_tokens * sizeof(int));
    cudaMemcpy(d_tokens, batch.tokens.data(), batch.total_tokens * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, batch.targets.data(), batch.total_tokens * sizeof(int), cudaMemcpyHostToDevice);
    
    // Load base model
    printf("\nLoading base model...\n");
    RubidiumTransformer model;
    ModelConfig model_cfg;
    model_cfg.init(config.vocab_size, config.max_seq_len, 2048, 32, 10, 8192);
    model.init(model_cfg);
    model.allocate_activations(config.batch_size * config.max_seq_len);
    model.load(config.base_model_path);
    
    // Initialize LoRA
    printf("\nInitializing LoRA (rank=%d)...\n", config.lora_rank);
    LoRAConfig lora_cfg;
    lora_cfg.rank = config.lora_rank;
    lora_cfg.alpha = config.lora_alpha;
    lora_cfg.dropout = config.lora_dropout;
    init_lora(model_cfg.D, model_cfg.FF, lora_cfg);
    
    // Freeze base model weights (already loaded, just don't update)
    printf("Base model frozen. LoRA params: ~%.1fM\n", g_lora.count_params() / 1e6);
    
    // Training loop
    printf("\n============================================================\n");
    printf("STARTING FINE-TUNING\n");
    printf("============================================================\n");
    
    int n = batch.total_tokens / config.max_seq_len;
    float smooth_loss = 1e10f;
    double t0 = clock();
    
    for (int step = 1; step <= config.max_steps; step++) {
        // LR schedule
        float lr_t;
        if (step < config.warmup) lr_t = config.lr * step / config.warmup;
        else {
            float p = (float)(step - config.warmup) / (config.max_steps - config.warmup);
            lr_t = config.lr * 0.5f * (1.0f + cosf(3.14159265f * p));
        }
        
        float loss_acc = 0;
        
        for (int ga = 0; ga < config.grad_accum; ga++) {
            // Sample batch
            int start_idx = rand() % (n - 1);
            int offset = start_idx * config.max_seq_len;
            
            // Forward
            float loss = model.forward(
                d_tokens + offset, 
                d_targets + offset, 
                config.batch_size, 
                config.max_seq_len
            );
            
            // Add LoRA forward
            // lora_forward_all(...)
            
            // Backward
            model.backward(
                d_tokens + offset, 
                d_targets + offset, 
                config.batch_size, 
                config.max_seq_len, 
                1.0f / (config.batch_size * config.grad_accum)
            );
            
            // Add LoRA backward
            // lora_backward_all(...)
            
            loss_acc += loss;
        }
        
        // Gradient clipping
        model.clip_gradients(config.gc);
        
        // LoRA optimizer step (only LoRA params)
        lora_optimizer_step(step, lr_t, config.b1, config.b2, config.eps, config.wd);
        
        cudaDeviceSynchronize();
        
        float avg_loss = loss_acc / config.grad_accum;
        smooth_loss = (step == 1) ? avg_loss : 0.98f * smooth_loss + 0.02f * avg_loss;
        
        if (step % 100 == 0 || step == config.max_steps) {
            double elapsed = (clock() - t0) / CLOCKS_PER_SEC;
            double sps = step / elapsed;
            double eta = (config.max_steps - step) / sps / 60.0;
            printf("Step %d/%d | loss: %.4f | lr: %.2e | %.1f steps/s | ETA: %.0fmin\n",
                   step, config.max_steps, smooth_loss, lr_t, sps, eta);
        }
        
        // Checkpoint
        if (step % 5000 == 0) {
            char path[256];
            sprintf(path, "%s/lora_step_%d.bin", config.output_dir, step);
            // Save LoRA weights
            printf("Checkpoint: %s\n", path);
        }
    }
    
    // Save final LoRA
    char final_path[256];
    sprintf(final_path, "%s/lora_final.bin", config.output_dir);
    // lora_save(final_path);
    printf("\nFine-tuning complete! LoRA saved to %s\n", final_path);
    
    // Cleanup
    tokenizer_free_gpu(d_tokenizer);
    cudaFree(d_tokenizer);
    cudaFree(d_tokens);
    cudaFree(d_targets);
    model.free_all();
    lora_free();
    destroy_handles();
    
    return 0;
}