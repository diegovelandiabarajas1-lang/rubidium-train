// ============================================================
// RUBIDIUM TRANSFORMER - TOKENIZER CUDA IMPLEMENTATION
// BPE + Unigram GPU-accelerated encoding/decoding
// ============================================================
#include "tokenizer_cuda.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <queue>
#include <algorithm>

// ============================================================
// FNV-1a HASH (compile-time for GPU)
// ============================================================
__device__ __host__ inline uint64_t fnv1a_hash(const char* data, size_t len) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) {
        hash ^= (unsigned char)data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

__device__ __host__ inline uint64_t fnv1a_hash_str(const char* str) {
    uint64_t hash = 14695981039346656037ULL;
    while (*str) {
        hash ^= (unsigned char)*str++;
        hash *= 1099511628211ULL;
    }
    return hash;
}

// ============================================================
// VOCAB LOOKUP KERNEL
// ============================================================
__device__ int vocab_lookup(const GPUTokenizer* tok, const char* str, int len) {
    if (len == 0) return tok->unk_id;
    if (len == 1) {
        unsigned char c = str[0];
        if (c < 256) return c + 5; // ASCII chars after special tokens
    }
    
    uint64_t hash = fnv1a_hash(str, len);
    int table_size = tok->vocab_table_size;
    int idx = hash % table_size;
    
    // Linear probing
    for (int probe = 0; probe < table_size; probe++) {
        int probe_idx = (idx + probe) % table_size;
        const VocabEntry* entry = &tok->d_vocab_table[probe_idx];
        
        if (entry->token_id == -1) {
            return tok->unk_id; // Empty slot
        }
        
        if (entry->key_hash == hash && entry->length == len) {
            // Verify string match
            bool match = true;
            for (int i = 0; i < len; i++) {
                if (entry->string_data[i] != str[i]) {
                    match = false;
                    break;
                }
            }
            if (match) return entry->token_id;
        }
    }
    return tok->unk_id;
}

// ============================================================
// MERGE TRIE TRAVERSAL
// ============================================================
__device__ int find_merge(const GPUTokenizer* tok, int first_id, int second_id) {
    // Sequential scan through merge rules (optimized for small merge sets)
    for (int i = 0; i < tok->num_merges; i++) {
        const MergeRule* rule = &tok->d_merge_rules[i];
        if (rule->first_id == first_id && rule->second_id == second_id) {
            return rule->result_id;
        }
    }
    return -1;
}

// ============================================================
// ENCODE KERNEL - One thread per character
// ============================================================
__global__ void encode_kernel(
    const GPUTokenizer* tok,
    const char* strings_data,     // Concatenated input strings
    const int* string_offsets,    // Offset of each string
    const int* string_lengths,    // Length of each string
    int batch_size,
    int* output_tokens,           // Flattened output
    int* output_lengths,          // Length per sequence
    int max_seq_len
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int str_offset = string_offsets[batch_idx];
    int str_len = string_lengths[batch_idx];
    const char* str = strings_data + str_offset;
    
    // Each thread processes one character initially
    int tid = threadIdx.x;
    if (tid >= str_len) return;
    
    // Shared memory for intermediate tokens
    extern __shared__ int shared_tokens[];
    int* s_tokens = shared_tokens;
    int* s_lengths = shared_tokens + blockDim.x;
    
    // Initial tokenization: each char -> token
    char c = str[tid];
    int token_id = vocab_lookup(tok, &c, 1);
    s_tokens[tid] = token_id;
    __syncthreads();
    
    // Sequential merge application (single thread for now)
    if (threadIdx.x == 0) {
        int current_len = str_len;
        int* tokens = s_tokens;
        int* out = output_tokens + batch_idx * max_seq_len;
        
        // Apply merges iteratively
        bool merged = true;
        while (merged && current_len < max_seq_len - 1) {
            merged = false;
            int write_idx = 0;
            int read_idx = 0;
            
            while (read_idx < current_len - 1) {
                int first = tokens[read_idx];
                int second = tokens[read_idx + 1];
                
                int merge_result = find_merge(tok, first, second);
                if (merge_result != -1) {
                    out[write_idx++] = merge_result;
                    read_idx += 2;
                    merged = true;
                } else {
                    out[write_idx++] = first;
                    read_idx++;
                }
            }
            if (read_idx < current_len) {
                out[write_idx++] = tokens[read_idx];
            }
            
            current_len = write_idx;
            // Copy back for next iteration
            for (int i = 0; i < current_len; i++) {
                tokens[i] = out[i];
            }
        }
        
        // Add special tokens
        out[current_len++] = tok->eos_id;
        output_lengths[batch_idx] = current_len;
    }
}

// ============================================================
// DECODE KERNEL
// ============================================================
__global__ void decode_kernel(
    const GPUTokenizer* tok,
    const int* input_tokens,
    const int* input_lengths,
    int batch_size,
    char* output_strings,
    int max_string_len
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int offset = input_lengths[batch_idx];
    const int* tokens = input_tokens + batch_idx * offset;
    char* out = output_strings + batch_idx * max_string_len;
    
    int write_pos = 0;
    for (int i = 0; i < offset; i++) {
        int token_id = tokens[i];
        if (token_id == tok->eos_id || token_id == tok->pad_id) break;
        
        int string_offset = tok->d_inv_vocab_offsets[token_id];
        int string_len = tok->d_inv_vocab_lengths[token_id];
        const char* token_str = tok->d_inv_vocab_strings + string_offset;
        
        if (write_pos + string_len < max_string_len) {
            for (int j = 0; j < string_len; j++) {
                out[write_pos++] = token_str[j];
            }
        }
    }
    out[write_pos] = '\0';
}

// ============================================================
// HOST INITIALIZATION
// ============================================================
cudaError_t tokenizer_init_gpu(GPUTokenizer* d_tok, const HostTokenizer& h_tok) {
    // Allocate device memory for tokenizer
    GPUTokenizer h_gpu_tok;
    memset(&h_gpu_tok, 0, sizeof(GPUTokenizer));
    
    h_gpu_tok.pad_id = h_tok.PAD_ID;
    h_gpu_tok.unk_id = h_tok.UNK_ID;
    h_gpu_tok.bos_id = h_tok.BOS_ID;
    h_gpu_tok.eos_id = h_tok.EOS_ID;
    h_gpu_tok.sep_id = h_tok.SEP_ID;
    h_gpu_tok.vocab_size = h_tok.vocab_size;
    h_gpu_tok.num_merges = (int)h_tok.merges.size();
    h_gpu_tok.max_token_len = 32;
    
    // Build vocab hash table
    std::vector<VocabEntry> vocab_table;
    int table_size;
    h_tok.build_vocab_hash_table(vocab_table, table_size);
    h_gpu_tok.vocab_table_size = table_size;
    
    // Build merge trie
    std::vector<MergeTrieNode> merge_trie;
    std::vector<MergeRule> merge_rules;
    h_tok.build_merge_trie(merge_trie, merge_rules);
    
    // Build inverse vocab arrays
    std::vector<char> inv_vocab_strings;
    std::vector<int> inv_vocab_offsets, inv_vocab_lengths;
    h_tok.build_inv_vocab_arrays(inv_vocab_strings, inv_vocab_offsets, inv_vocab_lengths);
    
    // Build unigram array
    std::vector<UnigramEntry> unigram;
    h_tok.build_unigram_array(unigram);
    
    // Allocate and copy to device
    cudaMalloc(&h_gpu_tok.d_vocab_table, vocab_table.size() * sizeof(VocabEntry));
    cudaMemcpy(h_gpu_tok.d_vocab_table, vocab_table.data(), 
               vocab_table.size() * sizeof(VocabEntry), cudaMemcpyHostToDevice);
    
    cudaMalloc(&h_gpu_tok.d_merge_trie, merge_trie.size() * sizeof(MergeTrieNode));
    cudaMemcpy(h_gpu_tok.d_merge_trie, merge_trie.data(),
               merge_trie.size() * sizeof(MergeTrieNode), cudaMemcpyHostToDevice);
    
    cudaMalloc(&h_gpu_tok.d_merge_rules, merge_rules.size() * sizeof(MergeRule));
    cudaMemcpy(h_gpu_tok.d_merge_rules, merge_rules.data(),
               merge_rules.size() * sizeof(MergeRule), cudaMemcpyHostToDevice);
    
    cudaMalloc(&h_gpu_tok.d_unigram_probs, unigram.size() * sizeof(UnigramEntry));
    cudaMemcpy(h_gpu_tok.d_unigram_probs, unigram.data(),
               unigram.size() * sizeof(UnigramEntry), cudaMemcpyHostToDevice);
    
    cudaMalloc(&h_gpu_tok.d_inv_vocab_strings, inv_vocab_strings.size());
    cudaMemcpy(h_gpu_tok.d_inv_vocab_strings, inv_vocab_strings.data(),
               inv_vocab_strings.size(), cudaMemcpyHostToDevice);
    
    cudaMalloc(&h_gpu_tok.d_inv_vocab_offsets, inv_vocab_offsets.size() * sizeof(int));
    cudaMemcpy(h_gpu_tok.d_inv_vocab_offsets, inv_vocab_offsets.data(),
               inv_vocab_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(&h_gpu_tok.d_inv_vocab_lengths, inv_vocab_lengths.size() * sizeof(int));
    cudaMemcpy(h_gpu_tok.d_inv_vocab_lengths, inv_vocab_lengths.data(),
               inv_vocab_lengths.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    // Copy struct to device
    cudaMemcpy(d_tok, &h_gpu_tok, sizeof(GPUTokenizer), cudaMemcpyHostToDevice);
    
    return cudaSuccess;
}

void tokenizer_free_gpu(GPUTokenizer* d_tok) {
    GPUTokenizer h_tok;
    cudaMemcpy(&h_tok, d_tok, sizeof(GPUTokenizer), cudaMemcpyDeviceToHost);
    
    if (h_tok.d_vocab_table) cudaFree(h_tok.d_vocab_table);
    if (h_tok.d_merge_trie) cudaFree(h_tok.d_merge_trie);
    if (h_tok.d_merge_rules) cudaFree(h_tok.d_merge_rules);
    if (h_tok.d_unigram_probs) cudaFree(h_tok.d_unigram_probs);
    if (h_tok.d_inv_vocab_strings) cudaFree(h_tok.d_inv_vocab_strings);
    if (h_tok.d_inv_vocab_offsets) cudaFree(h_tok.d_inv_vocab_offsets);
    if (h_tok.d_inv_vocab_lengths) cudaFree(h_tok.d_inv_vocab_lengths);
}

cudaError_t encode_batch(
    const GPUTokenizer* tok,
    const char** h_strings,
    int batch_size,
    int* d_output_tokens,
    int* d_output_lengths,
    int max_seq_len,
    cudaStream_t stream
) {
    // Concatenate strings on host
    std::vector<int> offsets(batch_size);
    std::vector<int> lengths(batch_size);
    std::vector<char> all_chars;
    
    for (int i = 0; i < batch_size; i++) {
        offsets[i] = all_chars.size();
        int len = strlen(h_strings[i]);
        lengths[i] = len;
        all_chars.insert(all_chars.end(), h_strings[i], h_strings[i] + len);
    }
    
    // Copy to device
    char* d_strings_data;
    int* d_offsets, *d_lengths;
    cudaMalloc(&d_strings_data, all_chars.size());
    cudaMalloc(&d_offsets, batch_size * sizeof(int));
    cudaMalloc(&d_lengths, batch_size * sizeof(int));
    
    cudaMemcpy(d_strings_data, all_chars.data(), all_chars.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, lengths.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threads = 256;
    size_t shared_mem = threads * 2 * sizeof(int);
    encode_kernel<<<batch_size, threads, shared_mem, stream>>>(
        tok, d_strings_data, d_offsets, d_lengths,
        batch_size, d_output_tokens, d_output_lengths, max_seq_len
    );
    
    cudaFree(d_strings_data);
    cudaFree(d_offsets);
    cudaFree(d_lengths);
    
    return cudaGetLastError();
}

cudaError_t decode_batch(
    const GPUTokenizer* tok,
    const int* d_input_tokens,
    const int* d_input_lengths,
    int batch_size,
    char** h_output_strings,
    int max_string_len,
    cudaStream_t stream
) {
    decode_kernel<<<batch_size, 1, 0, stream>>>(
        tok, d_input_tokens, d_input_lengths,
        batch_size, h_output_strings[0], max_string_len
    );
    return cudaGetLastError();
}

std::vector<int> encode_string(const GPUTokenizer* d_tok, const std::string& text, int max_seq_len) {
    const char* cstr = text.c_str();
    int* d_output_tokens, *d_output_lengths;
    cudaMalloc(&d_output_tokens, max_seq_len * sizeof(int));
    cudaMalloc(&d_output_lengths, sizeof(int));
    
    encode_batch(d_tok, &cstr, 1, d_output_tokens, d_output_lengths, max_seq_len, 0);
    
    int length;
    cudaMemcpy(&length, d_output_lengths, sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<int> tokens(length);
    cudaMemcpy(tokens.data(), d_output_tokens, length * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_output_tokens);
    cudaFree(d_output_lengths);
    return tokens;
}

std::string decode_tokens(const GPUTokenizer* d_tok, const std::vector<int>& tokens) {
    int* d_input_tokens, *d_input_lengths;
    char* d_output_string;
    cudaMalloc(&d_input_tokens, tokens.size() * sizeof(int));
    cudaMalloc(&d_input_lengths, sizeof(int));
    cudaMalloc(&d_output_string, 1024);
    
    cudaMemcpy(d_input_tokens, tokens.data(), tokens.size() * sizeof(int), cudaMemcpyHostToDevice);
    int len = tokens.size();
    cudaMemcpy(d_input_lengths, &len, sizeof(int), cudaMemcpyHostToDevice);
    
    decode_batch(d_tok, d_input_tokens, d_input_lengths, 1, &d_output_string, 1024, 0);
    
    char result[1024];
    cudaMemcpy(result, d_output_string, 1024, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input_tokens);
    cudaFree(d_input_lengths);
    cudaFree(d_output_string);
    
    return std::string(result);
}

cudaError_t tokenizer_copy_to_device(const HostTokenizer& h_tok, GPUTokenizer** d_tok) {
    cudaMalloc(d_tok, sizeof(GPUTokenizer));
    return tokenizer_init_gpu(*d_tok, h_tok);
}

void tokenizer_print_stats(const HostTokenizer& tok) {
    printf("Vocab size: %d\n", tok.vocab_size);
    printf("Merges: %zu\n", tok.merges.size());
    printf("Unigram entries: %zu\n", tok.unigram_probs.size());
}

bool HostTokenizer::load_from_json(const std::string& path) {
    // Implementation would parse JSON
    // For now, return false - use Python tokenizer
    return false;
}

void HostTokenizer::build_merge_trie(std::vector<MergeTrieNode>& trie, std::vector<MergeRule>& rules) {
    // Build trie from merges
    trie.clear();
    rules.clear();
    
    // Add root
    MergeTrieNode root;
    for (int i = 0; i < 256; i++) root.child[i] = -1;
    root.merge_id = -1;
    root.token_id = -1;
    trie.push_back(root);
    
    // Add merge rules
    for (size_t i = 0; i < merges.size(); i++) {
        const auto& m = merges[i];
        int first = m.first.first;
        int second = m.first.second;
        int result = vocab[m.second];
        
        // Navigate/create trie path
        int node_idx = 0;
        int chars[2] = {first % 256, second % 256}; // Simplified
        
        for (int c : chars) {
            if (trie[node_idx].child[c] == -1) {
                MergeTrieNode new_node;
                for (int j = 0; j < 256; j++) new_node.child[j] = -1;
                new_node.merge_id = -1;
                new_node.token_id = -1;
                trie[node_idx].child[c] = trie.size();
                trie.push_back(new_node);
            }
            node_idx = trie[node_idx].child[c];
        }
        
        trie[node_idx].merge_id = (int)i;
        trie[node_idx].token_id = result;
        
        // Add to rules array
        MergeRule rule;
        rule.first_id = first;
        rule.second_id = second;
        rule.result_id = result;
        rule.priority = (int)i;
        rules.push_back(rule);
    }
}

void HostTokenizer::build_vocab_hash_table(std::vector<VocabEntry>& table, int& table_size) {
    // Simple hash table with linear probing
    table_size = vocab_size * 2;
    while ((table_size & (table_size - 1)) != 0) table_size++; // Next power of 2
    
    table.resize(table_size);
    for (auto& e : table) {
        e.key_hash = 0;
        e.token_id = -1;
        e.length = 0;
        memset(e.string_data, 0, 32);
    }
    
    for (const auto& kv : vocab) {
        if (kv.second < 5) continue; // Skip special tokens
        const std::string& s = kv.first;
        uint64_t hash = fnv1a_hash_str(s.c_str());
        int idx = hash % table_size;
        
        for (int probe = 0; probe < table_size; probe++) {
            int probe_idx = (idx + probe) % table_size;
            if (table[probe_idx].token_id == -1) {
                table[probe_idx].key_hash = hash;
                table[probe_idx].token_id = kv.second;
                table[probe_idx].length = (uint16_t)s.length();
                memcpy(table[probe_idx].string_data, s.c_str(), std::min<size_t>(s.length(), 31));
                break;
            }
        }
    }
}

void HostTokenizer::build_inv_vocab_arrays(std::vector<char>& strings, std::vector<int>& offsets, std::vector<int>& lengths) {
    offsets.resize(vocab_size);
    lengths.resize(vocab_size);
    strings.clear();
    
    for (int i = 0; i < vocab_size; i++) {
        auto it = inv_vocab.find(i);
        if (it != inv_vocab.end()) {
            const std::string& s = it->second;
            offsets[i] = strings.size();
            lengths[i] = s.length();
            strings.insert(strings.end(), s.begin(), s.end());
        } else {
            offsets[i] = 0;
            lengths[i] = 0;
        }
    }
}

void HostTokenizer::build_unigram_array(std::vector<UnigramEntry>& unigram) {
    for (const auto& kv : unigram_probs) {
        auto it = vocab.find(kv.first);
        if (it != vocab.end()) {
            UnigramEntry e;
            e.token_id = it->second;
            e.log_prob = logf((float)kv.second);
            unigram.push_back(e);
        }
    }
}