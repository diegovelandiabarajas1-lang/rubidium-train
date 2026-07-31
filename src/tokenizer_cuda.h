#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <unordered_map>

// ============================================================
// TOKENIZER CONSTANTS
// ============================================================
#define MAX_VOCAB_SIZE 65536
#define MAX_MERGES 60000
#define MAX_BATCH_SIZE 256
#define MAX_SEQ_LEN 2048
#define MAX_TOKEN_LEN 32

// Special token IDs
#define PAD_ID 0
#define UNK_ID 1
#define BOS_ID 2
#define EOS_ID 3
#define SEP_ID 4

// ============================================================
// GPU DATA STRUCTURES
// ============================================================

// Hash table entry for vocab lookup (string -> id)
struct VocabEntry {
    uint64_t key_hash;  // FNV-1a hash of token string
    int token_id;       // Token ID
    uint16_t length;    // Token string length
    char string_data[32];  // Inline storage for short tokens
};

// Merge trie node for BPE merges
struct MergeTrieNode {
    int child[256];     // Child node indices (packed for GPU)
    int merge_id;       // Merge rule ID if this is a merge point, -1 otherwise
    int token_id;       // Resulting token ID for this merge
};

// BPE Merge rule
struct MergeRule {
    int first_id;       // First token ID in pair
    int second_id;      // Second token ID in pair
    int result_id;      // Resulting merged token ID
    int priority;       // Merge priority (lower = earlier)
};

// Unigram probability entry
struct UnigramEntry {
    int token_id;
    float log_prob;     // Log probability (negative)
};

// GPU Tokenizer context
struct GPUTokenizer {
    // Vocabulary hash table
    VocabEntry* d_vocab_table;
    int vocab_table_size;
    int vocab_size;
    
    // Merge trie
    MergeTrieNode* d_merge_trie;
    int merge_trie_size;
    int num_merges;
    
    // Merge rules array (for sequential processing)
    MergeRule* d_merge_rules;
    
    // Unigram probabilities
    UnigramEntry* d_unigram_probs;
    int unigram_size;
    
    // Inverse vocab (id -> string) for decoding
    char* d_inv_vocab_strings;  // Concatenated strings
    int* d_inv_vocab_offsets;   // Offset into strings array
    int* d_inv_vocab_lengths;   // Length of each string
    
    // Special tokens
    int pad_id, unk_id, bos_id, eos_id, sep_id;
    
    // Stats
    int max_token_len;
};

// ============================================================
// HOST-SIDE TOKENIZER LOADER
// ============================================================

struct HostTokenizer {
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> inv_vocab;
    std::vector<std::pair<std::pair<int, int>, std::string>> merges;
    std::unordered_map<std::string, double> unigram_probs;
    int vocab_size = 0;
    
    // Special tokens
    const int PAD_ID = 0;
    const int UNK_ID = 1;
    const int BOS_ID = 2;
    const int EOS_ID = 3;
    const int SEP_ID = 4;
    
    bool load_from_json(const std::string& path);
    void build_merge_trie(std::vector<MergeTrieNode>& trie, std::vector<MergeRule>& rules);
    void build_vocab_hash_table(std::vector<VocabEntry>& table, int& table_size);
    void build_inv_vocab_arrays(std::vector<char>& strings, std::vector<int>& offsets, std::vector<int>& lengths);
    void build_unigram_array(std::vector<UnigramEntry>& unigram);
};

// ============================================================
// CUDA KERNELS
// ============================================================

// Initialize GPU tokenizer context
cudaError_t tokenizer_init_gpu(GPUTokenizer* d_tok, const HostTokenizer& h_tok);

// Free GPU tokenizer context
void tokenizer_free_gpu(GPUTokenizer* d_tok);

// Batch encode: multiple strings -> token IDs
// Input: array of string pointers (on host), batch_size
// Output: d_output_tokens (flattened), d_output_lengths (per sequence)
// Returns: total tokens written
cudaError_t encode_batch(
    const GPUTokenizer* tok,
    const char** h_strings,
    int batch_size,
    int* d_output_tokens,
    int* d_output_lengths,
    int max_seq_len,
    cudaStream_t stream = 0
);

// Batch decode: token IDs -> strings
// Input: d_input_tokens (flattened), d_input_lengths (per sequence), batch_size
// Output: h_output_strings (pre-allocated host buffers)
cudaError_t decode_batch(
    const GPUTokenizer* tok,
    const int* d_input_tokens,
    const int* d_input_lengths,
    int batch_size,
    char** h_output_strings,
    int max_string_len,
    cudaStream_t stream = 0
);

// Encode single string (host helper, uses GPU kernel internally)
std::vector<int> encode_string(const GPUTokenizer* d_tok, const std::string& text, int max_seq_len = MAX_SEQ_LEN);

// Decode token IDs to string (host helper)
std::string decode_tokens(const GPUTokenizer* d_tok, const std::vector<int>& tokens);

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

// Copy tokenizer to device
cudaError_t tokenizer_copy_to_device(const HostTokenizer& h_tok, GPUTokenizer** d_tok);

// Get tokenizer stats
void tokenizer_print_stats(const HostTokenizer& tok);

// Validate tokenizer round-trip
bool tokenizer_validate_roundtrip(const HostTokenizer& h_tok, const GPUTokenizer* d_tok, const std::vector<std::string>& test_strings);
