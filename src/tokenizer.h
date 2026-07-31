// ============================================================
// RUBIDIUM TRANSFORMER - TOKENIZER BPE + UNIGRAM HÍBRIDO
// Implementación C++ para entrenamiento CUDA
// ============================================================
#pragma once
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
#include <climits>

// ============================================================
// TOKENIZER BPE + UNIGRAM
// ============================================================
class BPETokenizer {
public:
    int vocab_size;
    
    // Merges: [(pair, new_token)]
    std::vector<std::pair<std::pair<int,int>, std::string>> merges;
    
    // Vocab: token -> id
    std::unordered_map<std::string, int> vocab;
    
    // Inverse vocab: id -> token
    std::map<int, std::string> inv_vocab;
    
    // Special tokens
    int PAD_ID = 0;
    int UNK_ID = 1;
    int BOS_ID = 2;
    int EOS_ID = 3;
    int SEP_ID = 4;
    
    // Unigram probabilities
    std::unordered_map<std::string, double> unigram_probs;
    
    // ============================================================
    // CONSTRUCTOR
    // ============================================================
    BPETokenizer() : vocab_size(0) {
        vocab["<pad>"] = PAD_ID;
        vocab["<unk>"] = UNK_ID;
        vocab["<bos>"] = BOS_ID;
        vocab["<eos>"] = EOS_ID;
        vocab["<sep>"] = SEP_ID;
        
        inv_vocab[PAD_ID] = "<pad>";
        inv_vocab[UNK_ID] = "<unk>";
        inv_vocab[BOS_ID] = "<bos>";
        inv_vocab[EOS_ID] = "<eos>";
        inv_vocab[SEP_ID] = "<sep>";
    }
    
    // ============================================================
    // CARGAR TOKENIZER DESDE JSON
    // ============================================================
    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            fprintf(stderr, "No se pudo abrir tokenizer: %s\n", path.c_str());
            return false;
        }
        
        std::string content((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
        f.close();
        
        // Parsear JSON manualmente (simplificado)
        // Buscar vocab_size
        size_t pos = content.find("\"vocab_size\":");
        if (pos != std::string::npos) {
            pos += 13;
            vocab_size = std::stoi(content.substr(pos));
        }
        
        // Buscar vocab
        pos = content.find("\"vocab\":{");
        if (pos != std::string::npos) {
            pos += 9;
            parse_vocab(content, pos);
        }
        
        // Buscar merges
        pos = content.find("\"merges\":[");
        if (pos != std::string::npos) {
            pos += 10;
            parse_merges(content, pos);
        }
        
        printf("Tokenizer cargado: %d tokens\n", (int)vocab.size());
        return true;
    }
    
    // ============================================================
    // TOKENIZAR TEXTO
    // ============================================================
    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        tokens.push_back(BOS_ID);
        
        // Tokenizar palabra por palabra
        std::istringstream iss(text);
        std::string word;
        
        while (iss >> word) {
            std::vector<int> word_tokens = tokenize_word(word);
            tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
        }
        
        tokens.push_back(EOS_ID);
        return tokens;
    }
    
    // ============================================================
    // DECODIFCAR TOKENS A TEXTO
    // ============================================================
    std::string decode(const std::vector<int>& token_ids) {
        std::string text;
        for (int id : token_ids) {
            if (id >= 0 && inv_vocab.find(id) != inv_vocab.end()) {
                const std::string& token = inv_vocab.at(id);
                // Saltar tokens especiales
                if (token != "<pad>" && token != "<unk>" && 
                    token != "<bos>" && token != "<eos>" && token != "<sep>") {
                    text += token;
                }
            }
        }
        return text;
    }
    
    // ============================================================
    // TOKENIZAR UNA PALABRA
    // ============================================================
    std::vector<int> tokenize_word(const std::string& word) {
        // Empezar con caracteres
        std::vector<int> tokens;
        for (char c : word) {
            std::string s(1, c);
            if (vocab.find(s) != vocab.end()) {
                tokens.push_back(vocab[s]);
            } else {
                tokens.push_back(UNK_ID);
            }
        }
        
        // Aplicar mergeos en orden
        for (auto& merge : merges) {
            int new_id = vocab[merge.second];
            std::vector<int> new_tokens;
            
            for (size_t i = 0; i < tokens.size(); i++) {
                if (i < tokens.size() - 1 && 
                    tokens[i] == merge.first.first && 
                    tokens[i+1] == merge.first.second) {
                    new_tokens.push_back(new_id);
                    i++; // Saltar el siguiente
                } else {
                    new_tokens.push_back(tokens[i]);
                }
            }
            
            tokens = new_tokens;
        }
        
        return tokens;
    }
    
    // ============================================================
    // OBTENER TAMAÑO VOCABULARIO
    // ============================================================
    int get_vocab_size() const {
        return (int)vocab.size();
    }
    
    // ============================================================
    // VERIFICAR SI EL TOKENIZER ESTÁ CARGADO
    // ============================================================
    bool is_loaded() const {
        return vocab.size() > 5; // Más allá de tokens especiales
    }
    
private:
    // ============================================================
    // PARSEAR VOCABULARIO (JSON simplificado)
    // ============================================================
    void parse_vocab(const std::string& content, size_t start) {
        size_t pos = start;
        while (pos < content.size() && content[pos] != '}') {
            // Leer key
            pos = content.find('"', pos);
            if (pos == std::string::npos) break;
            pos++;
            size_t key_start = pos;
            pos = content.find('"', pos);
            if (pos == std::string::npos) break;
            std::string key = content.substr(key_start, pos - key_start);
            pos++;
            
            // Leer value
            pos = content.find(':', pos);
            if (pos == std::string::npos) break;
            pos++;
            while (pos < content.size() && content[pos] == ' ') pos++;
            
            // Leer número
            size_t val_start = pos;
            while (pos < content.size() && (isdigit(content[pos]) || content[pos] == '-')) pos++;
            int value = std::stoi(content.substr(val_start, pos - val_start));
            
            vocab[key] = value;
            inv_vocab[value] = key;
            
            // Buscar siguiente
            pos = content.find(',', pos);
            if (pos == std::string::npos) break;
            pos++;
        }
    }
    
    // ============================================================
    // PARSEAR MERGES (JSON simplificado)
    // ============================================================
    void parse_merges(const std::string& content, size_t start) {
        size_t pos = start;
        while (pos < content.size() && content[pos] != ']') {
            // Leer par [id1, id2]
            pos = content.find('[', pos);
            if (pos == std::string::npos) break;
            pos++;
            
            // Primer ID
            while (pos < content.size() && !isdigit(content[pos]) && content[pos] != '-') pos++;
            size_t val_start = pos;
            while (pos < content.size() && (isdigit(content[pos]) || content[pos] == '-')) pos++;
            int id1 = std::stoi(content.substr(val_start, pos - val_start));
            
            // Segundo ID
            pos = content.find(',', pos);
            if (pos == std::string::npos) break;
            pos++;
            while (pos < content.size() && !isdigit(content[pos]) && content[pos] != '-') pos++;
            val_start = pos;
            while (pos < content.size() && (isdigit(content[pos]) || content[pos] == '-')) pos++;
            int id2 = std::stoi(content.substr(val_start, pos - val_start));
            
            // Leer token string
            pos = content.find('"', pos);
            if (pos == std::string::npos) break;
            pos++;
            size_t tok_start = pos;
            pos = content.find('"', pos);
            if (pos == std::string::npos) break;
            std::string token = content.substr(tok_start, pos - tok_start);
            pos++;
            
            merges.push_back({{id1, id2}, token});
            
            // Buscar siguiente
            pos = content.find(']', pos);
            if (pos == std::string::npos) break;
            pos++;
            pos = content.find(',', pos);
            if (pos == std::string::npos) break;
            pos++;
        }
    }
};

// ============================================================
// GLOBAL TOKENIZER
// ============================================================
extern BPETokenizer g_tokenizer;

// ============================================================
// TOKENIZER FUNCTIONS
// ============================================================
void init_tokenizer(const std::string& path);
std::vector<int> tokenize_text(const std::string& text);
std::string detokenize_tokens(const std::vector<int>& tokens);
