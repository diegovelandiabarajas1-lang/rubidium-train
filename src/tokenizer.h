// ============================================================
// RUBIDIUM TRANSFORMER - TOKENIZER
// Simple character-level tokenizer
// ============================================================
#pragma once
#include <map>
#include <string>
#include <vector>

struct Tokenizer {
    std::map<unsigned char, int> char_to_id;
    std::map<int, unsigned char> id_to_char;
    int vocab_size;

    void build(const std::string &text) {
        char_to_id.clear();
        id_to_char.clear();
        for (unsigned char c : text) {
            if (char_to_id.find(c) == char_to_id.end()) {
                int id = char_to_id.size();
                char_to_id[c] = id;
                id_to_char[id] = c;
            }
        }
        vocab_size = char_to_id.size();
    }

    std::vector<int> encode(const std::string &text) const {
        std::vector<int> ids;
        for (unsigned char c : text) {
            auto it = char_to_id.find(c);
            ids.push_back(it != char_to_id.end() ? it->second : 0);
        }
        return ids;
    }

    std::string decode(const std::vector<int> &ids) const {
        std::string text;
        for (int id : ids) {
            auto it = id_to_char.find(id);
            text += (it != id_to_char.end()) ? (char)it->second : '?';
        }
        return text;
    }
};
