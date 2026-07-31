#!/usr/bin/env python3
"""
RUBIDIUM - Tokenizer BPE + Unigram Híbrido
Entrenamiento y uso del tokenizer para el modelo transformer
"""
import os
import json
import re
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import pickle

class BPETokenizer:
    """
    Tokenizer BPE + Unigram híbrido:
    1. Primero aplica BPE para aprender mergeos de caracteres
    2. Luego usa Unigram para seleccionar subpalabras óptimas
    """
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.merges = []  # Lista de mergeos BPE [(pair, new_token)]
        self.vocab = {}   # token -> id
        self.inv_vocab = {}  # id -> token
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<sep>": 4,
        }
        self.unigram_probs = {}  # Probabilidades Unigram
        
    def _get_stats(self, ids: List[List[int]]) -> Dict[Tuple[int, int], int]:
        """Cuenta frecuencias de pares de tokens"""
        counts = defaultdict(int)
        for word_ids in ids:
            for i in range(len(word_ids) - 1):
                counts[(word_ids[i], word_ids[i + 1])] += 1
        return counts
    
    def _merge(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """Aplica un mergeo a la secuencia de tokens"""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def _build_vocab_from_merges(self, corpus_chars: List[str]):
        """Construye vocabulario inicial desde caracteres"""
        # Empezar con todos los caracteres únicos
        char_counts = Counter(corpus_chars)
        sorted_chars = sorted(char_counts.items(), key=lambda x: -x[1])
        
        # Asignar IDs a caracteres especiales primero
        self.vocab = dict(self.special_tokens)
        next_id = len(self.special_tokens)
        
        # Asignar IDs a caracteres frecuentes
        for char, count in sorted_chars:
            if char not in self.vocab:
                self.vocab[char] = next_id
                next_id += 1
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def train_bpe(self, corpus: str, num_merges: int = 30000):
        """
        Entrena BPE en el corpus
        """
        print(f"Entrenando BPE con {num_merges} mergeos...")
        print(f"Corpus: {len(corpus)} caracteres")
        
        # Tokenizar en caracteres
        corpus_chars = list(corpus)
        self._build_vocab_from_merges(corpus_chars)
        
        # Convertir corpus a IDs
        ids = [self.vocab[c] for c in corpus_chars]
        
        # Entrenar mergeos
        for i in range(num_merges):
            if len(self.vocab) >= self.vocab_size:
                print(f"Vocabulario alcanzado: {len(self.vocab)} tokens")
                break
            
            stats = self._get_stats([ids])
            if not stats:
                break
            
            # Encontrar el par más frecuente
            best_pair = max(stats, key=stats.get)
            best_count = stats[best_pair]
            
            if best_count < 2:  # Umbral mínimo
                break
            
            # Crear nuevo token
            new_token = self.inv_vocab[best_pair[0]] + self.inv_vocab[best_pair[1]]
            new_id = len(self.vocab)
            
            # Agregar a vocabulario
            self.vocab[new_token] = new_id
            self.inv_vocab[new_id] = new_token
            self.merges.append((best_pair, new_token))
            
            # Aplicar mergeo al corpus
            ids = self._merge(ids, best_pair, new_id)
            
            if (i + 1) % 1000 == 0:
                print(f"  Merge {i+1}/{num_merges}: '{new_token}' (freq: {best_count})")
        
        print(f"Vocabulario final: {len(self.vocab)} tokens")
        print(f"Mergeos realizados: {len(self.merges)}")
        
        # Calcular probabilidades Unigram
        self._compute_unigram_probs(ids)
    
    def _compute_unigram_probs(self, ids: List[int]):
        """Calcula probabilidades Unigram para cada token"""
        token_counts = Counter(ids)
        total = sum(token_counts.values())
        
        self.unigram_probs = {}
        for token_id, count in token_counts.items():
            token = self.inv_vocab[token_id]
            self.unigram_probs[token] = count / total
    
    def encode(self, text: str) -> List[int]:
        """
        Tokeniza texto usando BPE + Unigram
        """
        # Agregar tokens especiales
        tokens = [self.special_tokens["<bos>"]]
        
        # Tokenizar cada palabra
        words = text.split()
        for word in words:
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        
        tokens.append(self.special_tokens["<eos>"])
        return tokens
    
    def _tokenize_word(self, word: str) -> List[int]:
        """Tokeniza una palabra usando BPE"""
        # Empezar con caracteres
        tokens = []
        for char in word:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.special_tokens["<unk>"])
        
        # Aplicar mergeos en orden
        for pair, new_token in self.merges:
            new_id = self.vocab[new_token]
            tokens = self._merge(tokens, pair, new_id)
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decodifica IDs a texto
        """
        text = ""
        for token_id in token_ids:
            token = self.inv_vocab.get(token_id, "<unk>")
            if token not in self.special_tokens:
                text += token
        return text
    
    def save(self, path: str):
        """Guarda el tokenizer"""
        data = {
            "vocab_size": self.vocab_size,
            "merges": [(list(p), t) for p, t in self.merges],
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
            "unigram_probs": self.unigram_probs,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer guardado: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Carga el tokenizer"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(data["vocab_size"])
        tokenizer.merges = [(tuple(p), t) for p, t in data["merges"]]
        tokenizer.vocab = data["vocab"]
        tokenizer.inv_vocab = {int(v): k for k, v in tokenizer.vocab.items()}
        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.unigram_probs = data["unigram_probs"]
        
        print(f"Tokenizer cargado: {len(tokenizer.vocab)} tokens")
        return tokenizer


class CorpusTokenizer:
    """
    Tokeniza corpus completo para entrenamiento
    """
    
    def __init__(self, tokenizer: BPETokenizer):
        self.tokenizer = tokenizer
    
    def tokenize_corpus(self, input_path: str, output_path: str):
        """
        Tokeniza un archivo de corpus y guarda los tokens
        """
        print(f"Tokenizando corpus: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenizar
        token_ids = self.tokenizer.encode(text)
        
        # Guardar como binario
        import numpy as np
        token_array = np.array(token_ids, dtype=np.int32)
        token_array.tofile(output_path)
        
        print(f"Tokens guardados: {output_path}")
        print(f"Total tokens: {len(token_ids)}")
        print(f"Ratio compresión: {len(text) / len(token_ids):.2f}x")
        
        return len(token_ids)
    
    def tokenize_directory(self, input_dir: str, output_dir: str):
        """
        Tokeniza todos los archivos .txt de un directorio
        """
        os.makedirs(output_dir, exist_ok=True)
        
        total_tokens = 0
        for filename in os.listdir(input_dir):
            if filename.endswith('.txt'):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename.replace('.txt', '.bin'))
                
                tokens = self.tokenize_corpus(input_path, output_path)
                total_tokens += tokens
        
        print(f"\nTotal tokens en corpus: {total_tokens}")
        return total_tokens


def train_tokenizer_on_corpus(corpus_dir: str, vocab_size: int = 32000, 
                              output_path: str = "tokenizer.json"):
    """
    Entrena el tokenizer en todo el corpus
    """
    print("=" * 60)
    print("ENTRENAMIENTO DE TOKENIZER BPE + UNIGRAM")
    print("=" * 60)
    
    # Leer todo el corpus
    corpus_text = ""
    for filename in os.listdir(corpus_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(corpus_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                corpus_text += f.read() + "\n"
            print(f"  + {filename}: {len(corpus_text):,} chars")
    
    print(f"\nCorpus total: {len(corpus_text):,} caracteres")
    
    # Entrenar tokenizer
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train_bpe(corpus_text, num_merges=vocab_size - 256 - len(tokenizer.special_tokens))
    
    # Guardar
    tokenizer.save(output_path)
    
    # Estadísticas
    print("\n" + "=" * 60)
    print("ESTADÍSTICAS DEL TOKENIZER")
    print("=" * 60)
    
    # Test de compresión
    test_texts = [
        "Hola, ¿cómo estás?",
        "Buenos días, ¿en qué puedo ayudarte?",
        "La inteligencia artificial está transformando el mundo.",
        "Python es un lenguaje de programación muy popular.",
    ]
    
    for text in test_texts:
        tokens = tokenizer.encode(text)
        print(f"  '{text[:40]}...' -> {len(tokens)} tokens ({len(text)/len(tokens):.1f}x)")
    
    return tokenizer


if __name__ == "__main__":
    import sys
    
    corpus_dir = sys.argv[1] if len(sys.argv) > 1 else "resources"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "tokenizer.json"
    
    tokenizer = train_tokenizer_on_corpus(corpus_dir, vocab_size=32000, output_path=output_path)
