#!/usr/bin/env python3
"""
Convert RBN1 binary model to Python pickle format
compatible with Rust inference (rubidium-core).
"""
import struct
import pickle
import numpy as np
import sys
import os

def read_binary_model(path):
    """Read RBN1 binary model file."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'RBN1':
            raise ValueError(f"Invalid magic: {magic}")
        
        V, T, D, H, L, FF = struct.unpack('6i', f.read(24))
        print(f"Config: V={V} T={T} D={D} H={H} L={L} FF={FF}")
        
        # char_to_id map (256 ints)
        char_to_id = {}
        id_to_char = {}
        for i in range(256):
            id_val = struct.unpack('i', f.read(4))[0]
            if id_val > 0 or i == 0:
                char_to_id[chr(i)] = id_val
                id_to_char[id_val] = chr(i)
        
        # Read weights
        def read_floats(n):
            return np.frombuffer(f.read(n * 4), dtype=np.float32)
        
        token_emb = read_floats(V * D).reshape(V, D)
        pos_emb = read_floats(T * D).reshape(1, T, D)
        
        layers = []
        for l in range(L):
            ln1_w = read_floats(D)
            ln1_b = read_floats(D)
            wq = read_floats(D * D).reshape(D, D)
            bq = read_floats(D)
            wk = read_floats(D * D).reshape(D, D)
            bk = read_floats(D)
            wv = read_floats(D * D).reshape(D, D)
            bv = read_floats(D)
            wo = read_floats(D * D).reshape(D, D)
            bo = read_floats(D)
            ln2_w = read_floats(D)
            ln2_b = read_floats(D)
            w1 = read_floats(D * FF).reshape(FF, D)
            b1 = read_floats(FF)
            w2 = read_floats(FF * D).reshape(D, FF)
            b2 = read_floats(D)
            
            layers.append({
                'ln1_w': ln1_w, 'ln1_b': ln1_b,
                'attn_wq_w': wq, 'attn_wq_b': bq,
                'attn_wk_w': wk, 'attn_wk_b': bk,
                'attn_wv_w': wv, 'attn_wv_b': bv,
                'attn_wo_w': wo, 'attn_wo_b': bo,
                'ln2_w': ln2_w, 'ln2_b': ln2_b,
                'ff_w1_w': w1, 'ff_w1_b': b1,
                'ff_w2_w': w2, 'ff_w2_b': b2,
            })
        
        ln_f_w = read_floats(D)
        ln_f_b = read_floats(D)
        lm_w = read_floats(V * D).reshape(V, D)
        lm_b = read_floats(V)
    
    return {
        'vocab_size': V,
        'block_size': T,
        'd_model': D,
        'n_head': H,
        'n_layer': L,
        'd_ff': FF,
        'char_to_id': char_to_id,
        'id_to_char': id_to_char,
        'token_emb': token_emb,
        'pos_emb': pos_emb,
        'ln_f_w': ln_f_w,
        'ln_f_b': ln_f_b,
        'lm_w': lm_w,
        'lm_b': lm_b,
        'layers': layers,
    }

def save_pickle(state, path):
    """Save as Python pickle (compatible with Rust inference)."""
    with open(path, 'wb') as f:
        pickle.dump(state, f)
    print(f"Saved: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: convert_to_pickle.py <input.bin> <output.pkl>")
        sys.exit(1)
    
    state = read_binary_model(sys.argv[1])
    save_pickle(state, sys.argv[2])
