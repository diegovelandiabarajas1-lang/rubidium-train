#!/usr/bin/env python3
"""
RUBIDIUM - Entrenamiento en AMD Developer Cloud
Script principal con soporte ROCm/HIP
"""
import os
import sys
import time
import json
import numpy as np
from pathlib import Path

# ============================================================
# VERIFICAR ENTORNO
# ============================================================
def check_environment():
    print("=" * 60)
    print("RUBIDIUM - Entrenamiento AMD Developer Cloud")
    print("=" * 60)
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
            print(f"CUDA/ROCm: {torch.version.cuda or torch.version.hip}")
            return True
        else:
            print("ERROR: GPU no disponible")
            return False
    except ImportError:
        print("ERROR: PyTorch no instalado")
        return False

# ============================================================
# CARGAR TOKENIZER
# ============================================================
def load_tokenizer(tokenizer_path: str):
    print(f"\nCargando tokenizer: {tokenizer_path}")
    
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    vocab = data['vocab']
    merges = data['merges']
    
    print(f"  Vocabulario: {len(vocab)} tokens")
    print(f"  Mergeos: {len(merges)}")
    
    return vocab, merges

# ============================================================
# CARGAR CORPUS TOKENIZADO
# ============================================================
def load_corpus(data_dir: str):
    print(f"\nCargando corpus tokenizado: {data_dir}")
    
    all_tokens = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.bin'):
            filepath = os.path.join(data_dir, filename)
            tokens = np.fromfile(filepath, dtype=np.int32)
            all_tokens.extend(tokens.tolist())
            print(f"  + {filename}: {len(tokens):,} tokens")
    
    print(f"Total tokens: {len(all_tokens):,}")
    return all_tokens

# ============================================================
# MODELO TRANSFORMER (PyTorch)
# ============================================================
class RubidiumTransformerPyTorch:
    """
    Transformer implementado en PyTorch para AMD Developer Cloud
    Compatible con pesos del motor CUDA original
    """
    
    def __init__(self, vocab_size=32000, d_model=2048, n_heads=32, 
                 n_layers=10, d_ff=8192, max_seq_len=512):
        import torch
        import torch.nn as nn
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // n_heads
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        # Final layer norm + LM head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True)
        
        # Inicializar pesos
        self.apply(self._init_weights)
        
        # Contar parámetros
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Parámetros: {n_params/1e6:.1f}M")
    
    def _init_weights(self, module):
        import torch
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
    
    def forward(self, idx, targets=None):
        import torch
        
        B, T = idx.shape
        
        # Embeddings
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Loss
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=0  # PAD
            )
            return logits, loss
        
        return logits
    
    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=40):
        import torch
        
        for _ in range(max_new_tokens):
            # Crop a max_seq_len
            idx_crop = idx[:, -self.max_seq_len:]
            
            # Forward
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


class TransformerLayer:
    """Capa transformer con attention + FFN"""
    
    def __init__(self, d_model, n_heads, d_ff):
        import torch
        import torch.nn as nn
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    
    def __call__(self, x):
        # Attention with residual
        x = x + self.attn(self.ln1(x))
        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


class MultiHeadAttention:
    """Multi-head self-attention"""
    
    def __init__(self, d_model, n_heads):
        import torch
        import torch.nn as nn
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
    
    def __call__(self, x):
        import torch
        
        B, T, C = x.shape
        
        # QKV
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = torch.nn.functional.softmax(att, dim=-1)
        att = torch.nn.functional.dropout(att, p=0.1, training=self.training)
        
        # Weighted sum
        y = (att @ v).reshape(B, T, C)
        y = self.proj(y)
        
        return y


# ============================================================
# LOOP DE ENTRENAMIENTO
# ============================================================
def train(model, data, config):
    import torch
    import torch.optim as optim
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        betas=(config['b1'], config['b2']),
        eps=config['eps'],
        weight_decay=config['wd']
    )
    
    # Scheduler
    def lr_lambda(step):
        if step < config['warmup']:
            return step / config['warmup']
        else:
            p = (step - config['warmup']) / (config['max_steps'] - config['warmup'])
            return 0.5 * (1 + math.cos(math.pi * p))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    print("\n" + "=" * 60)
    print("INICIANDO ENTRENAMIENTO")
    print("=" * 60)
    
    smooth_loss = float('inf')
    start_time = time.time()
    
    for step in range(1, config['max_steps'] + 1):
        # Gradient accumulation
        optimizer.zero_grad()
        
        loss_acc = 0
        for ga in range(config['ga']):
            # Sample batch
            idx = np.random.randint(0, len(data) - config['seq_len'] - 1, 
                                   size=(config['batch_size'],))
            
            x = torch.stack([
                torch.tensor(data[i:i+config['seq_len']], dtype=torch.long)
                for i in idx
            ]).to(device)
            
            y = torch.stack([
                torch.tensor(data[i+1:i+config['seq_len']+1], dtype=torch.long)
                for i in idx
            ]).to(device)
            
            # Forward
            _, loss = model(x, y)
            loss = loss / config['ga']
            loss.backward()
            
            loss_acc += loss.item()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gc'])
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        
        # Logging
        avg_loss = loss_acc
        smooth_loss = 0.98 * smooth_loss + 0.02 * avg_loss if smooth_loss != float('inf') else avg_loss
        
        if step % 100 == 0 or step == config['max_steps']:
            elapsed = time.time() - start_time
            sps = step / elapsed
            eta = (config['max_steps'] - step) / sps / 60
            
            print(f"Step {step}/{config['max_steps']} | loss: {smooth_loss:.4f} | "
                  f"lr: {scheduler.get_last_lr()[0]:.2e} | "
                  f"{sps:.1f} steps/s | ETA: {eta:.0f}min")
        
        # Checkpoint
        if step % 5000 == 0:
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }
            path = f'checkpoints/model_step_{step}.pt'
            torch.save(checkpoint, path)
            print(f"  Checkpoint: {path}")
    
    # Save final
    final_path = 'model_final.pt'
    torch.save(model.state_dict(), final_path)
    print(f"\nModelo guardado: {final_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    # Verificar entorno
    if not check_environment():
        sys.exit(1)
    
    # Configuración
    config = {
        'vocab_size': 32000,
        'd_model': 2048,
        'n_heads': 32,
        'n_layers': 10,
        'd_ff': 8192,
        'seq_len': 512,
        'batch_size': 2,
        'ga': 16,
        'max_steps': 200000,
        'lr': 3e-4,
        'b1': 0.9,
        'b2': 0.999,
        'eps': 1e-8,
        'wd': 0.1,
        'warmup': 6000,
        'gc': 1.0,
    }
    
    print(f"\nConfiguración:")
    print(f"  Vocab: {config['vocab_size']}")
    print(f"  D: {config['d_model']}")
    print(f"  H: {config['n_heads']}")
    print(f"  L: {config['n_layers']}")
    print(f"  FF: {config['d_ff']}")
    print(f"  T: {config['seq_len']}")
    print(f"  BS: {config['batch_size']}")
    print(f"  GA: {config['ga']}")
    print(f"  Steps: {config['max_steps']}")
    
    # Cargar tokenizer
    vocab, merges = load_tokenizer('tokenizer.json')
    config['vocab_size'] = len(vocab)
    
    # Cargar corpus
    data = load_corpus('data/')
    
    # Crear modelo
    model = RubidiumTransformerPyTorch(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['seq_len']
    )
    
    # Entrenar
    train(model, data, config)


if __name__ == "__main__":
    main()
