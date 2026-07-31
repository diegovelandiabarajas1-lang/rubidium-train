#!/usr/bin/env python3
"""
RUBIDIUM - Convert U:/B: Corpus to Fine-tuning Format
Converts chat_XX.txt and corpus_XX.txt to instruction pairs
"""
import os
import json
import re
from pathlib import Path

def parse_ub_file(filepath):
    """Parse U:/B: format file into (user, bot) pairs"""
    pairs = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    current_u = None
    current_b = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('U: '):
            if current_u is not None and current_b is not None:
                pairs.append({
                    "user": current_u,
                    "bot": current_b
                })
            current_u = line[3:]
            current_b = None
        elif line.startswith('B: '):
            current_b = line[3:]
    
    if current_u is not None and current_b is not None:
        pairs.append({
            "user": current_u,
            "bot": current_b
        })
    
    return pairs

def convert_corpus_to_finetune(corpus_dir, output_dir):
    """Convert all corpus files to fine-tuning format"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_pairs = []
    
    for filename in os.listdir(corpus_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(corpus_dir, filename)
            pairs = parse_ub_file(filepath)
            
            if pairs:
                all_pairs.extend(pairs)
                print(f"  {filename}: {len(pairs)} pairs")
    
    # Save as JSONL (one JSON per line)
    output_file = os.path.join(output_dir, 'finetune_data.jsonl')
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"\nTotal pairs: {len(all_pairs)}")
    print(f"Saved to: {output_file}")
    
    # Also save as instruction format
    instruct_file = os.path.join(output_dir, 'finetune_instructions.jsonl')
    with open(instruct_file, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            # Format for different templates
            formats = {
                "alpaca": {
                    "instruction": pair["user"],
                    "input": "",
                    "output": pair["bot"]
                },
                "chatml": {
                    "messages": [
                        {"role": "user", "content": pair["user"]},
                        {"role": "assistant", "content": pair["bot"]}
                    ]
                },
                "simple": {
                    "prompt": f"U: {pair['user']} B: {pair['bot']}"
                }
            }
            f.write(json.dumps(formats, ensure_ascii=False) + '\n')
    
    print(f"Instruction formats saved to: {instruct_file}")
    
    return all_pairs

def split_train_val(pairs, val_ratio=0.1):
    """Split into train/val"""
    import random
    random.seed(42)
    random.shuffle(pairs)
    
    split_idx = int(len(pairs) * (1 - val_ratio))
    train = pairs[:split_idx]
    val = pairs[split_idx:]
    
    return train, val

if __name__ == "__main__":
    import sys
    
    corpus_dir = sys.argv[1] if len(sys.argv) > 1 else "resources"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data_finetune"
    
    print("=" * 60)
    print("RUBIDIUM - Corpus to Fine-tuning Converter")
    print("=" * 60)
    print(f"Input: {corpus_dir}")
    print(f"Output: {output_dir}")
    
    pairs = convert_corpus_to_finetune(corpus_dir, output_dir)
    
    # Split
    train, val = split_train_val(pairs, val_ratio=0.1)
    
    # Save splits
    for name, data in [("train", train), ("val", val)]:
        with open(os.path.join(output_dir, f"{name}.jsonl"), 'w', encoding='utf-8') as f:
            for pair in data:
                f.write(json.dumps({"user": pair["user"], "bot": pair["bot"]}, ensure_ascii=False) + '\n')
    
    print(f"\nTrain: {len(train)} pairs")
    print(f"Val: {len(val)} pairs")
    print("Done!")