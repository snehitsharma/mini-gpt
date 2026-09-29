"""
Tokenize wikitext-2 once and cache train/val token ids to disk as .bin files
(uint16, since GPT-2's vocab of 50257 fits in 16 bits). get_batch() in
data.py memory-maps these so we never have to re-tokenize between runs.
"""

import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer


def tokenize_split(dataset, tokenizer, split):
    text = "\n".join(dataset[split]["text"])
    ids = tokenizer.encode(text)
    return np.array(ids, dtype=np.uint16)


def main():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    train_ids = tokenize_split(dataset, tokenizer, "train")
    val_ids = tokenize_split(dataset, tokenizer, "validation")

    train_ids.tofile("train.bin")
    val_ids.tofile("val.bin")

    print(f"train: {len(train_ids):,} tokens -> train.bin")
    print(f"val:   {len(val_ids):,} tokens -> val.bin")


if __name__ == "__main__":
    main()
