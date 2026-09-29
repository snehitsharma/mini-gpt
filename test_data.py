"""
Quick sanity check for the Hugging Face data pipeline: download wikitext-2,
tokenize a sample with GPT2Tokenizer, and print results end to end.
"""

import sys

from datasets import load_dataset
from transformers import GPT2Tokenizer

# Windows terminals default to cp1252, which can't encode some unicode
# characters (e.g. macrons) that show up in wikitext articles.
sys.stdout.reconfigure(encoding="utf-8")

# caps tokenization work so this stays a quick sanity check, not a full pass
MAX_TOKENS = 10_000_000


def main():
    print("Loading wikitext-2 dataset from Hugging Face...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    print("\n=== Dataset structure ===")
    print(dataset)

    print("\nLoading GPT2Tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("\n=== Vocab size ===")
    print(tokenizer.vocab_size)

    train_split = dataset["train"]
    sample_text = ""
    for line in train_split["text"]:
        if line.strip():
            sample_text += line
        if len(sample_text) > 2000:
            break

    print("\n=== Sample text ===")
    print(sample_text)

    print("\nTokenizing sample text...")
    tokens = tokenizer.encode(sample_text)
    tokens = tokens[:MAX_TOKENS]

    print(f"\nTotal tokens in sample: {len(tokens)} (capped at {MAX_TOKENS:,})")

    print("\n=== Sample tokens (first 20) ===")
    print(tokens[:20])

    print("\n=== Decoded tokens back to text ===")
    decoded = tokenizer.decode(tokens[:20])
    print(decoded)


if __name__ == "__main__":
    main()
