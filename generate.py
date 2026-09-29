"""
Sample text from a trained checkpoint.

Usage:
    python generate.py --checkpoint checkpoints/checkpoint_epoch2_final.pt --prompt "The history of"
"""

import argparse
import sys

import torch
from transformers import GPT2Tokenizer

from config import Config
from model import GPT

# our own checkpoints embed a Config dataclass; PyTorch 2.6 defaults
# torch.load to weights_only=True, which refuses to unpickle it unless
# explicitly allowlisted (safe here since we only ever load our own files)
torch.serialization.add_safe_globals([Config])

sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]

    model = GPT(vocab_size=config.vocab_size, block_size=config.block_size,
                n_embd=config.n_embd, n_head=config.n_head, n_layer=config.n_layer).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt['epoch']}, step {ckpt['step']})")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)

    out = model.generate(idx, max_new_tokens=args.max_new_tokens,
                          temperature=args.temperature, top_k=args.top_k)
    text = tokenizer.decode(out[0].tolist())

    print(f"\n--- prompt: {args.prompt!r} (temperature={args.temperature}, top_k={args.top_k}) ---\n")
    print(text)


if __name__ == "__main__":
    main()
