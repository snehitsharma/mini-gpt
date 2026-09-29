"""
GPT model: token + positional embeddings, causal self-attention,
feed-forward blocks, and the GPT class that stacks them.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import BLOCK_SIZE, DEVICE

VOCAB_SIZE = 50257  # GPT-2 BPE vocab size (matches GPT2Tokenizer)
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
FFN_HIDDEN = N_EMBD * 4
DROPOUT = 0.1


class Embeddings(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE, n_embd=N_EMBD):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)  # broadcasts over batch
        return tok_emb + pos_emb


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd=N_EMBD, n_head=N_HEAD):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.out_proj.RESIDUAL_SCALE_INIT = True  # writes into the residual stream, needs scaled init

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scaled_dot_product_attention uses flash/memory-efficient attention
        # on CUDA -- never materializes the full (T, T) score matrix, which
        # is what keeps this from blowing up VRAM at block_size=1024.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, n_embd=N_EMBD, hidden=FFN_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_embd),
        )
        self.net[2].RESIDUAL_SCALE_INIT = True  # writes into the residual stream, needs scaled init

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd=N_EMBD, n_head=N_HEAD, dropout=DROPOUT):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop1(self.attn(self.ln1(x)))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE,
                 n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, block_size, n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # tie output projection to token embedding weights (standard GPT-2
        # trick) -- saves ~25.7M params by not learning a second copy
        self.lm_head.weight = self.embeddings.token_embedding.weight

        self.n_layer = n_layer
        self.block_size = block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # GPT-2 style init: default PyTorch Linear/Embedding init isn't
        # scaled for a residual stack this deep -- without this, the
        # residual stream blows up across layers and the model starts
        # training from a much worse point than a random model would
        # (observed: loss ~250 instead of the expected ln(vocab_size)~10.8).
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "RESIDUAL_SCALE_INIT"):
                std /= math.sqrt(2 * self.n_layer)  # scale down projections that write into the residual stream
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.embeddings(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    import time
    from data import get_batch, BATCH_SIZE

    model = GPT().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {n_params:,} (lm_head tied to token_embedding, so counted once)")

    x, y = get_batch("train")
    print(f"input batch: {x.shape}")

    # no_grad here: this is just a shape/sanity check, not a real training
    # step -- without no_grad the autograd graph stays alive (referenced by
    # `logits`/`loss`) for the rest of the script and wastes VRAM.
    with torch.no_grad():
        logits, loss = model(x, y)
    print(f"logits: {logits.shape}  loss: {loss.item():.4f}")
    del logits, loss

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    amp_dtype = torch.bfloat16  # RTX 40-series has native bf16 tensor cores, no GradScaler needed

    n_steps = 20
    t0 = time.time()
    for _ in range(n_steps):
        x, y = get_batch("train")
        with torch.autocast(device_type=DEVICE, dtype=amp_dtype, enabled=(DEVICE == "cuda")):
            logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    step_time = (t1 - t0) / n_steps
    print(f"\n{n_steps} full train steps (fwd+bwd+optim): {t1 - t0:.2f}s -> {step_time*1000:.1f} ms/step")
    print(f"final loss: {loss.item():.4f}")

    if DEVICE == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"peak VRAM: {peak_gb:.2f} GB / {total_gb:.2f} GB")

    tokens_per_step = BATCH_SIZE * BLOCK_SIZE
    train_tokens = 2_403_644  # from prepare.py output
    steps_per_epoch = train_tokens / tokens_per_step
    epoch_time_min = steps_per_epoch * step_time / 60
    print(f"\n~{steps_per_epoch:.0f} steps/epoch -> ~{epoch_time_min:.1f} min/epoch (rough, no dataloader overhead)")
