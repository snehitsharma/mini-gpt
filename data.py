"""
get_batch() for language-model training: samples random block_size windows
from the flattened token stream, with y offset by 1 from x (next-token
prediction target). Sampling is random-with-replacement rather than a
strict epoch pass, since the dataset is small relative to block_size /
num training steps.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

BLOCK_SIZE = 512
BATCH_SIZE = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(split):
    # memmap re-opened each call (cheap) so we don't leak file handles /
    # keep a giant array pinned in memory the whole run.
    data = np.memmap(f"{split}.bin", dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i:i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK_SIZE].astype(np.int64)) for i in ix])

    if DEVICE == "cuda":
        x = x.pin_memory().to(DEVICE, non_blocking=True)
        y = y.pin_memory().to(DEVICE, non_blocking=True)
    else:
        x, y = x.to(DEVICE), y.to(DEVICE)

    return x, y


class TokenDataset(Dataset):
    """
    Non-overlapping block_size windows over a token .bin file, giving a
    well-defined "one epoch = one pass over the data" -- unlike get_batch's
    random-with-replacement sampling, which is only for quick benchmarks.
    """

    def __init__(self, bin_path, block_size):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.n_windows = (len(self.data) - 1) // block_size

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = torch.from_numpy(self.data[start:start + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[start + 1:start + 1 + self.block_size].astype(np.int64))
        return x, y


if __name__ == "__main__":
    x, y = get_batch("train")
    print("x shape:", x.shape, "y shape:", y.shape)
    print("x[0][:10]:", x[0][:10])
    print("y[0][:10]:", y[0][:10])
    assert torch.equal(x[0][1:10], y[0][0:9]), "y should be x shifted by 1"
    print("offset check passed: y[i] == x[i+1]")
