"""
Config dataclass + yaml loader. `python train.py --config config.yaml`
overrides these defaults with whatever keys the yaml file sets.
"""

from dataclasses import asdict, dataclass, fields

import yaml


@dataclass
class Config:
    # data
    train_bin: str = "train.bin"
    val_bin: str = "val.bin"
    block_size: int = 512
    batch_size: int = 16

    # model
    vocab_size: int = 50257
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.1

    # training
    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 50
    checkpoint_every: int = 200
    checkpoint_dir: str = "checkpoints"
    log_file: str = "training_log.csv"

    seed: int = 42
    device: str = "cuda"


def load_config(path=None):
    config = Config()
    if path is None:
        return config
    with open(path, "r") as f:
        overrides = yaml.safe_load(f) or {}
    valid_keys = {f.name for f in fields(Config)}
    unknown = set(overrides) - valid_keys
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {unknown}")
    return Config(**{**asdict(config), **overrides})
