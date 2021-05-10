from dataclasses import dataclass

@dataclass
class Config:
    lr: int
    epochs: int
    batch_size: int
    seed: int
    eval: bool
    augmentation: str
    criterion: str
    optimizer: str
    model: str
    eval_load: str
    continue_load: str
    dataset_path: str