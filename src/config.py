# config.py
from dataclasses import dataclass
from typing import List

# ====== DATA PATHS (Kaggle) ======
TRAIN_PATH = "./data/train.csv"
VAL_PATH   = "./data/val.csv"
TEST_PATH  = "./data/test.csv"

TEXT_COL = "comment"

LABELS: List[str] = [
    "antagonize",
    "condescending",
    "dismissive",
    "generalisation",
    "generalisation_unfair",
    "healthy",
    "hostile",
    "sarcastic",
]


@dataclass
class TrainConfig:
    # model / prompt tuning
    model_name: str = "t5-base"
    num_virtual_tokens: int = 10
    prompt_init_text: str = "Answer the question with yes or no."
    max_source_len: int = 256
    max_target_len: int = 4

    # training
    output_dir: str = "./t5_prompt_tuning_selected_labels"
    save_dir: str = "./t5_prompt_adapter_selected_labels"
    epochs: int = 5
    lr: float = 5e-3
    weight_decay: float = 1e-5
    train_bs: int = 16
    eval_bs: int = 32
    eval_each_epoch: bool = True
    eval_stream_bs: int = 8  
    fp16: bool = True
    seed: int = 42
    logging_steps: int = 100
    save_total_limit: int = 2
