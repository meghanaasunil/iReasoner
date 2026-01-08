from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "mm_projector",
)


@dataclass
class Config:
    # Models (Proposer–Solver loop)
    solver_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    proposer_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Device / precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    device_map: str = "auto"

    # Training
    total_steps: int = 100
    batch_size: int = 1
    lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    proposer_update_freq: int = 2

    # KL regularization to a frozen reference policy
    kl_coef: float = 1e-3
    kl_target: float = 0.02
    kl_adapt_rate: float = 0.10

    # Sampling
    temp: float = 1.0
    top_p: float = 1.0
    max_new_tokens_solver: int = 256
    max_new_tokens_proposer: int = 128
    num_solver_samples: int = 5

    # Intrinsic outcome reward shaping
    len_penalty_weight: float = 0.05
    len_penalty_target_words: int = 64
    solver_soft_gamma: float = 0.7

    # Proposer intrinsic reward (encourage “good” question difficulty via answer entropy)
    prop_entropy_mu: float = 0.90
    prop_entropy_sigma: float = 0.35

    # Trajectory-aware reasoning reward (step agreement within the dominant answer group)
    max_reasoning_steps: int = 8
    step_embed_max_tokens: int = 64
    min_step_group_size: int = 2
    step_group_density_gamma: float = 0.7
    step_position_decay: float = 0.15
    step_sim_temperature: float = 1.0

    # Mix outcome reward and reasoning-step agreement reward
    step_reward_w_start: float = 0.0
    step_reward_w_end: float = 0.7
    step_reward_warmup_frac: float = 0.3

    # Data / IO
    data_dir: str = "images/train"
    output_dir: str = "runs"
    save_every: int = 50
    max_checkpoints: int = 2
    include_subfolders: Optional[Tuple[str, ...]] = None

    # Freezing
    freeze_vision: bool = True

    # Repro
    seed: int = 42

    # LoRA
    use_lora_solver: bool = False
    use_lora_proposer: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = DEFAULT_LORA_TARGETS
    load_solver_adapter: Optional[str] = None
    load_proposer_adapter: Optional[str] = None
    start_step: int = 0

    # W&B
    wandb_mode: str = "disabled"
    wandb_project: str = "ireasoner"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_log_images_every: int = 0

    # Memory/fragmentation guard
    clear_cache_every: int = 25
