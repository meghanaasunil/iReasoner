# iREASONER: Trajectory-Aware Intrinsic Reasoning Supervision for Self-Evolving Large Multimodal Models

Code implementation for iReasoner: a self-evolving **Proposer–Solver** training loop for large multimodal models (LMMs) using unlabeled images. The Solver is optimized with intrinsic rewards that combine:

1. **Outcome-level self-consistency** (agreement on final answers)
2. **Trajectory-aware reasoning agreement** (agreement across intermediate reasoning steps inside `<think>`)

> This codebase is built on top of [EvoLMM](https://github.com/mbzuai-oryx/EvoLMM). <br>
> **Evaluation**: The evaluation pipeline follows the same setup as https://github.com/mbzuai-oryx/EvoLMM (lmms-eval-based harness).


## Repository Layout

- `config.py`: Configuration dataclass and defaults
- `data_and_utils.py`: Image loader, prompts, parsing, and small utilities
- `vlm.py`: Model wrapper for generation, LoRA adapters, and text embeddings for step agreement
- `rewards.py`: Reasoning-step agreement reward (trajectory-aware intrinsic signal)
- `train.py`: Training loop (Proposer–Solver), REINFORCE + KL, checkpoints, resume, logging
- `run.py`: (Optional) Thin wrapper entrypoint if you want `python run.py ...`



## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) HuggingFace Cache/Tokens

```bash
export HF_HOME=/path/to/cache
export HF_TOKEN=<your_hf_token>
```


## Data Preparation

Training requires only images (no labels). The loader scans `--data_dir` recursively and expects either:

- Images directly under the directory, or
- First-level subfolders under the directory

**Example structure:**

```
/path/to/data/
  split1/
    img_001.jpg
  split2/
    img_002.png
```

You can restrict training to specific first-level subfolders using:

```bash
--include_subfolders=split1,split2
```

Corrupted images are skipped; sampling is deterministic given `--seed`.



## Training

### Example: Qwen2.5-VL-7B with LoRA on Multi-GPU

```bash
python train.py \
  --data_dir "/path/to/data" \
  --wandb_mode online \
  --wandb_project iReasoner \
  --wandb_run_name iReasoner \
  --solver_model Qwen/Qwen2.5-VL-7B-Instruct \
  --proposer_model Qwen/Qwen2.5-VL-7B-Instruct \
  --use_lora_solver --use_lora_proposer \
  --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,mm_projector \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --num_solver_samples 5 \
  --proposer_update_freq 5 \
  --total_steps 2500 \
  --kl_target 0.020 \
  --kl_adapt_rate 0.10 \
  --solver_soft_gamma 0.7 \
  --clear_cache_every 5
```

### Training Notes

- Checkpoints and per-iteration logs are saved under `runs/<wandb_run_name>/`
- Auto-resume is supported: keep `--wandb_run_name` fixed and ensure previous checkpoints exist under `runs/`
- For multi-GPU sharding, keep `--device_map auto` (default)


## Checkpoints

Checkpoints are stored in the following structure:

```
runs/<run_name>/step_00050/
  solver/
  proposer/   (or solver if shared backbone)
  trainer_state.pt
  SAVE_OK
```

The checkpoint also stores optimizer state and RNG state for stable resumes.


## Evaluation

Evaluation follows the same harness and methodology as EvoLMM: https://github.com/mbzuai-oryx/EvoLMM

In particular, the evaluation uses the lmms-eval-based setup located in `Evaluation/lmms-eval` in the EvoLMM repository. Use the solver checkpoint path (LoRA directory) as the `lora_path` argument.



## Acknowledgments

This work builds upon [EvoLMM](https://github.com/mbzuai-oryx/EvoLMM).
