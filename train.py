from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import pathlib
import random
import re
import shutil
import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import set_seed
from PIL import Image

from config import Config, DEFAULT_LORA_TARGETS
from data_and_utils import (
    ImagePool,
    build_proposer_prompt,
    build_solver_prompt,
    clip_grad_norm_multi_device,
    extract_steps,
    gaussian_reward,
    majority_vote,
    normalize_answer,
    pre_answer_word_count,
    shannon_entropy_nats,
    strip_tags,
)
from rewards import compute_step_agreement_rewards
from vlm import VLMCore, VLMRole

try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False


class PolicyUpdater:
    def __init__(self, policy: VLMRole, ref_policy: VLMRole, cfg: Config, *, adapter_name: Optional[str] = None):
        self.policy = policy
        self.ref_policy = ref_policy
        self.cfg = cfg
        self.kl_coef = cfg.kl_coef
        self._step = 0
        params = self._collect_trainable_params(self.policy.core.model, adapter_name)
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    @staticmethod
    def _collect_trainable_params(model: nn.Module, adapter_name: Optional[str]) -> Iterable[nn.Parameter]:
        found, selected = False, []
        for n, p in model.named_parameters():
            if p.requires_grad:
                if adapter_name is not None and f".{adapter_name}." in n:
                    selected.append(p)
                    found = True
        if found:
            return selected
        return [p for p in model.parameters() if p.requires_grad]

    def _adapt_beta(self, kl_val: float) -> None:
        tgt = max(self.cfg.kl_target, 1e-8)
        delta = (kl_val - tgt) / tgt
        self.kl_coef = float(
            min(max(self.kl_coef * math.exp(self.cfg.kl_adapt_rate * delta), 1e-8), 1e2)
        )

    def step(self, image: Image.Image, prompt: str, completion: str, reward: float, baseline: float = 0.0) -> Dict[str, float]:
        core = self.policy.core
        device = torch.device(core.primary_device)
        self._step += 1

        chat_prompt = core._render_chat(image, prompt, add_generation_prompt=True)
        chat_full = chat_prompt + completion

        inputs_full = core.processor(text=chat_full, images=[image], return_tensors="pt").to(device)
        inputs_prompt = core.processor(text=chat_prompt, images=[image], return_tensors="pt").to(device)

        input_ids = inputs_full["input_ids"]
        attn = inputs_full.get("attention_mask")
        labels = input_ids.clone()
        prompt_len = inputs_prompt["input_ids"].shape[1]
        labels[:, :prompt_len] = -100

        shift_labels = labels[:, 1:].contiguous()
        valid_mask = shift_labels != -100

        core.model.train(True)
        with core.use_adapter(self.policy.adapter_name):
            out_pi = core.model(input_ids=input_ids, attention_mask=attn, labels=labels)
        ce_loss = out_pi.loss

        logp_pi = F.log_softmax(out_pi.logits, dim=-1)
        with torch.no_grad(), core.use_adapter(None):
            out_ref = core.model(input_ids=input_ids, attention_mask=attn)
            logp_ref = F.log_softmax(out_ref.logits, dim=-1)

        logp_pi_shift = logp_pi[:, :-1, :]
        logp_ref_shift = logp_ref[:, :-1, :]
        p_pi_shift = logp_pi_shift.exp()
        kl_per_tok = (p_pi_shift * (logp_pi_shift - logp_ref_shift)).sum(dim=-1)
        kl_loss = kl_per_tok[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, device=ce_loss.device)

        advantage = float(reward - baseline)
        beta_used = float(self.kl_coef)
        loss_total = advantage * ce_loss + beta_used * kl_loss

        self.opt.zero_grad(set_to_none=True)
        loss_total.backward()
        clip_grad_norm_multi_device(core.model, self.cfg.grad_clip)
        self.opt.step()
        core.model.train(False)

        kl_val = float(kl_loss.item())
        beta_before = beta_used
        self._adapt_beta(kl_val)

        if torch.cuda.is_available() and self.cfg.clear_cache_every > 0 and (self._step % self.cfg.clear_cache_every == 0):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            gc.collect()

        return {
            "ce_loss": float(ce_loss.item()),
            "kl_loss": kl_val,
            "advantage": advantage,
            "kl_coef_before": beta_before,
            "kl_coef_after": float(self.kl_coef),
            "loss_total": float(loss_total.item()),
        }


def _parse_step_num(name: str) -> int:
    m = re.match(r"^step_(\d+)$", name)
    return int(m.group(1)) if m else -1


def _is_complete_ckpt(step_dir: str) -> bool:
    if not os.path.isdir(step_dir):
        return False
    if not os.path.isfile(os.path.join(step_dir, "SAVE_OK")):
        return False
    solver_dir = os.path.join(step_dir, "solver")
    if not os.path.isdir(solver_dir):
        return False
    has_meta = os.path.isfile(os.path.join(solver_dir, "checkpoint_meta.json"))
    has_any_weight = any(
        os.path.isfile(os.path.join(solver_dir, f))
        for f in (
            "adapter_config.json",
            "adapter_model.bin",
            "adapter_model.safetensors",
            "pytorch_model.bin",
            "model.safetensors",
            "config.json",
        )
    )
    return has_meta or has_any_weight


def _list_valid_ckpts(run_dir: str) -> List[Tuple[int, str]]:
    if not os.path.isdir(run_dir):
        return []
    pairs: List[Tuple[int, str]] = []
    for d in os.listdir(run_dir):
        step = _parse_step_num(d)
        if step >= 0:
            full = os.path.join(run_dir, d)
            if _is_complete_ckpt(full):
                pairs.append((step, full))
    return sorted(pairs, key=lambda x: x[0])


def _retain_last_k_checkpoints(run_dir: str, k: int) -> None:
    k = max(2, int(k))
    ckpts = _list_valid_ckpts(run_dir)
    if len(ckpts) <= k:
        return
    for _step, path in ckpts[:-k]:
        shutil.rmtree(path, ignore_errors=True)
        print(f"[Checkpoint] Pruned: {os.path.basename(path)}")


def _read_meta_is_lora(dir_path: str) -> Optional[bool]:
    meta_path = os.path.join(dir_path, "checkpoint_meta.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return bool(meta.get("is_lora"))
    except Exception:
        return None


def _find_preferred_ckpt(run_dir: str) -> Tuple[Optional[int], Optional[str]]:
    ckpts = _list_valid_ckpts(run_dir)
    if not ckpts:
        return (None, None)
    if len(ckpts) >= 2:
        return ckpts[-2]
    return ckpts[-1]


def _prune_other_checkpoints(run_dir: str, keep_step: int) -> None:
    keep_name = f"step_{keep_step:05d}"
    for d in os.listdir(run_dir):
        if d.startswith("step_") and d != keep_name:
            shutil.rmtree(os.path.join(run_dir, d), ignore_errors=True)


def maybe_autoresume(cfg: Config) -> Config:
    run_name = cfg.wandb_run_name
    if not run_name:
        return cfg

    run_dir = os.path.join(cfg.output_dir, run_name)
    step, step_dir = _find_preferred_ckpt(run_dir)
    if step is None or step_dir is None:
        return cfg

    solver_dir = os.path.join(step_dir, "solver")
    proposer_dir = os.path.join(step_dir, "proposer")
    if not os.path.isdir(proposer_dir):
        proposer_dir = solver_dir

    is_lora = _read_meta_is_lora(solver_dir)
    print(f"[Auto-Resume] Using checkpoint: {step_dir} (is_lora={is_lora})")
    cfg.start_step = int(step)

    if is_lora is True:
        cfg.use_lora_solver = True
        cfg.use_lora_proposer = True
        cfg.load_solver_adapter = solver_dir
        cfg.load_proposer_adapter = proposer_dir
    elif is_lora is False:
        cfg.solver_model_name = solver_dir
        cfg.proposer_model_name = proposer_dir
        cfg.use_lora_solver = False
        cfg.use_lora_proposer = False
        cfg.load_solver_adapter = None
        cfg.load_proposer_adapter = None

    try:
        _prune_other_checkpoints(run_dir, keep_step=cfg.start_step)
        print(f"[Auto-Resume] Pruned others; kept step_{cfg.start_step:05d}")
    except Exception as e:
        print(f"[Auto-Resume] Prune skipped: {e}")

    cfg._resume_dir = step_dir  # type: ignore[attr-defined]
    return cfg


class IReasonerTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.seed)
        random.seed(cfg.seed)

        share_backbone = (
            cfg.use_lora_solver
            and cfg.use_lora_proposer
            and (cfg.solver_model_name == cfg.proposer_model_name)
        )

        if share_backbone:
            print("[Models] Shared backbone with two adapters (solver=default, proposer=proposer).")
            core = VLMCore(
                cfg.solver_model_name,
                cfg.device,
                cfg.dtype,
                cfg,
                apply_lora=True,
                make_adapters=["proposer"],
            )
            self.solver = VLMRole(core, adapter_name="default")
            self.proposer = VLMRole(core, adapter_name="proposer")
            self.solver_ref = VLMRole(core, adapter_name=None)
            self.proposer_ref = VLMRole(core, adapter_name=None)
        else:
            print("[Models] Separate backbones (with frozen refs).")
            solver_core = VLMCore(cfg.solver_model_name, cfg.device, cfg.dtype, cfg, apply_lora=cfg.use_lora_solver)
            proposer_core = VLMCore(cfg.proposer_model_name, cfg.device, cfg.dtype, cfg, apply_lora=cfg.use_lora_proposer)

            self.solver = VLMRole(solver_core, adapter_name=("default" if solver_core.is_lora else None))
            self.proposer = VLMRole(proposer_core, adapter_name=("default" if proposer_core.is_lora else None))

            self.solver_ref = VLMRole(VLMCore(cfg.solver_model_name, cfg.device, cfg.dtype, cfg, apply_lora=False), adapter_name=None)
            self.proposer_ref = VLMRole(VLMCore(cfg.proposer_model_name, cfg.device, cfg.dtype, cfg, apply_lora=False), adapter_name=None)

        self.solver_updater = PolicyUpdater(self.solver, self.solver_ref, cfg, adapter_name=self.solver.adapter_name)
        self.proposer_updater = PolicyUpdater(self.proposer, self.proposer_ref, cfg, adapter_name=self.proposer.adapter_name)

        self.pool = ImagePool(cfg)

        self.solver_baseline = 0.0
        self.proposer_baseline = 0.0
        self.momentum = 0.9

        self.run_name = cfg.wandb_run_name or f"{pathlib.Path(cfg.solver_model_name).name}_{int(time.time())}"
        self.run_dir = os.path.join(cfg.output_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        resume_dir = getattr(cfg, "_resume_dir", None)
        if resume_dir and os.path.isdir(resume_dir):
            state_path = os.path.join(resume_dir, "trainer_state.pt")
            if os.path.isfile(state_path):
                self._load_trainer_state(state_path)

        self.wandb_run = None
        if HAS_WANDB and cfg.wandb_mode != "disabled":
            self.wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=self.run_name,
                mode=cfg.wandb_mode,
                config=dataclasses.asdict(cfg),
            )
            try:
                wandb.watch(self.solver.core.model, log="gradients", log_freq=100)
            except Exception:
                pass

    def _load_trainer_state(self, state_path: str) -> None:
        try:
            state = torch.load(state_path, map_location="cpu")
            if "solver_opt" in state:
                self.solver_updater.opt.load_state_dict(state["solver_opt"])
            if "proposer_opt" in state:
                self.proposer_updater.opt.load_state_dict(state["proposer_opt"])
            if "solver_kl_coef" in state:
                self.solver_updater.kl_coef = float(state["solver_kl_coef"])
            if "proposer_kl_coef" in state:
                self.proposer_updater.kl_coef = float(state["proposer_kl_coef"])
            self.solver_baseline = float(state.get("solver_baseline", 0.0))
            self.proposer_baseline = float(state.get("proposer_baseline", 0.0))
            self.solver_updater._step = int(state.get("solver_updater_step", self.cfg.start_step))
            self.proposer_updater._step = int(state.get("proposer_updater_step", self.cfg.start_step))
            if "py_random_state" in state:
                random.setstate(state["py_random_state"])
            if "torch_rng_state" in state:
                torch.set_rng_state(state["torch_rng_state"])
            if torch.cuda.is_available() and ("torch_cuda_rng_state_all" in state):
                try:
                    torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])
                except Exception:
                    pass
            print(f"[Resume] Loaded trainer state from: {state_path}")
        except Exception as e:
            print(f"[Resume] WARNING: failed to load trainer state: {e}")

    def _append_iter_log(self, record: Dict[str, object]) -> None:
        try:
            log_path = os.path.join(self.run_dir, "iter_log.jsonl")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[IterLog] WARNING: failed to append log: {e}")

    def _save_checkpoint(self, step: int) -> None:
        cfg = self.cfg
        run_dir = self.run_dir
        os.makedirs(run_dir, exist_ok=True)

        final_dir = os.path.join(run_dir, f"step_{step:05d}")
        tmp_dir = final_dir + ".tmp"
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        def save_core(core: VLMCore, subdir: str) -> None:
            sub = os.path.join(tmp_dir, subdir)
            os.makedirs(sub, exist_ok=True)

            try:
                if core.tokenizer is not None:
                    core.tokenizer.save_pretrained(sub)
            except Exception:
                pass
            try:
                if core.processor is not None:
                    core.processor.save_pretrained(sub)
            except Exception:
                pass

            try:
                if core.is_lora:
                    core.model.save_pretrained(sub, save_adapter=True)
                else:
                    core.model.save_pretrained(sub)
            except Exception:
                torch.save(core.model.state_dict(), os.path.join(sub, "pytorch_model.bin"))

            meta = {
                "model_name": core.model_name,
                "is_lora": core.is_lora,
                "adapter_names": getattr(core, "adapter_names", []),
                "device_map": cfg.device_map,
                "dtype": str(core.dtype),
                "step": step,
                "time": int(time.time()),
            }
            with open(os.path.join(sub, "checkpoint_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

        save_core(self.solver.core, "solver")
        if self.proposer.core is not self.solver.core:
            save_core(self.proposer.core, "proposer")
        else:
            with open(os.path.join(tmp_dir, "SHARED_BACKBONE.txt"), "w") as f:
                f.write("Adapters 'default'(solver) and 'proposer' are stored in this checkpoint.\n")

        trainer_state = {
            "step": step,
            "solver_opt": self.solver_updater.opt.state_dict(),
            "proposer_opt": self.proposer_updater.opt.state_dict(),
            "solver_kl_coef": float(self.solver_updater.kl_coef),
            "proposer_kl_coef": float(self.proposer_updater.kl_coef),
            "solver_baseline": float(self.solver_baseline),
            "proposer_baseline": float(self.proposer_baseline),
            "solver_updater_step": int(self.solver_updater._step),
            "proposer_updater_step": int(self.proposer_updater._step),
            "py_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                trainer_state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
            except Exception:
                pass
        torch.save(trainer_state, os.path.join(tmp_dir, "trainer_state.pt"))

        with open(os.path.join(tmp_dir, "SAVE_OK"), "w") as f:
            f.write("ok\n")

        try:
            os.replace(tmp_dir, final_dir)
        except Exception:
            print(f"[Checkpoint] WARNING: atomic rename failed; keeping {tmp_dir}")
            return

        print(f"[Checkpoint] Saved: {os.path.basename(final_dir)}")
        try:
            _retain_last_k_checkpoints(run_dir, k=self.cfg.max_checkpoints)
        except Exception as e:
            print(f"[Checkpoint] Retention skipped due to error: {e}")

    def update_baseline(self, which: str, reward: float) -> None:
        if which == "solver":
            self.solver_baseline = self.momentum * self.solver_baseline + (1 - self.momentum) * reward
        else:
            self.proposer_baseline = self.momentum * self.proposer_baseline + (1 - self.momentum) * reward

    def _step_reward_weight(self, step: int) -> float:
        cfg = self.cfg
        if cfg.total_steps <= cfg.start_step:
            return float(cfg.step_reward_w_end)

        total_span = max(1, cfg.total_steps - cfg.start_step)
        progress = (step - cfg.start_step) / float(total_span)
        progress = max(0.0, min(1.0, progress))

        warm = max(0.0, min(1.0, cfg.step_reward_warmup_frac))
        if progress <= warm:
            interp = 0.0
        else:
            interp = (progress - warm) / max(1e-6, (1.0 - warm))

        w = cfg.step_reward_w_start + (cfg.step_reward_w_end - cfg.step_reward_w_start) * interp
        return float(max(0.0, min(1.0, w)))

    def _wandb_log(self, step: int, metrics: Dict[str, object]) -> None:
        if self.wandb_run is None:
            return
        wandb.log(metrics, step=step)

    def train(self) -> None:
        cfg = self.cfg
        print(f"Starting training for {cfg.total_steps} steps.")

        for step in range(cfg.start_step + 1, cfg.total_steps + 1):
            image, meta = self.pool.sample_by_iter(step)
            image_path = meta.get("path")

            proposer_prompt = build_proposer_prompt(meta)
            proposer_out = self.proposer.generate(
                image=image,
                prompt=proposer_prompt,
                max_new_tokens=cfg.max_new_tokens_proposer,
                temperature=cfg.temp,
                top_p=cfg.top_p,
            )
            question = strip_tags(proposer_out, "question") or proposer_out.strip()
            question = question.replace("\n", " ").strip() or "What is the most prominent value shown in the image?"

            solver_prompt = build_solver_prompt(question)
            solver_answers_raw: List[str] = []
            solver_answers_norm: List[str] = []
            solver_completions: List[str] = []
            solver_steps: List[List[str]] = []
            pre_words_list: List[int] = []

            for _ in range(cfg.num_solver_samples):
                sol_out = self.solver.generate(
                    image=image,
                    prompt=solver_prompt,
                    max_new_tokens=cfg.max_new_tokens_solver,
                    temperature=cfg.temp,
                    top_p=cfg.top_p,
                )

                ans = strip_tags(sol_out, "answer")
                if ans is None:
                    lines = [ln.strip() for ln in sol_out.strip().splitlines() if ln.strip()]
                    ans = lines[-1] if lines else "unknown"

                ans_norm = normalize_answer(ans)

                think = strip_tags(sol_out, "think")
                if think is None:
                    idx = sol_out.lower().find("<answer>")
                    think = sol_out[:idx].strip() if idx != -1 else sol_out

                steps = extract_steps(think, max_steps=cfg.max_reasoning_steps)

                solver_answers_raw.append(ans)
                solver_answers_norm.append(ans_norm)
                solver_completions.append(sol_out)
                solver_steps.append(steps)
                pre_words_list.append(pre_answer_word_count(sol_out))

            dominant_answer, dom_count = majority_vote(solver_answers_norm)
            dom_frac = dom_count / float(cfg.num_solver_samples)

            hist: Dict[str, int] = {}
            for a in solver_answers_norm:
                hist[a] = hist.get(a, 0) + 1
            probs = [c / float(cfg.num_solver_samples) for c in hist.values()]
            entropy_nats = shannon_entropy_nats(probs)

            # Outcome-level intrinsic reward: prefer answers that are self-consistent (soft majority).
            prob_map = {ans: c / float(cfg.num_solver_samples) for ans, c in hist.items()}
            target_w = max(1, cfg.len_penalty_target_words)
            penalties = [min(1.0, max(0.0, (pw - target_w) / float(target_w))) for pw in pre_words_list]

            solver_reward_soft: List[float] = []
            for a_norm, pen in zip(solver_answers_norm, penalties):
                p = prob_map.get(a_norm, 0.0)
                length_factor = 1.0 - cfg.len_penalty_weight * pen
                solver_reward_soft.append((p ** cfg.solver_soft_gamma) * length_factor)

            # Reasoning-aware intrinsic reward: step agreement inside dominant answer group.
            solver_reward_step, step_info = compute_step_agreement_rewards(
                core=self.solver.core,
                answers_norm=solver_answers_norm,
                steps_per_sample=solver_steps,
                dominant_answer=dominant_answer,
                cfg=cfg,
            )

            w_step = self._step_reward_weight(step)
            solver_reward_final = [
                (1.0 - w_step) * r_soft + w_step * r_step
                for r_soft, r_step in zip(solver_reward_soft, solver_reward_step)
            ]
            step_info = step_info or {}
            step_info["mix_step_weight"] = float(w_step)

            # Proposer intrinsic reward based on answer distribution entropy (encourage non-trivial but solvable questions).
            proposer_reward = gaussian_reward(entropy_nats, cfg.prop_entropy_mu, cfg.prop_entropy_sigma)

            solver_stats_list: List[Dict[str, float]] = []
            for sol_out, r_final in zip(solver_completions, solver_reward_final):
                stats = self.solver_updater.step(
                    image=image,
                    prompt=solver_prompt,
                    completion=sol_out,
                    reward=r_final,
                    baseline=self.solver_baseline,
                )
                solver_stats_list.append(stats)
                self.update_baseline("solver", r_final)

            proposer_stats = None
            if (step % cfg.proposer_update_freq) == 0:
                proposer_stats = self.proposer_updater.step(
                    image=image,
                    prompt=proposer_prompt,
                    completion=proposer_out,
                    reward=proposer_reward,
                    baseline=self.proposer_baseline,
                )
                self.update_baseline("proposer", proposer_reward)

            pre_words_mean = sum(pre_words_list) / max(1, len(pre_words_list))
            print(
                f"[Step {step:04d}] dom={dom_count}/{cfg.num_solver_samples} "
                f"dom_frac={dom_frac:.2f} H={entropy_nats:.3f} "
                f"prop_r={proposer_reward:.2f} "
                f"solver_soft={sum(solver_reward_soft)/len(solver_reward_soft):.2f} "
                f"solver_step={sum(solver_reward_step)/len(solver_reward_step):.2f} "
                f"solver_final={sum(solver_reward_final)/len(solver_reward_final):.2f} "
                f"w_step={w_step:.2f} pre_words={pre_words_mean:.1f}"
            )
            print(f"Q: {question}")
            print(f"A: [{', '.join(solver_answers_raw)}] | DOM: {dominant_answer}")

            metrics: Dict[str, object] = {
                "train/step": step,
                "train/dom_count": dom_count,
                "train/dom_frac": dom_frac,
                "train/entropy_nats": entropy_nats,
                "train/pre_words_mean": pre_words_mean,
                "train/proposer_reward": proposer_reward,
                "train/solver_reward_soft_mean": sum(solver_reward_soft) / max(1, len(solver_reward_soft)),
                "train/solver_reward_step_mean": sum(solver_reward_step) / max(1, len(solver_reward_step)),
                "train/solver_reward_final_mean": sum(solver_reward_final) / max(1, len(solver_reward_final)),
                "train/mix_step_weight": w_step,
                "text/question": question,
                "text/dominant_answer": dominant_answer,
                "text/answers_raw": ", ".join(solver_answers_raw),
                "data/image_path": image_path,
                "text/answer_hist": json.dumps(hist, ensure_ascii=False),
            }
            for k, v in (step_info or {}).items():
                metrics[f"reasoning_step_agreement/{k}"] = v

            if solver_stats_list:
                metrics["solver/ce_loss_mean"] = sum(s["ce_loss"] for s in solver_stats_list) / len(solver_stats_list)
                metrics["solver/kl_loss_mean"] = sum(s["kl_loss"] for s in solver_stats_list) / len(solver_stats_list)
                metrics["solver/adv_mean"] = sum(s["advantage"] for s in solver_stats_list) / len(solver_stats_list)
                metrics["solver/kl_coef"] = solver_stats_list[-1]["kl_coef_after"]

            if proposer_stats:
                metrics["proposer/ce_loss"] = proposer_stats["ce_loss"]
                metrics["proposer/kl_loss"] = proposer_stats["kl_loss"]
                metrics["proposer/adv"] = proposer_stats["advantage"]
                metrics["proposer/kl_coef"] = proposer_stats["kl_coef_after"]

            if HAS_WANDB and self.wandb_run is not None and cfg.wandb_log_images_every > 0 and (step % cfg.wandb_log_images_every) == 0:
                try:
                    metrics["vis/image"] = wandb.Image(image, caption=f"Step {step}")
                except Exception:
                    pass

            self._wandb_log(step, metrics)

            self._append_iter_log(
                {
                    "step": step,
                    "image_path": image_path,
                    "question": question,
                    "proposer_out": proposer_out,
                    "solver_answers_raw": solver_answers_raw,
                    "solver_answers_norm": solver_answers_norm,
                    "dominant_answer": dominant_answer,
                    "dominant_count": dom_count,
                    "entropy_nats": entropy_nats,
                    "solver_reward_soft": solver_reward_soft,
                    "solver_reward_step": solver_reward_step,
                    "solver_reward_final": solver_reward_final,
                    "proposer_reward": proposer_reward,
                    "pre_answer_word_counts": pre_words_list,
                    "answer_hist": hist,
                    "step_agreement_info": step_info,
                }
            )

            if torch.cuda.is_available() and cfg.clear_cache_every > 0 and (step % cfg.clear_cache_every == 0):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                gc.collect()

            if cfg.save_every and (step % cfg.save_every == 0):
                self._save_checkpoint(step)

        if cfg.save_every and (cfg.total_steps % cfg.save_every != 0):
            self._save_checkpoint(cfg.total_steps)

        if self.wandb_run is not None:
            try:
                wandb.finish()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("iReasoner — unsupervised Proposer–Solver self-improvement with reasoning-step agreement")

    p.add_argument("--data_dir", type=str, default=os.environ.get("DATA_DIR", "images/train"))
    p.add_argument(
        "--include_subfolders",
        type=str,
        default=os.environ.get("INCLUDE_SUBFOLDERS", None),
        help="Comma-separated FIRST-LEVEL subfolders under --data_dir. Default: all.",
    )

    p.add_argument("--solver_model", type=str, default=os.environ.get("SOLVER_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"))
    p.add_argument("--proposer_model", type=str, default=os.environ.get("PROPOSER_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"))

    p.add_argument("--device", type=str, default=os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--dtype", type=str, default=os.environ.get("DTYPE", "bfloat16"), choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device_map", type=str, default=os.environ.get("DEVICE_MAP", "auto"))

    p.add_argument("--total_steps", type=int, default=int(os.environ.get("TOTAL_STEPS", "100")))
    p.add_argument("--proposer_update_freq", type=int, default=int(os.environ.get("PROP_FREQ", "5")))
    p.add_argument("--num_solver_samples", type=int, default=int(os.environ.get("N_SAMPLES", "5")))
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-6")))
    p.add_argument("--weight_decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", "0.01")))
    p.add_argument("--grad_clip", type=float, default=float(os.environ.get("GRAD_CLIP", "1.0")))
    p.add_argument("--temp", type=float, default=float(os.environ.get("TEMP", "1.0")))
    p.add_argument("--top_p", type=float, default=float(os.environ.get("TOP_P", "1.0")))
    p.add_argument("--max_new_tokens_solver", type=int, default=int(os.environ.get("MAX_NEW_TOKENS_SOLVER", "256")))
    p.add_argument("--max_new_tokens_proposer", type=int, default=int(os.environ.get("MAX_NEW_TOKENS_PROPOSER", "128")))

    p.add_argument("--freeze_vision", action="store_true", default=(os.environ.get("FREEZE_VISION", "1") != "0"))
    p.add_argument("--no-freeze-vision", dest="freeze_vision", action="store_false")

    p.add_argument("--len_penalty_weight", type=float, default=float(os.environ.get("LEN_PENALTY_WEIGHT", "0.05")))
    p.add_argument("--len_penalty_target_words", type=int, default=int(os.environ.get("LEN_PENALTY_TARGET_WORDS", "64")))
    p.add_argument("--solver_soft_gamma", type=float, default=float(os.environ.get("SOLVER_SOFT_GAMMA", "0.7")))

    p.add_argument("--prop_entropy_mu", type=float, default=float(os.environ.get("PROP_ENTROPY_MU", "0.90")))
    p.add_argument("--prop_entropy_sigma", type=float, default=float(os.environ.get("PROP_ENTROPY_SIGMA", "0.35")))

    p.add_argument("--max_reasoning_steps", type=int, default=int(os.environ.get("MAX_REASONING_STEPS", "8")))
    p.add_argument("--step_embed_max_tokens", type=int, default=int(os.environ.get("STEP_EMBED_MAX_TOKENS", "64")))
    p.add_argument("--min_step_group_size", type=int, default=int(os.environ.get("MIN_STEP_GROUP_SIZE", "2")))
    p.add_argument("--step_group_density_gamma", type=float, default=float(os.environ.get("STEP_GROUP_DENSITY_GAMMA", "0.7")))
    p.add_argument("--step_position_decay", type=float, default=float(os.environ.get("STEP_POSITION_DECAY", "0.15")))
    p.add_argument("--step_sim_temperature", type=float, default=float(os.environ.get("STEP_SIM_TEMPERATURE", "1.0")))

    p.add_argument("--step_reward_w_start", type=float, default=float(os.environ.get("STEP_REWARD_W_START", "0.0")))
    p.add_argument("--step_reward_w_end", type=float, default=float(os.environ.get("STEP_REWARD_W_END", "0.7")))
    p.add_argument("--step_reward_warmup_frac", type=float, default=float(os.environ.get("STEP_REWARD_WARMUP_FRAC", "0.3")))

    p.add_argument("--kl_target", type=float, default=float(os.environ.get("KL_TARGET", "0.02")))
    p.add_argument("--kl_adapt_rate", type=float, default=float(os.environ.get("KL_ADAPT_RATE", "0.10")))
    p.add_argument("--kl_coef", type=float, default=float(os.environ.get("KL_COEF", "1e-3")))

    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", random.randint(0, 2**32 - 1))))
    p.add_argument("--output_dir", type=str, default=os.environ.get("OUTPUT_DIR", "runs"))
    p.add_argument("--save_every", type=int, default=int(os.environ.get("SAVE_EVERY", "50")))
    p.add_argument("--max_checkpoints", type=int, default=int(os.environ.get("MAX_CKPTS", "2")))

    p.add_argument("--use_lora_solver", action="store_true", default=(os.environ.get("USE_LORA_SOLVER", "0") == "1"))
    p.add_argument("--use_lora_proposer", action="store_true", default=(os.environ.get("USE_LORA_PROPOSER", "0") == "1"))
    p.add_argument("--lora_r", type=int, default=int(os.environ.get("LORA_R", "16")))
    p.add_argument("--lora_alpha", type=int, default=int(os.environ.get("LORA_ALPHA", "32")))
    p.add_argument("--lora_dropout", type=float, default=float(os.environ.get("LORA_DROPOUT", "0.05")))
    p.add_argument(
        "--lora_targets",
        type=str,
        default=os.environ.get("LORA_TARGETS", ",".join(DEFAULT_LORA_TARGETS)),
        help="Comma-separated LoRA target module names (include mm_projector for VLM adapters).",
    )

    p.add_argument("--wandb_mode", type=str, default=os.environ.get("WANDB_MODE", "disabled"), choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "ireasoner"))
    p.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY", None))
    p.add_argument("--wandb_run_name", type=str, default=os.environ.get("WANDB_RUN_NAME", None))
    p.add_argument("--wandb_log_images_every", type=int, default=int(os.environ.get("WANDB_LOG_IMAGES_EVERY", "0")))

    p.add_argument("--clear_cache_every", type=int, default=int(os.environ.get("CLEAR_CACHE_EVERY", "25")))
    return p.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Config:
    lora_targets = tuple([s.strip() for s in (args.lora_targets or "").split(",") if s.strip()]) or DEFAULT_LORA_TARGETS

    include_subfolders = None
    if args.include_subfolders:
        parsed = [s.strip() for s in args.include_subfolders.split(",") if s.strip()]
        include_subfolders = tuple(parsed) if parsed else None

    return Config(
        solver_model_name=args.solver_model,
        proposer_model_name=args.proposer_model,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        total_steps=args.total_steps,
        proposer_update_freq=args.proposer_update_freq,
        num_solver_samples=args.num_solver_samples,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        temp=args.temp,
        top_p=args.top_p,
        max_new_tokens_solver=args.max_new_tokens_solver,
        max_new_tokens_proposer=args.max_new_tokens_proposer,
        data_dir=args.data_dir,
        freeze_vision=args.freeze_vision,
        seed=args.seed,
        output_dir=args.output_dir,
        save_every=args.save_every,
        max_checkpoints=max(2, int(args.max_checkpoints)),
        include_subfolders=include_subfolders,
        len_penalty_weight=args.len_penalty_weight,
        len_penalty_target_words=args.len_penalty_target_words,
        solver_soft_gamma=args.solver_soft_gamma,
        prop_entropy_mu=args.prop_entropy_mu,
        prop_entropy_sigma=args.prop_entropy_sigma,
        max_reasoning_steps=args.max_reasoning_steps,
        step_embed_max_tokens=args.step_embed_max_tokens,
        min_step_group_size=args.min_step_group_size,
        step_group_density_gamma=args.step_group_density_gamma,
        step_position_decay=args.step_position_decay,
        step_sim_temperature=args.step_sim_temperature,
        step_reward_w_start=args.step_reward_w_start,
        step_reward_w_end=args.step_reward_w_end,
        step_reward_warmup_frac=args.step_reward_warmup_frac,
        kl_target=args.kl_target,
        kl_adapt_rate=args.kl_adapt_rate,
        kl_coef=args.kl_coef,
        use_lora_solver=args.use_lora_solver,
        use_lora_proposer=args.use_lora_proposer,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_targets,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_log_images_every=args.wandb_log_images_every,
        clear_cache_every=args.clear_cache_every,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)
    cfg = maybe_autoresume(cfg)
    print(cfg)
    print("=" * 60)
    trainer = IReasonerTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
