from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch

from config import Config
from vlm import VLMCore


def compute_step_agreement_rewards(
    core: VLMCore,
    answers_norm: List[str],
    steps_per_sample: List[List[str]],
    dominant_answer: str,
    cfg: Config,
) -> Tuple[List[float], Dict[str, float]]:
    """
    Trajectory-aware intrinsic reward:
    within the dominant (majority) answer group, reward samples whose intermediate
    reasoning steps are mutually consistent (step embedding agreement).
    """
    n = len(answers_norm)
    rewards = [0.0 for _ in range(n)]
    info: Dict[str, float] = {
        "dom_group_size": 0.0,
        "num_step_positions_used": 0.0,
        "mean_step_cosine": 0.0,
        "group_density": 0.0,
    }
    if n == 0:
        return rewards, info

    dom_indices = [i for i, a in enumerate(answers_norm) if a == dominant_answer]
    dom_size = len(dom_indices)
    info["dom_group_size"] = float(dom_size)
    if dom_size < cfg.min_step_group_size:
        return rewards, info

    group_density = dom_size / float(n)
    info["group_density"] = float(group_density)
    density_weight = group_density ** cfg.step_group_density_gamma

    step_embeds_by_j: Dict[int, List[Tuple[int, torch.Tensor]]] = {}
    for idx in dom_indices:
        steps = steps_per_sample[idx] if idx < len(steps_per_sample) else []
        for j, s_text in enumerate(steps):
            if j >= cfg.max_reasoning_steps:
                break
            if not s_text.strip():
                continue
            emb = core.embed_text(s_text)
            step_embeds_by_j.setdefault(j, []).append((idx, emb))

    score_sum = [0.0 for _ in range(n)]
    weight_sum = [0.0 for _ in range(n)]
    all_sims: List[float] = []
    used_positions = 0

    for j, pairs in step_embeds_by_j.items():
        if len(pairs) < cfg.min_step_group_size:
            continue

        vecs = torch.stack([e for (_, e) in pairs], dim=0)
        center = vecs.mean(dim=0)
        center = center / (center.norm(p=2) + 1e-8)

        w_j = math.exp(-cfg.step_position_decay * float(j))

        for idx, e_vec in pairs:
            v = e_vec / (e_vec.norm(p=2) + 1e-8)
            sim = float(torch.dot(v, center).item())
            sim = max(-1.0, min(1.0, sim))
            all_sims.append(sim)

            if cfg.step_sim_temperature != 1.0:
                sim = math.tanh(sim / max(cfg.step_sim_temperature, 1e-3))

            sim01 = (sim + 1.0) * 0.5
            score_sum[idx] += sim01 * w_j
            weight_sum[idx] += w_j

        used_positions += 1

    info["num_step_positions_used"] = float(used_positions)
    if used_positions == 0 or not all_sims:
        return rewards, info

    info["mean_step_cosine"] = float(sum(all_sims) / len(all_sims))

    for i in range(n):
        if weight_sum[i] > 0.0:
            rewards[i] = (score_sum[i] / weight_sum[i]) * density_weight

    return rewards, info
