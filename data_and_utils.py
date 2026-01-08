from __future__ import annotations

import gc
import math
import os
import random
import re
import shutil
import string
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from config import Config


def clip_grad_norm_multi_device(model: nn.Module, max_norm: float) -> None:
    by_dev: Dict[torch.device, List[nn.Parameter]] = {}
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            by_dev.setdefault(p.grad.device, []).append(p)
    for _dev, params in by_dev.items():
        nn.utils.clip_grad_norm_(params, max_norm)


def safe_dtype(dtype: str) -> torch.dtype:
    if dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if dtype == "float16" and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def strip_tags(text: str, tag: str) -> Optional[str]:
    lt, rt = f"<{tag}>", f"</{tag}>"
    if lt in text and rt in text:
        s = text.split(lt, 1)[1]
        s = s.split(rt, 1)[0]
        return s.strip()
    return None


def normalize_answer(ans: str) -> str:
    s = ans.strip().lower()
    s = s.replace(",", "")
    s = s.replace("\n", " ").replace("\t", " ")
    s = " ".join(s.split())
    return s.strip(string.punctuation + " ")


def majority_vote(answers: List[str]) -> Tuple[str, int]:
    counts: Dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    maj = max(counts.items(), key=lambda kv: kv[1])
    return maj[0], maj[1]


def shannon_entropy_nats(probs: List[float]) -> float:
    eps = 1e-12
    return -sum(p * math.log(max(p, eps)) for p in probs if p > 0.0)


def pre_answer_word_count(text: str) -> int:
    idx = text.lower().find("<answer>")
    pre = text if idx == -1 else text[:idx]
    return len(pre.strip().split())


def gaussian_reward(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    return math.exp(-((x - mu) ** 2) / (2.0 * sigma * sigma))


STEP_SPLIT_RE = re.compile(
    r"^\s*(?:step\s*(\d+)|(\d+)[\.\)])\s*[:\-]?\s*(.*)$",
    re.IGNORECASE,
)


def extract_steps(think_text: str, max_steps: int = 8) -> List[str]:
    if not think_text:
        return []

    steps: List[str] = []

    tag_matches = re.findall(r"<step>(.*?)</step>", think_text, flags=re.IGNORECASE | re.DOTALL)
    if tag_matches:
        for chunk in tag_matches:
            s = chunk.strip()
            if s:
                steps.append(s)
                if len(steps) >= max_steps:
                    break
        if steps:
            return steps

    lines = [ln.strip() for ln in think_text.splitlines() if ln.strip()]
    for ln in lines:
        m = STEP_SPLIT_RE.match(ln)
        if m:
            content = m.group(3) if m.group(3) else ln
            if content.strip():
                steps.append(content.strip())
        else:
            if steps:
                steps[-1] = (steps[-1] + " " + ln).strip()
            else:
                steps.append(ln)
        if len(steps) >= max_steps:
            break

    if not steps:
        sentences = re.split(r"(?<=[\.!?])\s+", think_text.strip())
        for s in sentences:
            s = s.strip()
            if s:
                steps.append(s)
                if len(steps) >= max_steps:
                    break

    return steps


class ImagePool:
    DEFAULT_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.paths: List[str] = []

        root = os.path.abspath(cfg.data_dir)
        if not os.path.isdir(root):
            raise RuntimeError(f"[ImagePool] data_dir not found: {root}")

        if cfg.include_subfolders:
            chosen = []
            for name in cfg.include_subfolders:
                sub = os.path.join(root, name)
                if os.path.isdir(sub):
                    chosen.append((name, sub))
                else:
                    print(f"[ImagePool] WARNING: requested subfolder not found: {name}")
        else:
            chosen = []
            for name in sorted(os.listdir(root)):
                sub = os.path.join(root, name)
                if os.path.isdir(sub) and not name.startswith("."):
                    chosen.append((name, sub))

        if not chosen:
            print(f"[ImagePool] NOTE: no subfolders found; scanning images under {root}.")
            chosen = [("", root)]

        def is_img(fn: str) -> bool:
            fnl = fn.lower()
            return fnl.endswith(self.DEFAULT_EXTS) and not os.path.basename(fnl).startswith(".")

        for _sub_name, sub_path in chosen:
            for r, _dirs, files in os.walk(sub_path):
                for fn in files:
                    if is_img(fn):
                        self.paths.append(os.path.join(r, fn))

        if not self.paths:
            raise RuntimeError(f"[ImagePool] No images found under: {root}")

        self.paths.sort()
        print(f"[ImagePool] Found {len(self.paths)} images under: {root}")

        self.indices = list(range(len(self.paths)))
        rnd = random.Random(cfg.seed)
        rnd.shuffle(self.indices)
        self._root = root

    def __len__(self) -> int:
        return len(self.paths)

    def _build_meta(self, p: str) -> dict:
        rel = os.path.relpath(p, self._root)
        parts = rel.split(os.sep)
        subfolder = parts[0] if len(parts) > 1 else ""
        return {
            "dataset": "folder",
            "split": "train",
            "path": p,
            "rel_path": rel,
            "subfolder": subfolder,
        }

    def sample_by_iter(self, iter_no: int) -> Tuple[Image.Image, dict]:
        idx = self.indices[(max(1, int(iter_no)) - 1) % len(self.paths)]
        p = self.paths[idx]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            return self.sample_by_iter(iter_no + 1)
        return img, self._build_meta(p)


def build_proposer_prompt(_meta: dict) -> str:
    return """
You are a Question Proposer.
Given the IMAGE, generate one short-answer question that can be answered from the image alone.
Avoid ambiguity and avoid overly trivial counting questions.

Rules:
- Output exactly in XML with two tags:
  <question> ... </question>
  <rationale>Briefly explain why this question is non-trivial but solvable.</rationale>
- Do NOT include the answer.
- Keep the question 1–2 sentences, clear and specific.
- If numeric, make sure units/context are clear.

Only output the two XML tags, nothing else.
""".strip()


def build_solver_prompt(question_text: str) -> str:
    return f"""
You are a precise Vision-Language Reasoner.
Task: Answer the user's question using ONLY the provided IMAGE.

Instructions:
- Produce intermediate reasoning steps inside <think>...</think> as numbered lines:
    <think>
    Step 1: ...
    Step 2: ...
    Step 3: ...
    </think>
- Each step should state one concrete visual/structural claim that supports the solution.
- Then output a short final answer in <answer>...</answer>.
- Do NOT add any text outside <think> and <answer>.

Question: {question_text}
""".strip()


@contextmanager
def maybe_cuda_cache_cleanup(enabled: bool):
    try:
        yield
    finally:
        if enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            gc.collect()
