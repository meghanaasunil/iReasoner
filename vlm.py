from __future__ import annotations

import math
from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer, PreTrainedModel

from config import Config
from data_and_utils import safe_dtype

try:
    from peft import LoraConfig, TaskType, get_peft_model
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False


class VLMCore:
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype: str,
        cfg: Config,
        *,
        apply_lora: bool = False,
        make_adapters: Optional[List[str]] = None,
    ):
        self.device = device
        self.dtype = safe_dtype(dtype)
        self.model_name = model_name
        self.cfg = cfg

        print(f"[Load] {model_name} on {device} ({self.dtype}), device_map={cfg.device_map}")
        self.model: PreTrainedModel = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=cfg.device_map,
        )

        self.processor = AutoProcessor.from_pretrained(model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception:
            self.tokenizer = getattr(self.processor, "tokenizer", None)

        if self.cfg.freeze_vision:
            for n, p in self.model.named_parameters():
                if "vision" in n.lower():
                    p.requires_grad_(False)

        self.is_lora = False
        self.adapter_names: List[str] = []

        if apply_lora:
            self._init_lora(make_adapters=make_adapters)

        self.primary_device = self._infer_primary_device()
        self.device = str(self.primary_device)

        dm = cfg.device_map
        if (dm is None) or (isinstance(dm, str) and dm.lower() == "cpu"):
            self.model.to(self.primary_device)

        self.model.eval()

    def _infer_primary_device(self) -> torch.device:
        dm = getattr(self.model, "hf_device_map", None)
        if isinstance(dm, dict):
            cuda_devs = [d for d in dm.values() if isinstance(d, str) and d.startswith("cuda")]
            if cuda_devs:
                try:
                    idx = min(int(d.split(":")[1]) for d in cuda_devs)
                    return torch.device(f"cuda:{idx}")
                except Exception:
                    pass
        try:
            return torch.device(self.cfg.device)
        except Exception:
            return torch.device("cpu")

    def _init_lora(self, make_adapters: Optional[List[str]]) -> None:
        if not HAS_PEFT:
            print("[LoRA] peft not installed; continuing without LoRA.")
            return

        targets = list(getattr(self.cfg, "lora_target_modules", ()))
        if not targets:
            return

        def apply_or_load(adapter_name: str, load_path: Optional[str]) -> bool:
            if load_path:
                from peft import PeftModel

                try:
                    self.model = PeftModel.from_pretrained(self.model, load_path)
                    if hasattr(self.model, "active_adapter") and self.model.active_adapter != adapter_name:
                        try:
                            self.model.load_adapter(load_path, adapter_name=adapter_name)
                        except Exception:
                            pass
                except Exception:
                    lcfg = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=self.cfg.lora_r,
                        lora_alpha=self.cfg.lora_alpha,
                        lora_dropout=self.cfg.lora_dropout,
                        target_modules=targets,
                    )
                    self.model = get_peft_model(self.model, lcfg)
                    self.model.load_adapter(load_path, adapter_name=adapter_name)

                self.adapter_names.append(adapter_name)
                return True

            lcfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=targets,
            )

            if not hasattr(self.model, "peft_config"):
                self.model = get_peft_model(self.model, lcfg)
                self.adapter_names.append(adapter_name)
                return True

            try:
                self.model.add_adapter(adapter_name, lcfg)
                self.adapter_names.append(adapter_name)
                return True
            except Exception:
                return False

        loaded_any = False
        loaded_any |= apply_or_load("default", getattr(self.cfg, "load_solver_adapter", None))

        if make_adapters:
            for name in make_adapters:
                if name == "proposer":
                    loaded_any |= apply_or_load("proposer", getattr(self.cfg, "load_proposer_adapter", None))
                elif name != "default":
                    loaded_any |= apply_or_load(name, None)

        if loaded_any:
            for n, p in self.model.named_parameters():
                if "lora_" in n.lower():
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)
            self.is_lora = True
            try:
                self.model.print_trainable_parameters()
            except Exception:
                pass

    def _set_active_adapter(self, name: Optional[str]) -> None:
        if not HAS_PEFT:
            return
        if hasattr(self.model, "set_adapter") and name is not None:
            try:
                self.model.set_adapter(name)
            except Exception:
                pass

    def _disable_adapters(self) -> None:
        if not HAS_PEFT:
            return
        if hasattr(self.model, "disable_adapter"):
            try:
                self.model.disable_adapter()
            except Exception:
                pass

    @contextmanager
    def use_adapter(self, name: Optional[str]):
        prev = None
        if HAS_PEFT and hasattr(self.model, "active_adapter"):
            prev = getattr(self.model, "active_adapter", None)
        if name is None:
            self._disable_adapters()
        else:
            self._set_active_adapter(name)
        try:
            yield
        finally:
            if prev is None:
                self._disable_adapters()
            else:
                self._set_active_adapter(prev)

    def _render_chat(self, image: Image.Image, prompt: str, add_generation_prompt: bool) -> str:
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
        }]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    @torch.no_grad()
    def generate(
        self,
        adapter: Optional[str],
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        chat_text = self._render_chat(image, prompt, add_generation_prompt=True)
        inputs = self.processor(text=chat_text, images=[image], return_tensors="pt").to(self.primary_device)
        with self.use_adapter(adapter):
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=(self.tokenizer.eos_token_id if self.tokenizer is not None else None),
            )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        text = (self.tokenizer or self.processor.tokenizer).decode(gen_ids, skip_special_tokens=True)
        return text.strip()

    def logprob_of_completion(self, adapter: Optional[str], image: Image.Image, prompt: str, completion: str) -> float:
        self.model.train(False)
        prompt_chat = self._render_chat(image, prompt, add_generation_prompt=True)
        full_text = prompt_chat + completion

        with torch.no_grad():
            prompt_inputs = self.processor(text=prompt_chat, images=[image], return_tensors="pt").to(self.primary_device)

        inputs = self.processor(text=full_text, images=[image], return_tensors="pt").to(self.primary_device)
        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask")
        labels = input_ids.clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:, :prompt_len] = -100

        with self.use_adapter(adapter), torch.enable_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attn, labels=labels)
            nll = outputs.loss.item()
        return -nll

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor:
        if self.tokenizer is None:
            raise RuntimeError("[embed_text] tokenizer not available for this model.")

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=getattr(self.cfg, "step_embed_max_tokens", 64),
        )
        emb_layer = self.model.get_input_embeddings()
        dev = emb_layer.weight.device
        input_ids = enc["input_ids"].to(dev)
        tok_emb = emb_layer(input_ids)
        vec = tok_emb.mean(dim=1).squeeze(0).float()
        vec = vec / (vec.norm(p=2) + 1e-8)
        return vec.cpu()


class VLMRole:
    def __init__(self, core: VLMCore, adapter_name: Optional[str]):
        self.core = core
        self.adapter_name = adapter_name

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        return self.core.generate(
            self.adapter_name,
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def logprob_of_completion(self, image: Image.Image, prompt: str, completion: str) -> float:
        return self.core.logprob_of_completion(self.adapter_name, image=image, prompt=prompt, completion=completion)
