"""HuggingFace transformers wrapper — drop-in for OllamaChatLM on Kaggle/Colab.

Implements the same ChatLM protocol so it can be passed directly to run().
Supports 4-bit quantisation (BitsAndBytes) for T4 / P100 budgets.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Optional

from .base import ChatMessage


@dataclass
class HFConfig:
    model_name: str = "google/gemma-2-2b-it"
    device_map: str = "auto"
    torch_dtype: str = "float16"        # "float16" | "bfloat16" | "float32"
    load_in_4bit: bool = False           # BitsAndBytes 4-bit — saves ~4× VRAM
    load_in_8bit: bool = False           # BitsAndBytes 8-bit — saves ~2× VRAM
    trust_remote_code: bool = False
    max_new_tokens: int = 256
    temperature: float = 0.0
    # Chat-template fallback when the tokenizer has no built-in template
    fallback_template: str = (
        "{% for m in messages %}"
        "{% if m['role'] == 'system' %}<start_of_turn>user\n{{ m['content'] }}<end_of_turn>\n"
        "{% elif m['role'] == 'user' %}<start_of_turn>user\n{{ m['content'] }}<end_of_turn>\n"
        "{% elif m['role'] == 'assistant' %}<start_of_turn>model\n{{ m['content'] }}<end_of_turn>\n"
        "{% endif %}{% endfor %}<start_of_turn>model\n"
    )


class HFChatLM:
    def __init__(self, config: HFConfig | None = None) -> None:
        self.config = config or HFConfig()
        self._model = None
        self._tokenizer = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        cfg = self.config
        print(f"[HFChatLM] loading {cfg.model_name}  (4bit={cfg.load_in_4bit}, 8bit={cfg.load_in_8bit})")

        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            trust_remote_code=cfg.trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtype_map.get(cfg.torch_dtype, torch.float16)

        model_kwargs: dict = dict(
            device_map=cfg.device_map,
            trust_remote_code=cfg.trust_remote_code,
        )

        if cfg.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )
        elif cfg.load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        self._model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
        self._model.eval()
        print(f"[HFChatLM] ready — {cfg.model_name}")

    def unload(self) -> None:
        """Free GPU memory before loading the next model."""
        import torch
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[HFChatLM] model unloaded")

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        return self.generate_batch(
            [messages],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            enable_thinking=enable_thinking,
        )[0]

    def generate_batch(
        self,
        messages_list: list[list[ChatMessage]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
        enable_thinking: bool = False,
    ) -> list[str]:
        import torch

        tok  = self._tokenizer
        mdl  = self._model
        cfg  = self.config

        # Apply chat template to each conversation
        prompts: list[str] = []
        for messages in messages_list:
            chat = [{"role": m.role, "content": m.content} for m in messages]
            try:
                try:
                    # Models like Qwen3 accept enable_thinking in apply_chat_template
                    text = tok.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                except TypeError:
                    # Tokenizer doesn't support enable_thinking (Gemma, Mistral, etc.)
                    text = tok.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=True
                    )
            except Exception:
                # Tokenizer has no built-in template — use our fallback
                from jinja2 import Environment
                env = Environment()
                tmpl = env.from_string(cfg.fallback_template)
                text = tmpl.render(messages=chat)
            prompts.append(text)

        device = next(mdl.parameters()).device
        inputs = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        gen_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        if do_sample and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = mdl.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens (strip the prompt)
        input_len = inputs["input_ids"].shape[1]
        results: list[str] = []
        for ids in output_ids:
            new_ids = ids[input_len:]
            text = tok.decode(new_ids, skip_special_tokens=True).strip()
            results.append(text)

        return results
