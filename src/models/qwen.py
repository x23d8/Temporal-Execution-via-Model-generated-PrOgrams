"""Qwen3.5-9B wrapper dùng HuggingFace transformers.

Qwen3.5 hỗ trợ 2 chế độ qua `apply_chat_template(..., enable_thinking=...)`:
- non-thinking (default eval): phản hồi ngắn, đúng format.
- thinking: phát sinh `<think>...</think>` trước đáp án; extractor sẽ strip.

Model được load 1 lần, có thể reuse qua nhiều method/dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import ChatMessage

DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-9B"


@dataclass
class QwenConfig:
    model_name: str = DEFAULT_MODEL_NAME
    dtype: str = "bfloat16"  # "float16" | "bfloat16" | "float32"
    device_map: str | None = "auto"
    trust_remote_code: bool = True
    use_flash_attention: bool = True   # Flash Attention 2 — A100/H100 only
    load_in_4bit: bool = False         # 4-bit NF4 quant — 9B → ~4.5 GB, fits any GPU
    load_kwargs: dict[str, Any] = field(default_factory=dict)


class QwenChatLM:
    def __init__(self, config: QwenConfig | None = None):
        self.config = config or QwenConfig()
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        import torch  # lazy import
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )

        extra: dict[str, Any] = {}

        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            extra["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            # 4-bit models must use device_map; flash_attn not compatible with bnb
        elif self.config.use_flash_attention:
            try:
                import flash_attn  # noqa: F401
                extra["attn_implementation"] = "flash_attention_2"
            except ImportError:
                pass

        # Pre-load CUDA diagnostic
        import torch as _torch
        if not _torch.cuda.is_available():
            print("[QwenChatLM] ⚠️  CUDA not available before model load — check GPU runtime!")
        else:
            free, total = _torch.cuda.mem_get_info(0)
            print(
                f"[QwenChatLM] GPU detected: {_torch.cuda.get_device_name(0)} | "
                f"Free VRAM: {free/1e9:.1f} GB / {total/1e9:.1f} GB"
            )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=torch_dtype,          # `torch_dtype` is deprecated → use `dtype`
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
            **extra,
            **self.config.load_kwargs,
        )
        self._model.eval()

        # Diagnostic: show where each layer ended up so CPU offload is visible
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            quant = "4-bit" if self.config.load_in_4bit else self.config.dtype
            print(
                f"[QwenChatLM] loaded {self.config.model_name} ({quant}) | "
                f"GPU: {allocated:.1f} GB allocated / {reserved:.1f} GB reserved / "
                f"{total:.1f} GB total"
            )
            if hasattr(self._model, "hf_device_map"):
                unique = set(str(d) for d in self._model.hf_device_map.values())
                if "cpu" in unique:
                    print(
                        "[QwenChatLM] ⚠️  Some layers on CPU! "
                        "Enable load_in_4bit=True or use a smaller model."
                    )
        else:
            print("[QwenChatLM] ⚠️  CUDA not available — running on CPU (very slow).")

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        if self._model is None or self._tokenizer is None:
            self.load()
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        chat = [{"role": m.role, "content": m.content} for m in messages]
        prompt_text = self._tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(device)

        # Stop tại <|im_end|> (end-of-turn) lẫn <|endoftext|> để tránh garbage tokens
        eos_ids = [self._tokenizer.eos_token_id]
        im_end_id = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id not in (self._tokenizer.unk_token_id, self._tokenizer.eos_token_id):
            eos_ids.append(im_end_id)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": eos_ids,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text

    def generate_batch(
        self,
        messages_list: list[list[ChatMessage]],
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool = False,
        enable_thinking: bool = False,
    ) -> list[str]:
        """Batch generation — left-pads inputs, single CUDA kernel launch per batch.

        All messages_list entries are processed together. Caller is responsible
        for grouping by compatible max_new_tokens if needed.
        """
        if self._model is None or self._tokenizer is None:
            self.load()
        import torch

        prompts = [
            self._tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in msgs],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for msgs in messages_list
        ]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer.padding_side = "left"
        inputs = self._tokenizer(
            prompts, return_tensors="pt", padding=True
        ).to(device)

        eos_ids = [self._tokenizer.eos_token_id]
        im_end_id = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id not in (self._tokenizer.unk_token_id, self._tokenizer.eos_token_id):
            eos_ids.append(im_end_id)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": eos_ids,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        results = []
        for out in output_ids:
            new_tokens = out[prompt_len:]
            results.append(self._tokenizer.decode(new_tokens, skip_special_tokens=True))
        return results
