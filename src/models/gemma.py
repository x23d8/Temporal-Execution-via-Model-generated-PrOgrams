"""Gemma-4-E4B-it wrapper — mirrors QwenChatLM / HFChatLM interface.

Gemma does not support `enable_thinking`.

System-prompt handling: probed at load time via the tokenizer's chat template.
- Older Gemma (≤3) remaps system→user, producing two consecutive user turns.
  In that case we merge system content into the first user message manually.
- Gemma-4 / 3n supports system role natively → pass it through unchanged.
This matches how Ollama handles the system role for the same model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import ChatMessage

DEFAULT_MODEL_NAME = "google/gemma-4-E4B-it"


def _merge_system_into_first_user(chat: list[dict]) -> list[dict]:
    """Prepend system content to first user message."""
    if not chat or chat[0]["role"] != "system":
        return chat
    sys_content = chat[0]["content"]
    merged: list[dict] = []
    prepended = False
    for msg in chat[1:]:
        if msg["role"] == "user" and not prepended:
            merged.append({"role": "user", "content": f"{sys_content}\n\n{msg['content']}"})
            prepended = True
        else:
            merged.append(msg)
    return merged


def _needs_system_merge(tokenizer) -> bool:
    """Return True if the tokenizer remaps system→user (requires manual merge).

    Probes by rendering [system, user] and checking whether two user turns
    appear in the output — the telltale sign of a system→user remap.
    Mirrors hf.py's _normalize_chat_for_template logic.
    """
    try:
        probe = tokenizer.apply_chat_template(
            [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
            tokenize=False,
            add_generation_prompt=False,
        )
        return probe.count("<start_of_turn>user") >= 2 or probe.count("<|im_start|>user") >= 2
    except Exception:
        return True  # safe default: merge


@dataclass
class GemmaConfig:
    model_name: str = DEFAULT_MODEL_NAME
    dtype: str = "bfloat16"           # "float16" | "bfloat16" | "float32"
    device_map: str | None = "auto"
    load_in_4bit: bool = False         # QLoRA mode via BitsAndBytes
    load_kwargs: dict[str, Any] = field(default_factory=dict)
    adapter_path: str | None = None    # PEFT/LoRA adapter directory


class GemmaChatLM:
    """Gemma-4-E4B-it inference wrapper.

    Matches the ChatLM protocol so it can replace QwenChatLM/HFChatLM in
    any method or runner.  Supports batched generation via generate_batch().
    """

    def __init__(self, config: GemmaConfig | None = None) -> None:
        self.config = config or GemmaConfig()
        self._model = None
        self._tokenizer = None
        self._merge_needed: bool = True  # conservative default; updated in load()

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self.config
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)

        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        # Base model tokenizers ship without chat_template; borrow from -it variant.
        if not getattr(self._tokenizer, "chat_template", None):
            it_name = cfg.model_name if cfg.model_name.endswith("-it") else cfg.model_name + "-it"
            print(f"[GemmaChatLM] no chat_template — borrowing from {it_name}")
            _it_tok = AutoTokenizer.from_pretrained(it_name)
            self._tokenizer.chat_template = _it_tok.chat_template

        self._merge_needed = _needs_system_merge(self._tokenizer)
        print(f"[GemmaChatLM] system→user merge needed: {self._merge_needed}")

        model_kwargs: dict[str, Any] = dict(device_map=cfg.device_map, **cfg.load_kwargs)

        if cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )
            print(f"[GemmaChatLM] loading {cfg.model_name} in 4-bit QLoRA mode")
        else:
            model_kwargs["torch_dtype"] = torch_dtype
            # Flash Attention 2: 2–4× speedup on attention for Ampere+ GPUs.
            # Requires float16/bfloat16 and the flash-attn package.
            if torch.cuda.is_available() and torch_dtype != torch.float32:
                try:
                    import flash_attn  # noqa: F401
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    print("[GemmaChatLM] Flash Attention 2 enabled")
                except ImportError:
                    pass  # flash-attn not installed — standard attention

        self._model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

        if cfg.adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, cfg.adapter_path)

        self._model.eval()

        # A100 / H100 inference speedups — auto-detected by VRAM > 20 GB.
        # Skipped for 4-bit quantized models (BitsAndBytes + torch.compile don't mix).
        if not cfg.load_in_4bit:
            try:
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    if props.total_memory > 20 * 1024 ** 3:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                        self._model = torch.compile(self._model, mode="reduce-overhead")
                        print(f"[GemmaChatLM] {props.name} — torch.compile + tf32 enabled")
                        # Warmup: drain JIT compilation so timed inference isn't penalised.
                        print("[GemmaChatLM] warming up compiled model (3 passes)...")
                        _w = self._tokenizer("warmup", return_tensors="pt").to(self._model.device)
                        for _ in range(3):
                            with torch.inference_mode():
                                self._model.generate(
                                    **_w, max_new_tokens=4, do_sample=False,
                                    pad_token_id=self._tokenizer.pad_token_id,
                                )
                        del _w
                        print("[GemmaChatLM] warmup done")
            except Exception as e:
                print(f"[GemmaChatLM] compile/warmup skipped: {e}")

        print(f"[GemmaChatLM] ready — {cfg.model_name}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_eos_ids(self) -> list[int]:
        eos_ids: list[int] = []
        raw_eos = self._tokenizer.eos_token_id
        if isinstance(raw_eos, list):
            eos_ids = [t for t in raw_eos if isinstance(t, int)]
        elif isinstance(raw_eos, int):
            eos_ids = [raw_eos]
        unk_id = getattr(self._tokenizer, "unk_token_id", None)
        tid = self._tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(tid, int) and tid and tid != unk_id and tid not in eos_ids:
            eos_ids.append(tid)
        return eos_ids

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool = False,
        enable_thinking: bool = False,  # kept for API compat, ignored for Gemma
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
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool = False,
        enable_thinking: bool = False,  # kept for API compat, ignored for Gemma
    ) -> list[str]:
        if self._model is None or self._tokenizer is None:
            self.load()

        import torch

        prompts: list[str] = []
        for messages in messages_list:
            chat = [{"role": m.role, "content": m.content} for m in messages]
            if self._merge_needed:
                chat = _merge_system_into_first_user(chat)
            prompts.append(
                self._tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            )

        # Left-pad so all sequences are right-aligned at the generation boundary.
        orig_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"
        inputs = self._tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self._model.device)
        self._tokenizer.padding_side = orig_side

        eos_ids = self._build_eos_ids()
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": eos_ids,
        }
        if do_sample and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        # Strip the (left-padded) prompt prefix — all seqs share the same padded length.
        input_len = inputs["input_ids"].shape[1]
        results: list[str] = []
        for ids in output_ids:
            new_tokens = ids[input_len:]
            results.append(
                self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            )
        return results
