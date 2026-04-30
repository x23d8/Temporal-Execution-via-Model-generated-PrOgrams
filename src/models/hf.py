"""HuggingFace transformers wrapper — drop-in for OllamaChatLM on Kaggle/Colab.

Implements the same ChatLM protocol so it can be passed directly to run().
Supports 4-bit quantisation (BitsAndBytes) for T4 / P100 budgets.
"""

from __future__ import annotations

import gc
import re
from dataclasses import dataclass

from .base import ChatMessage

# Matches any turn-boundary template tag that should never appear in a clean answer.
# If the decoded output contains one, everything from that point is prompt echo.
_TEMPLATE_TAG_RE = re.compile(
    r"<\|im_end\|>|<\|im_start\|>|<end_of_turn>|<start_of_turn>|<\|eot_id\|>",
    re.IGNORECASE,
)


def _merge_system_into_first_user(chat: list[dict]) -> list[dict]:
    """Prepend system content to the first user message.

    Matches Ollama's behavior for models without a native system role
    (e.g. Gemma), where the system prompt is prepended to the first user
    turn rather than emitted as a separate turn.
    """
    if not chat or chat[0]["role"] != "system":
        return chat
    sys_content = chat[0]["content"]
    merged, prepended = [], False
    for msg in chat[1:]:
        if msg["role"] == "user" and not prepended:
            merged.append({"role": "user", "content": f"{sys_content}\n\n{msg['content']}"})
            prepended = True
        else:
            merged.append(msg)
    return merged


def _normalize_chat_for_template(tok, chat: list[dict]) -> list[dict]:
    """Merge system into first user turn when the tokenizer remaps system→user.

    Gemma's chat template converts system messages to user turns, producing
    two consecutive user turns.  We detect this with a probe render and merge
    so the token sequence matches what Ollama sends to the model.
    """
    if not chat or chat[0]["role"] != "system":
        return chat
    try:
        probe = tok.apply_chat_template(
            [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
            tokenize=False, add_generation_prompt=False,
        )
        # Detect system→user remapping in both Gemma and ChatML format:
        #   Gemma:  <start_of_turn>user appears twice
        #   ChatML: <|im_start|>user appears twice
        if probe.count("<start_of_turn>user") >= 2 or probe.count("<|im_start|>user") >= 2:
            return _merge_system_into_first_user(chat)
    except Exception:
        pass
    return chat


@dataclass
class HFConfig:
    model_name: str = "google/gemma-2-2b-it"
    device_map: str = "auto"
    torch_dtype: str = "float16"        # "float16" | "bfloat16" | "float32"
    load_in_4bit: bool = False           # BitsAndBytes 4-bit — saves ~4× VRAM
    load_in_8bit: bool = False           # BitsAndBytes 8-bit — saves ~2× VRAM
    trust_remote_code: bool = False
    # Generic fallback when the tokenizer has no built-in chat template.
    # Uses ChatML format which is broadly understood by instruction-tuned models.
    fallback_template: str = (
        "{% for m in messages %}"
        "{% if m['role'] == 'system' %}<|im_start|>system\n{{ m['content'] }}<|im_end|>\n"
        "{% elif m['role'] == 'user' %}<|im_start|>user\n{{ m['content'] }}<|im_end|>\n"
        "{% elif m['role'] == 'assistant' %}<|im_start|>assistant\n{{ m['content'] }}<|im_end|>\n"
        "{% endif %}{% endfor %}<|im_start|>assistant\n"
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
            # Merge system into first user for models that remap system→user (Gemma).
            # This matches how Ollama formats the prompt for the same model family.
            chat = _normalize_chat_for_template(tok, chat)
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

        # Include model-specific turn-end tokens so generation stops after the
        # answer turn instead of looping back into the prompt (Gemma: <end_of_turn>,
        # Qwen: <|im_end|>, Llama3: <|eot_id|>).
        eos_ids = [tok.eos_token_id]
        for turn_end in ["<end_of_turn>", "<|im_end|>", "<|eot_id|>"]:
            tid = tok.convert_tokens_to_ids(turn_end)
            if tid and tid not in (tok.unk_token_id, tok.eos_token_id) and tid not in eos_ids:
                eos_ids.append(tid)

        gen_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=eos_ids,
        )
        if do_sample and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = mdl.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens (strip the prompt)
        input_len = inputs["input_ids"].shape[1]
        results: list[str] = []
        for ids in output_ids:
            new_ids = ids[input_len:]
            text = tok.decode(new_ids, skip_special_tokens=True).strip()
            # If any template tag survived decoding (generated as text rather than
            # a special token), everything from that tag onwards is prompt echo —
            # keep only the content that precedes it.
            m = _TEMPLATE_TAG_RE.search(text)
            if m:
                text = text[:m.start()].strip()
            results.append(text)

        return results
