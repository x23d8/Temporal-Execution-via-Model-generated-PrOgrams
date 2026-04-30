"""Helpers để chuyển Sample sang SFT chat-format string cho Gemma-4-E4B.

Khác với Qwen version (duc_temporal):
- System prompt được merge vào first user turn (Gemma không có native system role).
- Response template cho loss masking: '<start_of_turn>model\n' (Gemma format).
- Không có enable_thinking (Gemma không hỗ trợ).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Sequence

from ..data.schema import Sample
from ..prompts.templates import build_messages, get_template

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase


def split_train_val(
    samples: Sequence[Sample],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """Shuffle deterministically, split off val from the tail."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    items = list(samples)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = max(1, int(round(len(items) * val_ratio)))
    return items[:-n_val], items[-n_val:]


def _merge_system_into_first_user(chat: list[dict]) -> list[dict]:
    """Prepend system content to first user message.

    Gemma's chat template remaps system→user, producing two consecutive user
    turns. We merge manually to match what Gemma actually sees at inference.
    """
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


def _needs_system_merge(tokenizer: "PreTrainedTokenizerBase") -> bool:
    """Return True if the tokenizer remaps system→user (requires manual merge).

    Mirrors gemma.py's probe so training and inference use identical logic.
    """
    try:
        probe = tokenizer.apply_chat_template(
            [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
            tokenize=False,
            add_generation_prompt=False,
        )
        return probe.count("<start_of_turn>user") >= 2 or probe.count("<|im_start|>user") >= 2
    except Exception:
        return True


def _render_one(
    sample: Sample,
    tokenizer: "PreTrainedTokenizerBase",
    merge_system: bool,
) -> str:
    """Convert one Sample to a full chat-format string (system+user+assistant)."""
    msgs = build_messages(sample, shots=(), enable_thinking=False)
    chat = [{"role": m.role, "content": m.content} for m in msgs]
    if merge_system:
        chat = _merge_system_into_first_user(chat)
    tmpl = get_template(sample["task"], sample["language"])
    chat.append({"role": "assistant", "content": tmpl.render_shot_assistant(sample)})
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=False
    )


def samples_to_chat_dataset(
    samples: Sequence[Sample],
    tokenizer: "PreTrainedTokenizerBase",
) -> "Dataset":
    """Convert list[Sample] -> datasets.Dataset with a single 'text' column."""
    from datasets import Dataset

    merge_system = _needs_system_merge(tokenizer)
    rows = [{"text": _render_one(s, tokenizer, merge_system)} for s in samples]
    return Dataset.from_list(rows)


def resolve_assistant_response_template(
    tokenizer: "PreTrainedTokenizerBase",
) -> str:
    """Dynamically resolve Gemma's assistant turn header for loss masking.

    Probes the tokenizer by comparing with/without add_generation_prompt to
    extract the suffix that opens the model turn. Falls back to the Gemma
    standard '<start_of_turn>model\n'.
    """
    dummy_user = [{"role": "user", "content": "x"}]
    try:
        with_gen = tokenizer.apply_chat_template(
            dummy_user, tokenize=False, add_generation_prompt=True
        )
        without_gen = tokenizer.apply_chat_template(
            dummy_user, tokenize=False, add_generation_prompt=False
        )
        if with_gen.startswith(without_gen):
            suffix = with_gen[len(without_gen):]
            if suffix:
                return suffix
    except Exception:
        pass
    return "<start_of_turn>model\n"


class CompletionOnlyCollator:
    """Pad batch, mask prompt labels, train only on the assistant response.

    Finds the response_template token sequence in each sample's input_ids,
    then sets labels[:end_of_template] = -100. Samples where the template
    is not found (e.g. truncated) are fully masked — they contribute no loss.
    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        response_template: str,
        pad_to_multiple_of: int | None = 8,
    ) -> None:
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have pad_token_id set.")
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.response_template = response_template
        self.response_token_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        if not self.response_token_ids:
            raise ValueError(
                f"Empty tokenization for response_template={response_template!r}"
            )

    def __call__(self, examples: list[dict]) -> dict:
        import torch

        batch = self.tokenizer.pad(
            [{"input_ids": ex["input_ids"]} for ex in examples],
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        input_ids: torch.Tensor = batch["input_ids"]
        attention_mask: torch.Tensor = batch["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        rti = self.response_token_ids
        n = len(rti)
        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            cut = -1
            for start in range(len(ids) - n + 1):
                if ids[start : start + n] == rti:
                    cut = start + n
                    break
            if cut < 0:
                labels[i, :] = -100
            else:
                labels[i, :cut] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
