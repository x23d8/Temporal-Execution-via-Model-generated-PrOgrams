"""Ollama local API wrapper — drop-in replacement for QwenChatLM.

Uses Ollama's native /api/chat endpoint (no extra Python dependencies beyond stdlib).
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

from .base import ChatMessage


@dataclass
class OllamaConfig:
    model_name: str = "gemma4:e4b"
    base_url: str = "http://localhost:11434"
    timeout: int = 120


class OllamaChatLM:
    def __init__(self, config: OllamaConfig | None = None):
        self.config = config or OllamaConfig()

    def load(self) -> None:
        url = f"{self.config.base_url}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
            names = [m["name"] for m in data.get("models", [])]
            if any(self.config.model_name in n for n in names):
                print(f"[OllamaChatLM] model '{self.config.model_name}' found in Ollama.")
            else:
                print(
                    f"[OllamaChatLM] ⚠ model '{self.config.model_name}' not found. "
                    f"Run: ollama pull {self.config.model_name}\n"
                    f"Available: {names}"
                )
        except Exception as e:
            print(f"[OllamaChatLM] ⚠ Cannot connect to Ollama at {self.config.base_url}: {e}")
        print(f"[OllamaChatLM] ready — {self.config.model_name} @ {self.config.base_url}")

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        payload: dict = {
            "model": self.config.model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature if do_sample else 0.0,
            },
        }
        if enable_thinking:
            payload["think"] = True

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.config.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            result = json.loads(resp.read())
        # Strip thinking block if model emitted one but enable_thinking was False
        content: str = result["message"]["content"]
        if not enable_thinking and content.startswith("<think>"):
            end = content.find("</think>")
            if end != -1:
                content = content[end + len("</think>"):].lstrip()
        return content

    def generate_batch(
        self,
        messages_list: list[list[ChatMessage]],
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool = False,
        enable_thinking: bool = False,
    ) -> list[str]:
        # Ollama does not support true batching — process sequentially
        return [
            self.generate(
                msgs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                enable_thinking=enable_thinking,
            )
            for msgs in messages_list
        ]
