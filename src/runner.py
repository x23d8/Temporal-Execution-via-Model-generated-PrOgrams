"""Entry-point runner: load config YAML, chạy method trên dataset, lưu output.

Usage:
    python -m src.runner --config configs/<name>.yaml

Output:
    <output_dir>/<method>/<dataset>/predictions.jsonl
    <output_dir>/<method>/<dataset>/metrics.json
    append 1 row vào <output_dir>/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .data.registry import load_dataset
from .evaluation.evaluate import build_record, score_records
from .evaluation.metrics import avg_inference_time
from .methods.registry import build_method
from .models.qwen import QwenChatLM, QwenConfig
from .prompts.shot_pools import get_shots
from .utils.io import read_jsonl, write_json, write_jsonl
from .utils.seed import set_seed
from .utils.timing import timer


@dataclass
class RunConfig:
    experiment_name: str
    method: str
    dataset: str
    seed: int = 42
    model_name: str = "Qwen/Qwen3.5-9B"
    dtype: str = "bfloat16"
    enable_thinking: bool = False
    k_shot: int = 0
    max_samples: int | None = ...  # ...= dùng default theo dataset
    dataset_path: str | None = None
    output_dir: str = "outputs"
    progress_every: int = 50
    # Verbose controls
    verbose: bool = False            # bật log per-sample
    verbose_first_n: int = 5         # full log cho N sample đầu (raw + extracted + gold + correct)
    verbose_every: int = 0           # >0: log rút gọn mỗi N sample
    running_score_every: int = 100   # in F1/accuracy tạm thời mỗi N sample
    # symbolic_cot controls (bỏ qua với zero_shot / few_shot)
    n_hypotheses: int = 1            # số chương trình độc lập mỗi sample
    max_correction_attempts: int = 1 # số lần self-correction nếu execution fail
    inference_batch_size: int = 1    # batch N samples per generate() call (symbolic_cot only)
    use_planner: bool = True                 # Layer 1: CoT planner
    use_kb_for_duration: bool = True         # Layer 2B: KB hint for duration
    use_retrospective_verify: bool = True    # Layer 5: retrospective verifier


def load_config(path: str | Path) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw.get("max_samples", "default") == "default":
        raw["max_samples"] = ...
    return RunConfig(**raw)


def _summary_row(cfg: RunConfig, metrics: dict, avg_time: float, n: int) -> dict:
    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "experiment": cfg.experiment_name,
        "method": cfg.method,
        "dataset": cfg.dataset,
        "k_shot": cfg.k_shot,
        "enable_thinking": cfg.enable_thinking,
        "seed": cfg.seed,
        "model": cfg.model_name,
        "num_samples": n,
        "avg_inference_sec": round(avg_time, 4),
    }
    if "f1" in metrics:
        row["metric"] = "f1"
        row["score"] = round(metrics["f1"], 4)
        row["precision"] = round(metrics["precision"], 4)
        row["recall"] = round(metrics["recall"], 4)
    elif "accuracy" in metrics:
        row["metric"] = "accuracy"
        row["score"] = round(metrics["accuracy"], 4)
    row["parse_fail"] = metrics.get("parse_fail", 0)
    return row


def _append_summary(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if header_needed:
            writer.writeheader()
        writer.writerow(row)


def _running_score(records: list[dict], task: str, language: str) -> str:
    """Running F1 (duration) or accuracy (date_arith), tagged with language."""
    if not records:
        return "n/a"
    tag = f"[{language.upper()}]"
    if task == "duration":
        from .evaluation.metrics import binary_f1_yes
        m = binary_f1_yes(
            [r["gold_normalized"] for r in records],
            [r["extracted"] for r in records],
        )
        return (
            f"{tag} f1={m['f1']:.3f} p={m['precision']:.3f} r={m['recall']:.3f} "
            f"parse_fail={m['parse_fail']}"
        )
    from .evaluation.metrics import accuracy
    m = accuracy(
        [r["gold_normalized"] for r in records],
        [r["extracted"] for r in records],
    )
    return (
        f"{tag} acc={m['accuracy']:.3f} correct={m['correct']}/{m['support']} "
        f"parse_fail={m['parse_fail']}"
    )


def _log_sample(idx: int, total: int, rec: dict, full: bool) -> None:
    mark = "✓" if rec["correct"] else "✗"
    if full:
        raw_short = rec["raw_output"].replace("\n", "\\n")
        if len(raw_short) > 200:
            raw_short = raw_short[:200] + "…"
        print(
            f"  [{idx+1}/{total}] {mark} "
            f"gold={rec['gold_normalized']!r} "
            f"extracted={rec['extracted']!r} "
            f"elapsed={rec['elapsed_sec']:.2f}s"
        )
        print(f"      Q: {rec['question'][:160]}")
        print(f"      raw: {raw_short}")
    else:
        print(
            f"  [{idx+1}/{total}] {mark} "
            f"gold={rec['gold_normalized']!r} extracted={rec['extracted']!r}"
        )


def run(
    cfg: RunConfig,
    *,
    model: QwenChatLM | None = None,
) -> dict:
    """Chạy 1 experiment. Truyền model đã load để tránh reload giữa các cell."""
    set_seed(cfg.seed)
    print(f"[runner] experiment={cfg.experiment_name} method={cfg.method} dataset={cfg.dataset} "
          f"verbose={cfg.verbose}")

    # Load dataset
    kwargs: dict[str, Any] = {}
    if cfg.dataset_path:
        kwargs["path"] = cfg.dataset_path
    if cfg.max_samples is not ...:
        kwargs["max_samples"] = cfg.max_samples
    samples = load_dataset(cfg.dataset, **kwargs)
    print(f"[runner] loaded {len(samples)} samples "
          f"(task={samples[0]['task']}, lang={samples[0]['language']})")

    # Build / reuse model
    if model is None:
        model = QwenChatLM(QwenConfig(model_name=cfg.model_name, dtype=cfg.dtype))
        model.load()
    else:
        print(f"[runner] reusing pre-loaded model: {model.config.model_name}")

    # Build method
    method_kwargs: dict[str, Any] = {"enable_thinking": cfg.enable_thinking}
    if cfg.method == "few_shot":
        task = samples[0]["task"]
        language = samples[0]["language"]
        method_kwargs["shots"] = get_shots(task, language, cfg.k_shot)
        print(f"[runner] few-shot k={cfg.k_shot} for ({task},{language})")
    if cfg.method == "symbolic_cot":
        method_kwargs["n_hypotheses"] = cfg.n_hypotheses
        method_kwargs["max_correction_attempts"] = cfg.max_correction_attempts
        print(
            f"[runner] symbolic_cot n_hypotheses={cfg.n_hypotheses} "
            f"max_correction_attempts={cfg.max_correction_attempts}"
        )
    method = build_method(cfg.method, model, **method_kwargs)
    # Method may provide its own extractor (e.g. free_think); None falls back to default.
    custom_extractor = getattr(method, "extract_answer", None)

    task     = samples[0]["task"]
    language = samples[0]["language"]
    records: list[dict] = []
    times: list[float] = []
    n = len(samples)

    use_batch = (
        cfg.inference_batch_size > 1
        and hasattr(method, "predict_batch")
    )
    if use_batch:
        print(f"[runner] batch inference: batch_size={cfg.inference_batch_size}")

    out_dir = Path(cfg.output_dir) / cfg.method / cfg.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"

    existing = list(read_jsonl(pred_path)) if pred_path.exists() else []
    records = existing
    times = [r["elapsed_sec"] for r in existing]
    i = len(existing)
    if i:
        print(f"[runner] resuming from sample {i}/{n} ({i} already saved, {n - i} remaining)")

    with open(pred_path, "a" if i else "w", encoding="utf-8") as pred_f:
        while i < n:
            if use_batch:
                batch = samples[i : i + cfg.inference_batch_size]
                with timer() as t:
                    raws = method.predict_batch(batch)
                batch_elapsed = t["elapsed"]
                per_sample_elapsed = batch_elapsed / len(batch)
                for j, (sample, raw) in enumerate(zip(batch, raws)):
                    times.append(per_sample_elapsed)
                    rec = build_record(sample, raw, per_sample_elapsed, extractor=custom_extractor)
                    records.append(rec)
                    pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    pred_f.flush()
                    idx = i + j
                    if cfg.verbose:
                        in_first_n = idx < cfg.verbose_first_n
                        periodic = cfg.verbose_every > 0 and ((idx + 1) % cfg.verbose_every == 0)
                        if in_first_n or periodic:
                            _log_sample(idx, n, rec, full=in_first_n)
                    if cfg.running_score_every > 0 and (idx + 1) % cfg.running_score_every == 0:
                        print(
                            f"  [{idx+1}/{n}] running: {_running_score(records, task, language)} "
                            f"avg_time={sum(times)/len(times):.3f}s"
                        )
                    elif (not cfg.verbose) and (idx + 1) % cfg.progress_every == 0:
                        print(f"  [{idx+1}/{n}] avg={sum(times)/len(times):.3f}s")
                i += len(batch)
            else:
                sample = samples[i]
                with timer() as t:
                    raw = method.predict(sample)
                elapsed = t["elapsed"]
                times.append(elapsed)
                rec = build_record(sample, raw, elapsed, extractor=custom_extractor)
                records.append(rec)
                pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                pred_f.flush()

                if cfg.verbose:
                    in_first_n = i < cfg.verbose_first_n
                    periodic = cfg.verbose_every > 0 and ((i + 1) % cfg.verbose_every == 0)
                    if in_first_n or periodic:
                        _log_sample(i, n, rec, full=in_first_n)

                # Print running score right after the first-N verbose block ends
                if cfg.verbose and i + 1 == cfg.verbose_first_n:
                    print(
                        f"  [after first {cfg.verbose_first_n}] running: "
                        f"{_running_score(records, task, language)} "
                        f"avg_time={sum(times)/len(times):.3f}s"
                    )
                elif cfg.running_score_every > 0 and (i + 1) % cfg.running_score_every == 0:
                    print(
                        f"  [{i+1}/{n}] running: {_running_score(records, task, language)} "
                        f"avg_time={sum(times)/len(times):.3f}s"
                    )
                elif (not cfg.verbose) and (i + 1) % cfg.progress_every == 0:
                    print(f"  [{i+1}/{n}] avg={sum(times)/len(times):.3f}s")
                i += 1

    metrics = score_records(records, task, language)
    avg_t = avg_inference_time(times)
    metrics_payload = {
        "experiment": cfg.experiment_name,
        "method": cfg.method,
        "dataset": cfg.dataset,
        "task": task,
        "language": language,
        "k_shot": cfg.k_shot,
        "enable_thinking": cfg.enable_thinking,
        "seed": cfg.seed,
        "model": cfg.model_name,
        "num_samples": len(samples),
        "avg_inference_sec": avg_t,
        "metrics": metrics,
        "config": cfg.__dict__ if cfg.max_samples is not ... else {**cfg.__dict__, "max_samples": "default"},
    }
    write_json(out_dir / "metrics.json", metrics_payload)
    _append_summary(
        Path(cfg.output_dir) / "summary.csv",
        _summary_row(cfg, metrics, avg_t, len(samples)),
    )
    extra_info = ""
    if hasattr(method, "rule_ratio"):
        extra_info = f" rule_solved={method.rule_ratio:.1%}"
    print(f"[runner] done. metrics: {json.dumps(metrics, ensure_ascii=False)} "
          f"avg_time={avg_t:.3f}s{extra_info} -> {out_dir}")
    return metrics_payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--verbose", action="store_true", help="override cfg.verbose=True")
    ap.add_argument("--verbose-first-n", type=int, default=None)
    ap.add_argument("--verbose-every", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.verbose:
        cfg.verbose = True
    if args.verbose_first_n is not None:
        cfg.verbose_first_n = args.verbose_first_n
    if args.verbose_every is not None:
        cfg.verbose_every = args.verbose_every
    run(cfg)


if __name__ == "__main__":
    main()
