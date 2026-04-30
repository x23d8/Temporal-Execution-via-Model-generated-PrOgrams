"""main.py — Run evaluation experiments locally (Ollama) or on Kaggle (HuggingFace).

Mirrors the PotCot.ipynb notebook but runs from the command line against
Dataset/ on disk.

Usage — Ollama (local):
    python main.py                        # run all experiments with gemma4:e4b
    python main.py --methods zero_shot    # only zero-shot (4 datasets)
    python main.py --datasets bigbench_date vlsp_date  # 2 datasets, all methods
    python main.py --model gemma3:4b      # different Ollama model

Usage — HuggingFace / Kaggle:
    python main.py --models google/gemma-2-2b-it
    python main.py --models google/gemma-2-2b-it,mistralai/Mistral-7B-v0.1
    python main.py --models google/gemma-2-2b-it --load-in-4bit
    python main.py --models google/gemma-2-2b-it --hf_token hf_xxx...

    # Kaggle notebook cell (token auto-read from Kaggle secrets):
    !python main.py --models google/gemma-2-2b-it --load-in-4bit

Other flags:
    python main.py --per-sample           # compact log for every sample
    python main.py --list                 # print experiment table and exit
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# ── repo root on sys.path so src.* imports work when run as `python main.py`
sys.path.insert(0, str(Path(__file__).parent))

from src.models.ollama import OllamaChatLM, OllamaConfig
from src.runner import RunConfig, run

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent
OUTPUT_DIR  = REPO_ROOT / "outputs"
DATASET_DIR = REPO_ROOT / "Dataset"

# ── experiment table (mirrors notebook cells 1-12) ────────────────────────────
EXPERIMENTS: list[dict] = [
    # Zero-shot (EXP 1-4)
    # dict(experiment_name="zero_shot_udst_duration",    method="zero_shot",    dataset="udst_duration",  k_shot=0),
    dict(experiment_name="zero_shot_bigbench_date",    method="zero_shot",    dataset="bigbench_date",  k_shot=0),
    dict(experiment_name="zero_shot_vlsp_date",        method="zero_shot",    dataset="vlsp_date",      k_shot=0),
    # dict(experiment_name="zero_shot_vlsp_duration",    method="zero_shot",    dataset="vlsp_duration",  k_shot=0),
    # Few-shot (EXP 5-8)
    # dict(experiment_name="few_shot_udst_duration",     method="few_shot",     dataset="udst_duration",  k_shot=4),
    dict(experiment_name="few_shot_bigbench_date",     method="few_shot",     dataset="bigbench_date",  k_shot=3),
    dict(experiment_name="few_shot_vlsp_date",         method="few_shot",     dataset="vlsp_date",      k_shot=3),
    # dict(experiment_name="few_shot_vlsp_duration",     method="few_shot",     dataset="vlsp_duration",  k_shot=4),
    # Symbolic CoT (EXP 9-12)
    dict(experiment_name="symbolic_cot_udst_duration", method="symbolic_cot", dataset="udst_duration",  k_shot=0),
    dict(experiment_name="symbolic_cot_bigbench_date", method="symbolic_cot", dataset="bigbench_date",  k_shot=0),
    dict(experiment_name="symbolic_cot_vlsp_date",     method="symbolic_cot", dataset="vlsp_date",      k_shot=0),
    dict(experiment_name="symbolic_cot_vlsp_duration", method="symbolic_cot", dataset="vlsp_duration",  k_shot=0),
    # Free Think — no system prompt, model reasons freely, custom regex+code extractor (EXP 13-16)
    dict(experiment_name="free_think_udst_duration",   method="free_think",   dataset="udst_duration",  k_shot=0),
    dict(experiment_name="free_think_bigbench_date",   method="free_think",   dataset="bigbench_date",  k_shot=0),
    dict(experiment_name="free_think_vlsp_date",       method="free_think",   dataset="vlsp_date",      k_shot=0),
    dict(experiment_name="free_think_vlsp_duration",   method="free_think",   dataset="vlsp_duration",  k_shot=0),
]

ALL_METHODS  = ["zero_shot", "few_shot", "symbolic_cot", "free_think"]
ALL_DATASETS = ["udst_duration", "bigbench_date", "vlsp_date", "vlsp_duration"]


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _show_device_info() -> None:
    """Print GPU (VRAM) and RAM availability before the run."""
    print("-" * 60)
    print("  Device info")
    print("-" * 60)

    # ── GPU via nvidia-smi ────────────────────────────────────────
    gpu_found = False
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        for line in out.strip().splitlines():
            idx, name, total, free, util = [s.strip() for s in line.split(",")]
            total_mb, free_mb = int(total), int(free)
            used_mb = total_mb - free_mb
            bar_len = 20
            filled = round(bar_len * used_mb / total_mb) if total_mb else 0
            bar = "#" * filled + "." * (bar_len - filled)
            print(
                f"  GPU {idx}: {name}\n"
                f"    VRAM  {used_mb:>6} / {total_mb} MB used  [{bar}]  util={util}%\n"
                f"    Free  {free_mb} MB ({free_mb/total_mb*100:.1f}%)"
            )
        gpu_found = True
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass  # nvidia-smi not available

    # ── fallback: try torch for GPU info (covers ROCm / MPS too) ─
    if not gpu_found:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    total_mb = props.total_memory // 1024 ** 2
                    free_b, total_b = torch.cuda.mem_get_info(i)
                    free_mb = free_b // 1024 ** 2
                    used_mb = total_mb - free_mb
                    bar_len = 20
                    filled = round(bar_len * used_mb / total_mb) if total_mb else 0
                    bar = "#" * filled + "." * (bar_len - filled)
                    print(
                        f"  GPU {i}: {props.name}\n"
                        f"    VRAM  {used_mb:>6} / {total_mb} MB used  [{bar}]"
                    )
                gpu_found = True
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                print("  GPU: Apple MPS (unified memory — see RAM below)")
                gpu_found = True
        except ImportError:
            pass

    if not gpu_found:
        print("  GPU: none detected (nvidia-smi not found, torch not installed)")

    # ── RAM ───────────────────────────────────────────────────────
    ram_shown = False
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        bar_len = 20
        filled = round(bar_len * vm.percent / 100)
        bar = "#" * filled + "." * (bar_len - filled)
        print(
            f"  RAM   {_fmt_bytes(vm.used):>10} / {_fmt_bytes(vm.total)} used  "
            f"[{bar}]  {vm.percent:.1f}%\n"
            f"    Available  {_fmt_bytes(vm.available)}"
        )
        ram_shown = True
    except ImportError:
        pass

    if not ram_shown:
        # Windows fallback via wmic
        try:
            out = subprocess.check_output(
                ["wmic", "OS", "get", "FreePhysicalMemory,TotalVisibleMemorySize", "/Value"],
                stderr=subprocess.DEVNULL, text=True, timeout=10,
            )
            vals: dict[str, int] = {}
            for line in out.strip().splitlines():
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    if v.strip().isdigit():
                        vals[k.strip()] = int(v.strip()) * 1024  # KB → bytes
            total = vals.get("TotalVisibleMemorySize", 0)
            free  = vals.get("FreePhysicalMemory", 0)
            used  = total - free
            pct   = used / total * 100 if total else 0
            bar_len = 20
            filled = round(bar_len * pct / 100)
            bar = "#" * filled + "." * (bar_len - filled)
            print(
                f"  RAM   {_fmt_bytes(used):>10} / {_fmt_bytes(total)} used  "
                f"[{bar}]  {pct:.1f}%\n"
                f"    Available  {_fmt_bytes(free)}"
            )
            ram_shown = True
        except Exception:
            pass

    if not ram_shown:
        print("  RAM: unavailable (install psutil for details)")

    print("-" * 60)


def _print_table() -> None:
    print(f"{'#':<4} {'experiment':<40} {'method':<14} {'dataset':<22} {'k_shot'}")
    print("-" * 90)
    for i, e in enumerate(EXPERIMENTS, 1):
        print(f"{i:<4} {e['experiment_name']:<40} {e['method']:<14} {e['dataset']:<22} {e['k_shot']}")


def _hf_login(token: str | None = None) -> None:
    """Authenticate with HuggingFace Hub for gated models.

    Resolution order:
      1. --hf_token CLI argument
      2. Kaggle secret 'HF_TOKEN'  (Kaggle notebooks only)
      3. Environment variable HF_TOKEN
      4. Environment variable HUGGING_FACE_HUB_TOKEN
    """
    import os
    resolved = token

    if not resolved:
        try:
            from kaggle_secrets import UserSecretsClient
            resolved = UserSecretsClient().get_secret("HF_TOKEN")
            print("[main] HF token loaded from Kaggle secrets")
        except Exception:
            pass

    if not resolved:
        resolved = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if resolved:
            print("[main] HF token loaded from environment variable")

    if resolved:
        from huggingface_hub import login
        login(token=resolved, add_to_git_credential=False)
        print("[main] HuggingFace login successful")
    else:
        print("[main] No HF token — only public models will be accessible")


def _model_slug(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", model_id)


def _build_cfg(exp: dict, *, per_sample: bool, enable_thinking: bool,
               n_hypotheses: int, max_correction_attempts: int,
               output_dir: str = str(OUTPUT_DIR)) -> RunConfig:
    return RunConfig(
        experiment_name=exp["experiment_name"],
        method=exp["method"],
        dataset=exp["dataset"],
        seed=42,
        model_name="gemma4:e4b",
        dtype="bfloat16",
        enable_thinking=enable_thinking,
        k_shot=exp["k_shot"],
        max_samples=...,
        output_dir=output_dir,
        progress_every=50,
        verbose=True,               # always show first-5 full detail
        verbose_first_n=5,          # full log: Q / raw / extracted / gold / time
        verbose_every=1 if per_sample else 0,  # compact line every sample only with --per-sample
        running_score_every=50,     # running avg metric every 50 samples
        n_hypotheses=n_hypotheses,
        max_correction_attempts=max_correction_attempts,
        inference_batch_size=1,
    )


def _run_experiments(
    selected: list[dict],
    model,
    model_label: str,
    out_root: Path,
    args,
) -> tuple[list[dict], list[str]]:
    """Run all selected experiments with a single already-loaded model."""
    summary: list[dict] = []
    failed:  list[str]  = []

    for i, exp in enumerate(selected, 1):
        sep = "─" * 72
        print(f"\n{sep}")
        print(f"  EXP {i}/{len(selected)} — {exp['experiment_name']}  [{model_label}]")
        print(sep)

        cfg = _build_cfg(
            exp,
            per_sample=args.per_sample,
            enable_thinking=args.enable_thinking,
            n_hypotheses=args.n_hypotheses,
            max_correction_attempts=args.max_correction_attempts,
            output_dir=str(out_root),
        )
        cfg.model_name = model_label

        try:
            result  = run(cfg, model=model)
            metrics = result["metrics"]
            summary.append({"experiment": exp["experiment_name"], "metrics": metrics, "model": model_label})

            out_dir   = out_root / exp["method"] / exp["dataset"]
            pred_file = out_dir / "predictions.jsonl"
            n_saved   = sum(1 for _ in pred_file.open(encoding="utf-8")) if pred_file.exists() else 0
            print(f"\n── Saved outputs ───────────────────────────────")
            print(f"  predictions ({n_saved} rows): {pred_file}")
            print(f"  metrics    : {out_dir / 'metrics.json'}")
            print(f"  summary    : {out_root / 'summary.csv'}")
            print(f"───────────────────────────────────────────────")
            print(f"  metrics: {json.dumps(metrics, ensure_ascii=False)}")

        except Exception as exc:
            print(f"[main] ERROR in {exp['experiment_name']}: {exc}")
            failed.append(exp["experiment_name"])

    return summary, failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods",   nargs="+", choices=ALL_METHODS,  default=ALL_METHODS,  metavar="METHOD",  help="methods to run (default: all)")
    ap.add_argument("--datasets",  nargs="+", choices=ALL_DATASETS, default=ALL_DATASETS, metavar="DATASET", help="datasets to run (default: all)")
    ap.add_argument("--per-sample", action="store_true", dest="per_sample",
                    help="compact log line for every sample")
    ap.add_argument("--enable-thinking", action="store_true", dest="enable_thinking",
                    help="enable extended thinking (Ollama/HF models that support it)")
    ap.add_argument("--n-hypotheses",            type=int, default=1, dest="n_hypotheses")
    ap.add_argument("--max-correction-attempts", type=int, default=1, dest="max_correction_attempts")
    # ── Ollama (local) ────────────────────────────────────────────────────────
    ap.add_argument("--ollama-url", default="http://localhost:11434", dest="ollama_url")
    ap.add_argument("--model",      default="gemma4:e4b",             dest="model_name",
                    help="Ollama model tag — used when --models is NOT given")
    # ── HuggingFace / Kaggle ──────────────────────────────────────────────────
    ap.add_argument("--models", default=None, dest="hf_models",
                    help=(
                        "Comma-separated HuggingFace model IDs to evaluate. "
                        "When set, uses transformers instead of Ollama. "
                        "E.g. --models google/gemma-2-2b-it,mistralai/Mistral-7B-v0.1"
                    ))
    ap.add_argument("--hf_token", default=None, dest="hf_token",
                    help="HuggingFace token for gated models (falls back to Kaggle secret / HF_TOKEN env var)")
    ap.add_argument("--load-in-4bit", action="store_true", dest="load_in_4bit",
                    help="Load HF model in 4-bit (BitsAndBytes) — recommended for Kaggle T4/P100")
    ap.add_argument("--load-in-8bit", action="store_true", dest="load_in_8bit",
                    help="Load HF model in 8-bit (BitsAndBytes)")
    ap.add_argument("--hf-fallback", default=None, dest="hf_fallback",
                    help=(
                        "HuggingFace model ID to use when Ollama is not reachable. "
                        "E.g. --hf-fallback google/gemma-2-2b-it"
                    ))
    ap.add_argument("--list", action="store_true", help="print experiment table and exit")
    args = ap.parse_args()

    if args.list:
        _print_table()
        return

    # Preprocess dataset if needed
    preprocessed = DATASET_DIR / "Preprocessed"
    if not any(preprocessed.glob("*.jsonl")):
        print("[main] Preprocessed data not found — running preprocess step…")
        subprocess.run([sys.executable, "-m", "src.data.preprocess"], check=True)

    # Filter experiments
    selected = [
        e for e in EXPERIMENTS
        if e["method"] in args.methods and e["dataset"] in args.datasets
    ]
    if not selected:
        print("[main] No experiments match the given --methods / --datasets filters.")
        return

    print(f"[main] Selected {len(selected)} experiment(s): {[e['experiment_name'] for e in selected]}")
    _show_device_info()

    all_summary: list[dict] = []
    all_failed:  list[str]  = []

    # ── HuggingFace path (--models) ───────────────────────────────────────────
    if args.hf_models:
        _hf_login(args.hf_token)

        from src.models.hf import HFChatLM, HFConfig

        model_ids = [m.strip() for m in args.hf_models.split(",") if m.strip()]
        print(f"[main] HF models to evaluate: {model_ids}")

        for model_id in model_ids:
            slug     = _model_slug(model_id)
            out_root = OUTPUT_DIR / slug
            out_root.mkdir(parents=True, exist_ok=True)
            print(f"\n{'═'*72}")
            print(f"  MODEL: {model_id}")
            print(f"  Output: {out_root}")
            print(f"{'═'*72}")

            hf_cfg = HFConfig(
                model_name=model_id,
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
            )
            model = HFChatLM(hf_cfg)
            model.load()

            summary, failed = _run_experiments(selected, model, model_id, out_root, args)
            all_summary.extend(summary)
            all_failed.extend(failed)

            model.unload()

    # ── Ollama path (default) ─────────────────────────────────────────────────
    else:
        if not OllamaChatLM.is_available(args.ollama_url):
            if args.hf_fallback:
                print(
                    f"[main] Ollama not reachable at {args.ollama_url} — "
                    f"falling back to HF model: {args.hf_fallback}"
                )
                args.hf_models = args.hf_fallback
                _hf_login(args.hf_token)

                from src.models.hf import HFChatLM, HFConfig

                slug     = _model_slug(args.hf_fallback)
                out_root = OUTPUT_DIR / slug
                out_root.mkdir(parents=True, exist_ok=True)

                hf_cfg = HFConfig(
                    model_name=args.hf_fallback,
                    load_in_4bit=args.load_in_4bit,
                    load_in_8bit=args.load_in_8bit,
                )
                model = HFChatLM(hf_cfg)
                model.load()

                summary, failed = _run_experiments(selected, model, args.hf_fallback, out_root, args)
                all_summary.extend(summary)
                all_failed.extend(failed)
                model.unload()
            else:
                print(
                    f"[main] ✗ Ollama not reachable at {args.ollama_url}. "
                    f"Start Ollama or pass --hf-fallback MODEL_ID to use a HuggingFace model instead."
                )
                return
        else:
            print(f"[main] Ollama model: {args.model_name}  |  Output: {OUTPUT_DIR}")
            model_cfg = OllamaConfig(model_name=args.model_name, base_url=args.ollama_url)
            model = OllamaChatLM(model_cfg)
            model.load()

            summary, failed = _run_experiments(selected, model, args.model_name, OUTPUT_DIR, args)
            all_summary.extend(summary)
            all_failed.extend(failed)

    # ── Final summary ─────────────────────────────────────────────────────────
    total = len(all_summary) + len(all_failed)
    print("\n" + "═" * 72)
    print(f"  DONE — {len(all_summary)}/{total} experiments succeeded")
    if all_failed:
        print(f"  Failed: {all_failed}")
    print("═" * 72)
    for row in all_summary:
        m = row["metrics"]
        score_str = (
            f"f1={m['f1']:.4f}" if "f1" in m
            else f"acc={m['accuracy']:.4f}" if "accuracy" in m
            else str(m)
        )
        model_tag = f"[{_model_slug(row['model'])[:20]}]"
        print(f"  {model_tag:<22} {row['experiment']:<40} {score_str}")


if __name__ == "__main__":
    main()
