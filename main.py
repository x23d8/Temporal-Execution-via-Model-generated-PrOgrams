"""main.py — Run all 12 experiments locally with Ollama (gemma4:e4b).

Mirrors the PotCot.ipynb notebook but runs from the command line against
Dataset/ on disk and Ollama instead of Colab + transformers.

Usage:
    python main.py                        # run all 12 experiments
    python main.py --methods zero_shot    # only zero-shot (4 datasets)
    python main.py --datasets bigbench_date vlsp_date  # 2 datasets, all methods
    python main.py --methods symbolic_cot --datasets udst_duration
    python main.py --per-sample           # compact log for every sample
    python main.py --list                 # print experiment table and exit
"""
from __future__ import annotations

import argparse
import json
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


def _build_cfg(exp: dict, *, per_sample: bool, enable_thinking: bool,
               n_hypotheses: int, max_correction_attempts: int) -> RunConfig:
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
        output_dir=str(OUTPUT_DIR),
        progress_every=50,
        verbose=True,               # always show first-5 full detail
        verbose_first_n=5,          # full log: Q / raw / extracted / gold / time
        verbose_every=1 if per_sample else 0,  # compact line every sample only with --per-sample
        running_score_every=50,     # running avg metric every 50 samples
        n_hypotheses=n_hypotheses,
        max_correction_attempts=max_correction_attempts,
        inference_batch_size=1,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods",   nargs="+", choices=ALL_METHODS,  default=ALL_METHODS,  metavar="METHOD",  help="methods to run (default: all three)")
    ap.add_argument("--datasets",  nargs="+", choices=ALL_DATASETS, default=ALL_DATASETS, metavar="DATASET", help="datasets to run (default: all four)")
    ap.add_argument("--per-sample", action="store_true", dest="per_sample",
                    help="add a compact log line for every sample (default: only first 5 + avg/50)")
    ap.add_argument("--enable-thinking", action="store_true", dest="enable_thinking",
                    help="enable gemma4 extended thinking (slower but may improve accuracy)")
    ap.add_argument("--n-hypotheses",         type=int, default=1, dest="n_hypotheses",         help="symbolic_cot: independent programs per sample")
    ap.add_argument("--max-correction-attempts", type=int, default=1, dest="max_correction_attempts", help="symbolic_cot: self-correction attempts on exec failure")
    ap.add_argument("--ollama-url",  default="http://localhost:11434", dest="ollama_url",  help="Ollama base URL")
    ap.add_argument("--model",       default="gemma4:e4b",             dest="model_name",  help="Ollama model tag (default: gemma4:e4b)")
    ap.add_argument("--list",        action="store_true", help="print experiment table and exit")
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
    print(f"[main] Output dir: {OUTPUT_DIR}")

    _show_device_info()

    # Load model once, reuse across all experiments
    model_cfg = OllamaConfig(model_name=args.model_name, base_url=args.ollama_url)
    model = OllamaChatLM(model_cfg)
    model.load()

    summary: list[dict] = []
    failed: list[str]   = []

    for i, exp in enumerate(selected, 1):
        sep = "─" * 72
        print(f"\n{sep}")
        print(f"  EXP {i}/{len(selected)} — {exp['experiment_name']}")
        print(sep)

        cfg = _build_cfg(
            exp,
            per_sample=args.per_sample,
            enable_thinking=args.enable_thinking,
            n_hypotheses=args.n_hypotheses,
            max_correction_attempts=args.max_correction_attempts,
        )
        # Point runner at the local model (override the QwenChatLM default)
        cfg.model_name = args.model_name

        try:
            result = run(cfg, model=model)
            metrics = result["metrics"]
            summary.append({"experiment": exp["experiment_name"], "metrics": metrics})

            # saved-file report (mirrors notebook run_exp helper)
            out_dir   = OUTPUT_DIR / exp["method"] / exp["dataset"]
            pred_file = out_dir / "predictions.jsonl"
            n_saved   = sum(1 for _ in pred_file.open(encoding="utf-8")) if pred_file.exists() else 0
            print(f"\n── Saved outputs ───────────────────────────────")
            print(f"  predictions ({n_saved} rows): {pred_file}")
            print(f"  metrics    : {out_dir / 'metrics.json'}")
            print(f"  summary    : {OUTPUT_DIR / 'summary.csv'}")
            print(f"───────────────────────────────────────────────\n")
            print(f"  metrics: {json.dumps(metrics, ensure_ascii=False)}")

        except Exception as exc:
            print(f"[main] ERROR in {exp['experiment_name']}: {exc}")
            failed.append(exp["experiment_name"])

    # Final summary
    print("\n" + "═" * 72)
    print(f"  DONE — {len(selected) - len(failed)}/{len(selected)} experiments succeeded")
    if failed:
        print(f"  Failed: {failed}")
    print("═" * 72)
    for row in summary:
        m = row["metrics"]
        score_str = (
            f"f1={m['f1']:.4f}" if "f1" in m
            else f"acc={m['accuracy']:.4f}" if "accuracy" in m
            else str(m)
        )
        print(f"  {row['experiment']:<42} {score_str}")


if __name__ == "__main__":
    main()
