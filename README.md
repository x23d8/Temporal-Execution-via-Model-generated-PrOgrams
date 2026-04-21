# Temporal Reasoning for Small LLMs (< 10B)

Nghiên cứu cải thiện khả năng **temporal reasoning** của LLM nhỏ hơn 10B tham số trên hai ngôn ngữ (English, Vietnamese) và hai loại task (Date Arithmetic, Duration Reasoning). Phase 1 benchmark `Qwen/Qwen3.5-9B` với ba phương pháp prompting: **zero-shot**, **few-shot**, và **symbolic-cot** (program synthesis + symbolic execution).

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Datasets](#datasets)
- [Metric & Output](#metric--output)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt cục bộ](#cài-đặt-cục-bộ)
- [Chạy trên Google Colab](#chạy-trên-google-colab)
- [Chạy thủ công (CLI)](#chạy-thủ-công-cli)
- [Verbose & Debugging](#verbose--debugging)
- [Kiến trúc code](#kiến-trúc-code)
- [Symbolic CoT — Pipeline chi tiết](#symbolic-cot--pipeline-chi-tiết)
- [Mở rộng](#mở-rộng)
- [Reproducibility](#reproducibility)
- [Testing](#testing)

---

## Tổng quan

| Hạng mục | Giá trị |
|---|---|
| **Model** | `Qwen/Qwen3.5-9B` (HuggingFace, hỗ trợ thinking / non-thinking mode) |
| **Ngôn ngữ** | English, Vietnamese |
| **Task** | Date Arithmetic (open-ended), Duration Reasoning (binary yes/no) |
| **Phương pháp Phase 1** | Zero-shot · Few-shot (k cố định) · **Symbolic CoT** (program synthesis + symbolic execution) |
| **Phương pháp tương lai** | Dynamic Few-shot, Chain-of-Thought, Majority Voting, Hybrid |
| **Môi trường** | Google Colab Pro (A100 / L4); code cũng chạy cục bộ không GPU cho preprocess + tests |
| **Pipeline** | Source code trên GitHub → Colab `git clone/pull` → chạy eval → dump kết quả sang Google Drive |

---

## Datasets

| Ngôn ngữ | Task | Source | Phase 1 size |
|---|---|---|---|
| English | Duration | [UDST-DurationQA](Dataset/Raw/English/UDST-DurationQA/data/test.tsv) | 1500 rows đầu của `test.tsv` |
| English | Date Arithmetic | [BigBench DateUnderstanding](Dataset/Raw/English/BigBench_DateUnderstanding/task.json) | **Toàn bộ 369 examples** |
| Vietnamese | Date Arithmetic | [VLSP 2025 ViTempQA — DateArith](Dataset/Raw/Vietnamese/VLSP%202025%20ViTempQA%20%28DateArith%20%2B%20DurationQA%29%20Task/TrainingDataset/date_train_dataset/date_training_dataset.txt) | 1500 dòng đầu JSONL |
| Vietnamese | Duration | [VLSP 2025 ViTempQA — DurationQA](Dataset/Raw/Vietnamese/VLSP%202025%20ViTempQA%20%28DateArith%20%2B%20DurationQA%29%20Task/TrainingDataset/durationQA_train_dataset/duration_training_dataset.txt) | **1500 rows sau khi expand** 4 options/question (≈ 375 questions gốc) |

### Chuẩn hoá (xem [src/data/](src/data/))

- **UDST** — TSV 4 cột (`context, question, candidate_answer, label`) không header → binary classification.
- **BigBench** — JSON `examples[i].target_scores` → giữ duy nhất đáp án có score=1, prompt open-ended sinh `MM/DD/YYYY`.
- **VLSP DateArith** — JSONL `{question, answer, context}` → gold `"Tháng M, YYYY"`.
- **VLSP DurationQA** — JSONL `{context, question, options[4], labels[4], qid}` → expand 4 row binary / question.

Preprocess dump JSONL chuẩn schema vào [Dataset/Preprocessed/](Dataset/Preprocessed/):

```bash
python -m src.data.preprocess
```

---

## Metric & Output

| Task | Metric | Ghi chú |
|---|---|---|
| Duration (EN + VI) | **F1** binary, positive class = `yes` | Sample không parse được coi như predict `no` |
| Date Arithmetic (EN + VI) | **Accuracy** string match sau normalize | |
| Chung | **Avg inference time / sample** (giây) | Log per-dataset |

Mỗi experiment sinh:

- `outputs/<method>/<dataset>/predictions.jsonl` — mỗi dòng: `sample_id, task, language, dataset, question, gold_raw, gold_normalized, raw_output, extracted, correct, elapsed_sec`.
- `outputs/<method>/<dataset>/metrics.json` — snapshot metric + config.
- `outputs/summary.csv` — bảng tổng quan (append mỗi lần chạy).

---

## Cấu trúc thư mục

```
Temporal_Reasoning/
├── Dataset/
│   ├── Raw/                    # read-only, gitignored
│   └── Preprocessed/           # JSONL chuẩn schema (sinh bởi preprocess.py)
├── src/
│   ├── data/                   # loaders + registry + preprocess
│   ├── models/                 # Qwen wrapper + ChatLM protocol
│   ├── prompts/                # templates per (task × ngôn ngữ) + shot pools
│   ├── methods/
│   │   ├── base.py             # Method protocol + DEFAULT_GEN_KWARGS
│   │   ├── registry.py         # METHOD_BUILDERS — đăng ký mọi method tại đây
│   │   ├── zero_shot.py        # ZeroShotMethod
│   │   ├── few_shot.py         # FewShotMethod (fixed / dynamic selector)
│   │   └── symbolic_cot.py     # SymbolicCoTMethod — program synthesis + symbolic execution
│   ├── evaluation/             # extractor, metrics, evaluate
│   └── utils/
│       ├── io.py
│       ├── seed.py
│       ├── timing.py
│       └── temporal_executor.py  # sandbox execution engine cho symbolic_cot
├── configs/                    # 12 YAML (4 datasets × {zero_shot, few_shot, symbolic_cot})
├── notebooks/
│   └── run_phase1_colab.ipynb  # 5 setup + 12 experiment + 3 debug cells
├── tests/
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## Cài đặt cục bộ

Dùng Python 3.10+ (đã test với 3.13).

```bash
git clone <repo_url> Temporal_Reasoning
cd Temporal_Reasoning

# (Khuyên) venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Lưu ý: để **chạy inference** Qwen3.5-9B bạn cần GPU ≥ 16 GB VRAM (khuyến nghị A100/L4 trên Colab). Nếu chỉ muốn thử loader / extractor / preprocess / unit tests thì không cần GPU.

---

## Chạy trên Google Colab

Notebook chính: [notebooks/run_phase1_colab.ipynb](notebooks/run_phase1_colab.ipynb).

Bố cục notebook:

1. **5 SETUP cells** (chạy 1 lần):
   1. Cài `transformers`, `accelerate`, `scikit-learn`, `pyyaml`.
   2. Mount Drive + `git clone/pull` repo.
   3. Config path + symlink `Dataset/` từ Drive.
   4. Chạy preprocess → dump JSONL chuẩn schema.
   5. **Load Qwen3.5-9B một lần** và định nghĩa helper `run_exp(cfg_path, ...)`.

2. **12 EXPERIMENT cells** (mỗi cell 1 experiment, rerun độc lập):

   | Cell | Method | Dataset | Metric |
   |---|---|---|---|
   | EXP 1/12 | zero_shot | udst_duration (EN) | F1 |
   | EXP 2/12 | zero_shot | bigbench_date (EN) | Accuracy |
   | EXP 3/12 | zero_shot | vlsp_date (VI) | Accuracy |
   | EXP 4/12 | zero_shot | vlsp_duration (VI) | F1 |
   | EXP 5/12 | few_shot k=4 | udst_duration (EN) | F1 |
   | EXP 6/12 | few_shot k=3 | bigbench_date (EN) | Accuracy |
   | EXP 7/12 | few_shot k=3 | vlsp_date (VI) | Accuracy |
   | EXP 8/12 | few_shot k=4 | vlsp_duration (VI) | F1 |
   | EXP 9/12 | symbolic_cot | udst_duration (EN) | F1 |
   | EXP 10/12 | symbolic_cot | bigbench_date (EN) | Accuracy |
   | EXP 11/12 | symbolic_cot | vlsp_date (VI) | Accuracy |
   | EXP 12/12 | symbolic_cot | vlsp_duration (VI) | F1 |

3. **3 DEBUG cells**:
   - **A** — audit parse-fail + sample sai đầu tiên của 1 run.
   - **B** — probe 1 sample bất kỳ: in đầy đủ messages + raw output + extracted + gold.
   - **C** — hiển thị `summary.csv`.

Trước khi chạy: đổi `REPO_URL` ở SETUP cell 2 thành URL GitHub của bạn và upload `Dataset/Raw/` lên Drive ở path trùng với `DATASET_ROOT`.

---

## Chạy thủ công (CLI)

Mỗi experiment là 1 YAML config. Chạy:

```bash
# Zero-shot
python -m src.runner --config configs/zero_shot_udst_duration.yaml

# Few-shot
python -m src.runner --config configs/few_shot_vlsp_date.yaml --verbose --verbose-first-n 10

# Symbolic CoT (program synthesis + symbolic execution)
python -m src.runner --config configs/symbolic_cot_bigbench_date.yaml --verbose
```

Output lưu vào `outputs/<method>/<dataset>/` theo đúng cấu trúc ở [Metric & Output](#metric--output).

---

## Verbose & Debugging

`RunConfig` expose các trường kiểm soát log per-sample:

| Field | Mặc định | Ý nghĩa |
|---|---|---|
| `verbose` | `False` | Bật log per-sample |
| `verbose_first_n` | `5` | In **full** (question + raw + extracted + gold + ✓/✗ + time) cho N sample đầu |
| `verbose_every` | `0` | `>0` → in log rút gọn mỗi N sample |
| `running_score_every` | `100` | In running F1 / accuracy mỗi N sample |
| `n_hypotheses` | `1` | *(symbolic_cot)* Số chương trình độc lập mỗi sample; tăng để có multi-hypothesis voting |
| `max_correction_attempts` | `1` | *(symbolic_cot)* Số lần self-correction loop khi execution fail; `0` = không sửa |

Trong Colab notebook, helper `run_exp()` có thể override tất cả các trường này mà không cần sửa YAML:

```python
# Chạy symbolic_cot với 3 hypothesis (voting) và tắt self-correction
run_exp('configs/symbolic_cot_bigbench_date.yaml',
        verbose=True, verbose_every=50,
        n_hypotheses=3, max_correction_attempts=0)
```

---

## Kiến trúc code

### Flow chung (zero-shot / few-shot)

```
YAML config ──► RunConfig ──► load_dataset(name) ──► [Sample, ...]
                                   │
                                   ▼
                        build_method(model, **kwargs)
                                   │
                                   ▼ for each sample
                    method.predict(sample) ──► raw_output
                                   │
                                   ▼
                    build_record(sample, raw, time)   ← extractor + normalize_gold
                                   │
                                   ▼
                    score_records(records, task, lang)
                                   │
                                   ▼
                    predictions.jsonl + metrics.json + summary.csv
```

### Các interface chính

- [src/data/schema.py](src/data/schema.py) — `Sample` TypedDict (`sample_id, task, language, dataset, context, question, gold, meta`).
- [src/models/base.py](src/models/base.py) — `ChatLM` protocol (`generate(messages, ...) -> str`), `ChatMessage`.
- [src/methods/base.py](src/methods/base.py) — `Method` protocol (`predict(sample) -> str`), `DEFAULT_GEN_KWARGS`.
- [src/prompts/templates.py](src/prompts/templates.py) — `PromptTemplate` + `build_messages(sample, shots)`.
- [src/evaluation/extractor.py](src/evaluation/extractor.py) — `extract(task, lang, raw)`, `normalize_gold(task, lang, gold)`; strip `<think>...</think>` cho thinking mode.
- [src/evaluation/metrics.py](src/evaluation/metrics.py) — `binary_f1_yes`, `accuracy`, `avg_inference_time`.

### Registry

- **Datasets** → [src/data/registry.py](src/data/registry.py) (`DATASET_LOADERS`, `DEFAULT_PATHS`, `DEFAULT_MAX_SAMPLES`).
- **Methods** → [src/methods/registry.py](src/methods/registry.py) (`METHOD_BUILDERS: {"zero_shot", "few_shot", "symbolic_cot"}`).
- **Prompt templates** → [src/prompts/templates.py](src/prompts/templates.py) (`TEMPLATES` theo key `(task, language)`).

---

## Symbolic CoT — Pipeline chi tiết

`SymbolicCoTMethod` hiện thực hoá kiến trúc 5-layer với hai cơ chế bổ sung: **self-correction loop** và **multi-hypothesis voting**. Interface `predict(sample) -> str` không đổi — output vẫn đi qua extractor hiện tại như mọi method khác.

### 5 Layers

```
Input question
      │
      ▼  Layer 1 — Temporal Understanding
      │  LLM nhận system prompt chuyên biệt → trích structured temporal info
      │  (embedded trong prompt synthesis bên dưới)
      │
      ▼  Layer 2 — Temporal Normalization
      │  Biểu diễn thời gian được chuẩn hoá trong quá trình program generation
      │
      ▼  Layer 3 — Program Synthesis
      │  LLM sinh Python datetime program (max 512 tokens)
      │  date_arith → tính date bằng date/timedelta/relativedelta
      │  duration   → so sánh candidate với min/max plausible range (giây)
      │
      ▼  Layer 4 — Symbolic Execution  [temporal_executor.py]
      │  exec() trong namespace sandbox (date, datetime, timedelta đã import sẵn)
      │  Kết quả lấy từ biến `answer`; date object tự convert sang đúng format
      │
      ▼  Layer 5 — Verification & Consistency
      │  verify_answer(): kiểm tra format (MM/DD/YYYY, "Tháng M, YYYY", yes/no)
      │                   + calendar constraints (ngày hợp lệ, năm trong 1000–2200)
      │
      ├─► PASS → trả về answer
      │
      └─► FAIL → Self-Correction Loop
                  LLM nhận error message → regenerate program
                  lặp tối đa max_correction_attempts lần
```

### Multi-Hypothesis Voting

Khi `n_hypotheses > 1`, pipeline trên được chạy `n` lần với temperature tăng dần `[0.0, 0.3, 0.6, 0.9]` để tạo sự đa dạng. Kết quả cuối là **majority vote** của `n` câu trả lời.

```
hypothesis 0 (temp=0.0, greedy) ──► answer_0 ─┐
hypothesis 1 (temp=0.3)         ──► answer_1 ─┤──► majority_vote() ──► final answer
hypothesis 2 (temp=0.6)         ──► answer_2 ─┘
```

### Fallback

Nếu tất cả hypothesis fail (execution error + verification fail), method tự động fallback về zero-shot direct generation dùng `build_messages()` và `gen_kwargs_for()` từ các module hiện tại — không cần LLM call thêm.

### Cấu hình

```yaml
# configs/symbolic_cot_bigbench_date.yaml
method: symbolic_cot
n_hypotheses: 1          # tăng lên 3 để bật multi-hypothesis voting (chậm hơn 3x)
max_correction_attempts: 1  # 0 = không có self-correction loop
```

---

## Mở rộng

### Thêm phương pháp mới (ví dụ Dynamic Few-shot)

1. Tạo `src/methods/dynamic_few_shot.py` với class implement `predict(sample) -> str`.
2. Đăng ký trong [src/methods/registry.py](src/methods/registry.py):

   ```python
   def build_dynamic_few_shot(model, **kwargs): return DynamicFewShotMethod(model, **kwargs)
   METHOD_BUILDERS["dynamic_few_shot"] = build_dynamic_few_shot
   ```

3. Tạo YAML config `configs/dynamic_few_shot_<dataset>.yaml` với `method: dynamic_few_shot`.
4. Chạy: `python -m src.runner --config configs/dynamic_few_shot_<dataset>.yaml`.

> `FewShotMethod` đã nhận `shot_selector: Callable[[Sample], Sequence[Sample]]` — Dynamic Few-shot chỉ cần thay `fixed_shots(...)` bằng selector dùng TF-IDF / embedding retrieval.

### Thêm dataset mới

1. Viết loader trong `src/data/<new_dataset>.py` trả về `list[Sample]` đúng schema.
2. Đăng ký trong [src/data/registry.py](src/data/registry.py) (`DATASET_LOADERS`, `DEFAULT_PATHS`, `DEFAULT_MAX_SAMPLES`).
3. Nếu task/ngôn ngữ chưa có: bổ sung `PromptTemplate` trong [src/prompts/templates.py](src/prompts/templates.py) và `extract/normalize_gold` trong [src/evaluation/extractor.py](src/evaluation/extractor.py).

---

## Reproducibility

- Seed Python + NumPy + PyTorch + transformers được fix trong mỗi run qua [src/utils/seed.py](src/utils/seed.py) (default `seed=42`).
- Eval hyperparams: `temperature=0`, `do_sample=False`, `max_new_tokens` nhỏ theo task (duration: 8, date_arith: 24). Xem [src/methods/base.py](src/methods/base.py).
- **Symbolic CoT**: các LLM call nội bộ (program synthesis, correction) dùng `temperature=0.0, do_sample=False` cho hypothesis đầu tiên; hypothesis `k > 0` dùng sampling với temperature cố định `[0.3, 0.6, 0.9]`.
- `metrics.json` snapshot toàn bộ config (bao gồm `n_hypotheses`, `max_correction_attempts`) cho mỗi run.

---

## Testing

Unit tests cho loader / extractor / metrics / prompts / evaluate / executor. Chạy:

```bash
python -m pytest tests/ -q
```

Test integration cho 4 loader sẽ **skip tự động** nếu `Dataset/Raw/` không tồn tại — an toàn cho CI không có dataset.

---

## Licens and Datasets

- **UDST-DurationQA** — [Original repo](https://github.com/UBC-NLP/UDST) (license theo tác giả).
- **BigBench DateUnderstanding** — [BIG-bench](https://github.com/google/BIG-bench) (Apache-2.0).
- **VLSP 2025 ViTempQA** — VLSP 2025 shared task (sử dụng theo điều khoản ban tổ chức).

Vui lòng trích nguồn khi công bố kết quả.
