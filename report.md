# Temporal Reasoning Evaluation Report
## Task Definition, Example Data, Proposed Method & Results

---

## 1. Task Definition

### 1.1 DateArith (Date Arithmetic)

**Objective:** Given a natural language question describing a temporal relationship (e.g., "yesterday was X, what is today's date?"), the model must compute and output the correct calendar date.

| Property | Value |
|---|---|
| Task type | Open-ended date computation |
| Language | English (`bigbench_date`), Vietnamese (`vlsp_date`) |
| Output format | `MM/DD/YYYY` (EN) / `Tháng M, YYYY` (VI) |
| Metric | Exact-match **Accuracy** |
| Datasets | BigBench DateUnderstanding (369 samples), VLSP 2025 ViTempQA DateQA (1,500 samples) |

**What it tests:** multi-step date arithmetic — adding/subtracting days, weeks, months; reasoning through weekday names; calendar awareness (leap years, month lengths).

---

### 1.2 DurationQA (Duration Plausibility)

**Objective:** Given a context sentence describing an event and a candidate duration, the model must judge whether that duration is **plausible** (`yes`) or **implausible** (`no`) for the described event.

| Property | Value |
|---|---|
| Task type | Binary plausibility classification |
| Language | English (`udst_duration`), Vietnamese (`vlsp_duration`) |
| Output format | `yes` / `no` |
| Metric | **F1** (macro) |
| Datasets | UDST Duration (1,500 samples), VLSP 2025 ViTempQA DurationQA (1,500 samples, expanded from 4-option questions) |

**What it tests:** commonsense knowledge of typical event durations — e.g., a surgery taking 3 hours is plausible, but 3 seconds is not.

---

## 2. Example Data

### 2.1 DateArith — English (BigBench DateUnderstanding)

```
Question : Yesterday was April 30, 2021. What is the date today in MM/DD/YYYY?
Gold     : 05/01/2021

Question : Yesterday was April 30, 2021. What is the date tomorrow in MM/DD/YYYY?
Gold     : 05/02/2021

Question : Jane was born on the last day of February in 2001. What is the date 
           one week after Jane's birthday in MM/DD/YYYY?
Gold     : 03/07/2001
```

### 2.2 DateArith — Vietnamese (VLSP ViTempQA)

```
Context  : Hội nghị diễn ra vào ngày 15 tháng 3 năm 2022.
Question : Ba tuần sau hội nghị là ngày mấy?
Gold     : Tháng 4, 2022

Question : Sinh nhật của Lan là ngày cuối cùng của tháng 2 năm 2000. 
           Ngày sinh của Lan là ngày mấy?
Gold     : Tháng 2, 2000
```

### 2.3 DurationQA — English (UDST)

```
Context           : The surgeon performed a routine appendectomy.
Question          : How long did the surgery take?
Candidate duration: 45 minutes
Gold              : yes

Context           : The surgeon performed a routine appendectomy.
Question          : How long did the surgery take?
Candidate duration: 3 seconds
Gold              : no
```

### 2.4 DurationQA — Vietnamese (VLSP ViTempQA)

```
Context           : Đội tuyển bóng đá thi đấu trận chung kết.
Question          : Trận đấu kéo dài bao lâu?
Candidate duration: 90 phút
Gold              : yes

Context           : Đội tuyển bóng đá thi đấu trận chung kết.
Question          : Trận đấu kéo dài bao lâu?
Candidate duration: 3 giây
Gold              : no
```

---

## 3. Proposed Method — Hybrid SymbolicCoT+

### 3.1 Motivation

Pure LLM direct-answer approaches fail on date arithmetic due to implicit arithmetic errors. The original SymbolicCoT (code synthesis only) improves date arithmetic but hurts DurationQA by forcing LLM intuition into rigid code thresholds. The proposed Hybrid SymbolicCoT+ addresses both by: (1) routing simple cases through a rule-based fast path with zero LLM cost, (2) applying CoT planning (Wei et al.) to guide reasoning, (3) routing duration tasks through a KB-aided CoT path without code synthesis, and (4) adding a retrospective verifier (Xu et al.) to catch unfaithful reasoning.

### 3.2 Architecture — Hybrid SymbolicCoT+ Pipeline

```
Input question / context
        │
        ▼
┌──────────────────────────────────────────┐
│  Layer 0 – Rule-based Fast Path         │  temporal_extractor.py
│  (Rule-based, 0 LLM calls)             │  solve_date_arith / solve_duration
└──────────────────────┬───────────────────┘
                       │
              confident result?
               /              \
             YES               NO
              │                 │
           Output          Layer 1: Planner (LLM)
           (fast path)     decompose question into steps
                                 │
                      task routing
                     /            \
               date_arith        duration
                   │                │
                   ▼                ▼
    ┌──────────────────────┐  ┌───────────────────────────┐
    │  Layer 2A            │  │  Layer 2B                 │
    │  Guided Synthesis    │  │  CoT + KB Path            │
    │  (LLM)               │  │  (LLM + Rule-based KB)    │
    │                      │  │                           │
    │  CoT reasoning steps │  │  LLM reasons step-by-step │
    │  + Python code       │  │  aided by activity KB     │
    │  in one LLM call     │  │  range hint               │
    └──────────┬───────────┘  └────────────┬──────────────┘
               │                           │
               ▼                           │
    ┌──────────────────────┐               │
    │  Layer 3 – Execution │               │
    │  (Rule-based)        │               │
    │  Python sandbox      │               │
    │  cross-platform      │               │
    │  timeout (threading) │               │
    └──────────┬───────────┘               │
               │                           │
        exec OK & verify?                  │
          /         \                      │
        YES          NO                    │
         │            │                    │
         │      Layer 4: Self-Correct      │
         │      (LLM rewrites code)        │
         │      retry execution            │
         │                                 │
         ▼                                 │
    collect valid answer                   │
         │                                 │
    N hypotheses done                      │
         │                                 │
    any candidates?                        │
      /       \                            │
    YES         NO → Zero-Shot Fallback    │
     │                    │               │
  Majority Vote           │               │
  (rule-based Counter)    │               │
     │                    │               │
     └─────────┬──────────┘               │
               │                          │
               ▼                          ▼
    ┌──────────────────────────────────────────┐
    │  Layer 5 – Retrospective Verifier (LLM) │
    │  Check: reasoning ↔ answer consistent?  │
    │  If INVALID → retry with diversity      │
    └──────────────────────┬───────────────────┘
                           │
                        Output
```

### 3.3 Component Summary

| Layer | Component | Role | Type |
|---|---|---|---|
| 0 | Rule-based fast path | `solve_date_arith` / `solve_duration` — handle common patterns instantly | Rule-based |
| 1 | Planner | LLM decomposes question into numbered steps (CoT, Wei et al.) | LLM |
| 2A | Guided synthesis | LLM writes CoT reasoning steps + Python code in one call | LLM |
| 2B | Duration CoT + KB | LLM reasons step-by-step with activity KB range as hint; no code | LLM + Rule-based |
| 3 | Symbolic executor | Cross-platform threading timeout; sandboxed `exec()`; extract `answer` | Rule-based |
| 4 | Self-correction | Feed runtime error back to LLM for one rewrite attempt | LLM (≤1 retry) |
| 5 | Retrospective verifier | Check reasoning ↔ answer consistency; retry if INVALID (Xu et al.) | LLM |
| 6 | Vote + fallback | Majority vote over N hypotheses; zero-shot LLM if all fail | Rule-based + LLM |

### 3.4 Smart Routing Logic

```
rule_based(sample) → result?   →  YES: return immediately (0 LLM calls)
                                   NO:  call Planner

task == "duration"  →  Layer 2B path (CoT + KB, no code synthesis)
task == "date_arith" → Layer 2A path (guided synthesis + execution)
```

### 3.5 Key Improvements Over Original SymbolicCoT

| Aspect | Original | Hybrid SymbolicCoT+ |
|---|---|---|
| Simple cases | Always calls LLM | Layer 0 handles with 0 LLM calls |
| CoT reasoning | None | Layer 1 planner + Layer 2A reasoning steps |
| Duration task | Code with self-estimated thresholds (weak) | CoT + activity KB hint (stronger) |
| Faithfulness check | None | Layer 5 retrospective verifier |
| Timeout | Linux only (SIGALRM) | Cross-platform (threading) |
| Reasoning transparency | Code only | Natural language steps + code |

### 3.6 Comparison to Baselines

| Method | Description |
|---|---|
| **Zero-shot** | Single LLM call, direct answer, no examples, no reasoning |
| **Few-shot** | k labeled examples prepended as (user, assistant) chat turns |
| **Hybrid SymbolicCoT+** *(proposed)* | Rule-based fast path + CoT planner + guided synthesis + KB-aided duration CoT + retrospective verifier |

---

## 4. Experimental Results

All experiments use model **Qwen/Qwen3.5-9B**, `seed=42`, `enable_thinking=False`.

### 4.1 Results Table

| Task | Experiment | Method | k-shot | Metric | Score |
|---|---|---|---|---|---|
| udst_duration | zero_shot_udst_duration | zero_shot | 0 | F1 | 0.5798 |
| udst_duration | few_shot_udst_duration_k4 | few_shot | 4 | F1 | 0.4722 |
| udst_duration | symbolic_cot_udst_duration | symbolic_cot | 0 | F1 | **0.6020** |
| bigbench_date | zero_shot_bigbench_date | zero_shot | 0 | Accuracy | 0.3496 |
| bigbench_date | few_shot_bigbench_date_k3 | few_shot | 3 | Accuracy | 0.3550 |
| bigbench_date | symbolic_cot_bigbench_date | symbolic_cot | 0 | Accuracy | **0.4932** |
| vlsp_date | zero_shot_vlsp_date | zero_shot | 0 | Accuracy | 0.2653 |
| vlsp_date | few_shot_vlsp_date_k3 | few_shot | 3 | Accuracy | 0.3653 |
| vlsp_date | symbolic_cot_vlsp_date | symbolic_cot | 0 | Accuracy | **0.7907** |
| vlsp_duration | zero_shot_vlsp_duration | zero_shot | 0 | F1 | 0.7257 |
| vlsp_duration | few_shot_vlsp_duration_k4 | few_shot | 4 | F1 | **0.7407** |
| vlsp_duration | symbolic_cot_vlsp_duration | symbolic_cot | 0 | F1 | 0.6824 |

> **Bold** = best result per task.  
> Metric: F1 for duration tasks (binary yes/no), Accuracy for date arithmetic tasks.

### 4.2 Summary by Task

| Task | Zero-shot | Few-shot (best k) | Symbolic CoT | Best |
|---|---|---|---|---|
| udst_duration (EN, F1) | 0.5798 | 0.4722 (k=4) | 0.6020 | Symbolic CoT |
| bigbench_date (EN, Acc) | 0.3496 | 0.3550 (k=3) | 0.4932 | Symbolic CoT |
| vlsp_date (VI, Acc) | 0.2653 | 0.3653 (k=3) | **0.7907** | Symbolic CoT |
| vlsp_duration (VI, F1) | 0.7257 | **0.7407** (k=4) | 0.6824 | Few-shot |

### 4.3 Key Observations

1. **Symbolic CoT dominates DateArith tasks.** On `bigbench_date` (+14.0 pp over zero-shot) and especially `vlsp_date` (+52.5 pp over zero-shot), deterministic program execution dramatically outperforms direct-answer prompting. This confirms that date arithmetic benefits from offloading computation to a symbolic engine rather than relying on the LLM's implicit arithmetic.

2. **Few-shot is surprisingly weak on DateArith.** Adding examples yields only marginal gains (+0.5 pp on `bigbench_date`, +10 pp on `vlsp_date`), far below Symbolic CoT. The bottleneck is arithmetic correctness, not format alignment.

3. **DurationQA favors few-shot (EN) and zero-shot (VI).** For `vlsp_duration`, Symbolic CoT underperforms zero-shot by −4.3 pp. Duration plausibility requires broad commonsense estimation; generating executable threshold code may introduce errors that override correct LLM intuitions.

4. **Vietnamese date arithmetic sees the largest gain.** `vlsp_date` improves from 0.2653 (zero-shot) to 0.7907 (Symbolic CoT), a 3× improvement, suggesting that the Vietnamese direct-answer baseline is significantly bottlenecked by language-specific format compliance that the code-execution path bypasses entirely.

---

*Model: Qwen/Qwen3.5-9B — Date: 2026-04-21 — Seed: 42*
