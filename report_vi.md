# Báo Cáo Đánh Giá Suy Luận Thời Gian
## Định Nghĩa Bài Toán, Dữ Liệu Mẫu, Phương Pháp Đề Xuất & Kết Quả

---

## 1. Định Nghĩa Bài Toán

### 1.1 DateArith (Tính Toán Ngày Tháng)

**Mục tiêu:** Cho một câu hỏi ngôn ngữ tự nhiên mô tả quan hệ thời gian (ví dụ: "hôm qua là ngày X, hôm nay là ngày mấy?"), mô hình phải tính toán và trả về đúng ngày trong lịch.

| Thuộc tính | Giá trị |
|---|---|
| Loại bài toán | Tính toán ngày tháng dạng mở |
| Ngôn ngữ | Tiếng Anh (`bigbench_date`), Tiếng Việt (`vlsp_date`) |
| Định dạng đầu ra | `MM/DD/YYYY` (EN) / `Tháng M, YYYY` (VI) |
| Metric | Exact-match **Accuracy** |
| Tập dữ liệu | BigBench DateUnderstanding (369 mẫu), VLSP 2025 ViTempQA DateQA (1.500 mẫu) |

**Kiểm tra khả năng:** tính toán ngày tháng nhiều bước — cộng/trừ ngày, tuần, tháng; suy luận qua tên thứ trong tuần; nhận thức lịch (năm nhuận, số ngày trong tháng).

---

### 1.2 DurationQA (Đánh Giá Tính Hợp Lý Khoảng Thời Gian)

**Mục tiêu:** Cho một câu văn mô tả sự kiện và một khoảng thời gian đề xuất, mô hình phải đánh giá khoảng thời gian đó có **hợp lý** (`yes`) hay **không hợp lý** (`no`) với sự kiện được mô tả.

| Thuộc tính | Giá trị |
|---|---|
| Loại bài toán | Phân loại nhị phân tính hợp lý |
| Ngôn ngữ | Tiếng Anh (`udst_duration`), Tiếng Việt (`vlsp_duration`) |
| Định dạng đầu ra | `yes` / `no` |
| Metric | **F1** (macro) |
| Tập dữ liệu | UDST Duration (1.500 mẫu), VLSP 2025 ViTempQA DurationQA (1.500 mẫu, mở rộng từ câu hỏi 4 lựa chọn) |

**Kiểm tra khả năng:** kiến thức thường thức về thời lượng điển hình của các sự kiện — ví dụ: phẫu thuật kéo dài 3 tiếng là hợp lý, nhưng 3 giây thì không.

---

## 2. Dữ Liệu Mẫu

### 2.1 DateArith — Tiếng Anh (BigBench DateUnderstanding)

```
Câu hỏi : Yesterday was April 30, 2021. What is the date today in MM/DD/YYYY?
Đáp án  : 05/01/2021

Câu hỏi : Yesterday was April 30, 2021. What is the date tomorrow in MM/DD/YYYY?
Đáp án  : 05/02/2021

Câu hỏi : Jane was born on the last day of February in 2001. What is the date 
           one week after Jane's birthday in MM/DD/YYYY?
Đáp án  : 03/07/2001
```

### 2.2 DateArith — Tiếng Việt (VLSP ViTempQA)

```
Ngữ cảnh : Hội nghị diễn ra vào ngày 15 tháng 3 năm 2022.
Câu hỏi  : Ba tuần sau hội nghị là ngày mấy?
Đáp án   : Tháng 4, 2022

Câu hỏi  : Sinh nhật của Lan là ngày cuối cùng của tháng 2 năm 2000.
            Ngày sinh của Lan là ngày mấy?
Đáp án   : Tháng 2, 2000
```

### 2.3 DurationQA — Tiếng Anh (UDST)

```
Ngữ cảnh          : The surgeon performed a routine appendectomy.
Câu hỏi           : How long did the surgery take?
Khoảng thời gian  : 45 minutes
Đáp án            : yes

Ngữ cảnh          : The surgeon performed a routine appendectomy.
Câu hỏi           : How long did the surgery take?
Khoảng thời gian  : 3 seconds
Đáp án            : no
```

### 2.4 DurationQA — Tiếng Việt (VLSP ViTempQA)

```
Ngữ cảnh          : Đội tuyển bóng đá thi đấu trận chung kết.
Câu hỏi           : Trận đấu kéo dài bao lâu?
Khoảng thời gian  : 90 phút
Đáp án            : yes

Ngữ cảnh          : Đội tuyển bóng đá thi đấu trận chung kết.
Câu hỏi           : Trận đấu kéo dài bao lâu?
Khoảng thời gian  : 3 giây
Đáp án            : no
```

---

## 3. Phương Pháp Đề Xuất — Hybrid SymbolicCoT+

### 3.1 Động Lực

Các phương pháp LLM trả lời trực tiếp thất bại do lỗi tính toán ngầm định. SymbolicCoT gốc (chỉ sinh code) cải thiện bài toán ngày tháng nhưng làm giảm DurationQA vì ép trực giác LLM vào ngưỡng số cứng. Hybrid SymbolicCoT+ giải quyết cả hai bằng cách: (1) route các trường hợp đơn giản qua fast path rule-based không tốn LLM, (2) áp dụng CoT planning (Wei et al.) để dẫn dắt suy luận, (3) route DurationQA qua CoT kết hợp KB không sinh code, và (4) thêm retrospective verifier (Xu et al.) để phát hiện suy luận không trung thực.

### 3.2 Kiến Trúc — Pipeline Hybrid SymbolicCoT+

```
Câu hỏi đầu vào / ngữ cảnh
        │
        ▼
┌──────────────────────────────────────────┐
│  Layer 0 – Rule-based Fast Path         │  temporal_extractor.py
│  (Rule-based, 0 lần gọi LLM)           │  solve_date_arith / solve_duration
└──────────────────────┬───────────────────┘
                       │
              có kết quả tự tin?
               /              \
              CÓ               KHÔNG
              │                 │
           Đầu ra          Layer 1: Planner (LLM)
           (fast path)     chia nhỏ câu hỏi thành các bước
                                 │
                      routing theo task
                     /            \
               date_arith        duration
                   │                │
                   ▼                ▼
    ┌──────────────────────┐  ┌───────────────────────────┐
    │  Layer 2A            │  │  Layer 2B                 │
    │  Guided Synthesis    │  │  CoT + KB Path            │
    │  (LLM)               │  │  (LLM + Rule-based KB)    │
    │                      │  │                           │
    │  Bước suy luận CoT   │  │  LLM suy luận từng bước  │
    │  + Python code       │  │  với gợi ý range từ KB    │
    │  trong 1 lần gọi LLM │  │  hoạt động               │
    └──────────┬───────────┘  └────────────┬──────────────┘
               │                           │
               ▼                           │
    ┌──────────────────────┐               │
    │  Layer 3 – Thực thi  │               │
    │  (Rule-based)        │               │
    │  Python sandbox      │               │
    │  timeout đa nền tảng │               │
    │  (threading)         │               │
    └──────────┬───────────┘               │
               │                           │
        thực thi OK & verify?              │
          /         \                      │
        CÓ          KHÔNG                  │
         │            │                    │
         │      Layer 4: Tự sửa lỗi       │
         │      (LLM viết lại code)        │
         │      thử lại thực thi           │
         │                                 │
         ▼                                 │
    thu thập answer hợp lệ                 │
         │                                 │
    N hypotheses xong                      │
         │                                 │
    có candidate không?                    │
      /       \                            │
    CÓ         KHÔNG → Zero-Shot Fallback  │
     │                    │               │
  Majority Vote           │               │
  (rule-based Counter)    │               │
     │                    │               │
     └─────────┬──────────┘               │
               │                          │
               ▼                          ▼
    ┌──────────────────────────────────────────┐
    │  Layer 5 – Retrospective Verifier (LLM) │
    │  Kiểm tra: suy luận ↔ answer nhất quán? │
    │  Nếu INVALID → thử lại với diversity    │
    └──────────────────────┬───────────────────┘
                           │
                        Đầu ra
```

### 3.3 Tóm Tắt Các Thành Phần

| Layer | Thành phần | Vai trò | Loại |
|---|---|---|---|
| 0 | Rule-based fast path | `solve_date_arith` / `solve_duration` — xử lý pattern đơn giản ngay lập tức | Rule-based |
| 1 | Planner | LLM chia câu hỏi thành các bước có đánh số (CoT, Wei et al.) | LLM |
| 2A | Guided synthesis | LLM viết bước suy luận CoT + Python code trong một lần gọi | LLM |
| 2B | Duration CoT + KB | LLM suy luận từng bước với gợi ý range từ activity KB; không sinh code | LLM + Rule-based |
| 3 | Symbolic executor | Timeout đa nền tảng (threading); `exec()` sandbox; trích biến `answer` | Rule-based |
| 4 | Tự sửa lỗi | Gửi lỗi runtime cho LLM để viết lại một lần | LLM (≤1 lần) |
| 5 | Retrospective verifier | Kiểm tra tính nhất quán suy luận ↔ answer; thử lại nếu INVALID (Xu et al.) | LLM |
| 6 | Vote + fallback | Majority vote trên N hypotheses; zero-shot LLM nếu tất cả thất bại | Rule-based + LLM |

### 3.4 Logic Routing Thông Minh

```
rule_based(sample) → có kết quả?  →  CÓ: return ngay (0 lần gọi LLM)
                                       KHÔNG: gọi Planner

task == "duration"   →  Layer 2B (CoT + KB, không sinh code)
task == "date_arith" →  Layer 2A (guided synthesis + thực thi)
```

### 3.5 Cải Tiến So Với SymbolicCoT Gốc

| Khía cạnh | Gốc | Hybrid SymbolicCoT+ |
|---|---|---|
| Trường hợp đơn giản | Luôn gọi LLM | Layer 0 xử lý với 0 lần gọi LLM |
| Suy luận CoT | Không có | Layer 1 planner + bước suy luận Layer 2A |
| DurationQA | Code với ngưỡng LLM tự ước (yếu) | CoT + gợi ý KB hoạt động (mạnh hơn) |
| Kiểm tra tính trung thực | Không có | Layer 5 retrospective verifier |
| Timeout | Chỉ Linux (SIGALRM) | Đa nền tảng (threading) |
| Tính minh bạch suy luận | Chỉ có code | Bước ngôn ngữ tự nhiên + code |

### 3.6 So Sánh Với Các Baseline

| Phương pháp | Mô tả |
|---|---|
| **Zero-shot** | Một lần gọi LLM, trả lời trực tiếp, không có ví dụ, không có bước suy luận |
| **Few-shot** | k ví dụ có nhãn được thêm vào trước dưới dạng các lượt hội thoại (user, assistant) |
| **Hybrid SymbolicCoT+** *(đề xuất)* | Rule-based fast path + CoT planner + guided synthesis + CoT KB cho duration + retrospective verifier |

---

## 4. Kết Quả Thực Nghiệm

Tất cả thực nghiệm dùng model **Qwen/Qwen3.5-9B**, `seed=42`, `enable_thinking=False`.

### 4.1 Bảng Kết Quả

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

> **In đậm** = kết quả tốt nhất theo từng task.  
> Metric: F1 cho bài toán duration (phân loại yes/no), Accuracy cho bài toán tính toán ngày tháng.

### 4.2 Tổng Hợp Theo Task

| Task | Zero-shot | Few-shot (k tốt nhất) | Symbolic CoT | Tốt nhất |
|---|---|---|---|---|
| udst_duration (EN, F1) | 0.5798 | 0.4722 (k=4) | 0.6020 | Symbolic CoT |
| bigbench_date (EN, Acc) | 0.3496 | 0.3550 (k=3) | 0.4932 | Symbolic CoT |
| vlsp_date (VI, Acc) | 0.2653 | 0.3653 (k=3) | **0.7907** | Symbolic CoT |
| vlsp_duration (VI, F1) | 0.7257 | **0.7407** (k=4) | 0.6824 | Few-shot |

### 4.3 Nhận Xét Chính

1. **Symbolic CoT vượt trội trên các bài toán DateArith.** Với `bigbench_date` (+14.0 điểm % so với zero-shot) và đặc biệt `vlsp_date` (+52.5 điểm % so với zero-shot), thực thi chương trình tất định vượt xa phương pháp trả lời trực tiếp. Điều này xác nhận rằng bài toán tính ngày tháng được hưởng lợi từ việc chuyển phép tính sang symbolic engine thay vì dựa vào phép tính ngầm định của LLM.

2. **Few-shot yếu đáng ngạc nhiên trên DateArith.** Thêm ví dụ chỉ cải thiện nhỏ (+0.5 điểm % trên `bigbench_date`, +10 điểm % trên `vlsp_date`), thấp hơn nhiều so với Symbolic CoT. Điểm nghẽn cổ chai là độ chính xác tính toán, không phải căn chỉnh định dạng.

3. **DurationQA ưu tiên few-shot (EN) và zero-shot (VI).** Với `vlsp_duration`, Symbolic CoT kém hơn zero-shot 4.3 điểm %. Đánh giá tính hợp lý khoảng thời gian đòi hỏi ước lượng thường thức rộng; việc sinh code ngưỡng có thể gây lỗi và ghi đè trực giác đúng của LLM.

4. **Bài toán ngày tháng tiếng Việt có cải thiện lớn nhất.** `vlsp_date` tăng từ 0.2653 (zero-shot) lên 0.7907 (Symbolic CoT), cải thiện gấp 3 lần, cho thấy baseline tiếng Việt trả lời trực tiếp bị giới hạn đáng kể bởi việc tuân thủ định dạng ngôn ngữ cụ thể — điều mà hướng thực thi code bỏ qua hoàn toàn.

---

*Model: Qwen/Qwen3.5-9B — Ngày chạy: 2026-04-21 — Seed: 42*
