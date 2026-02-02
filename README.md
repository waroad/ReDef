# ReDef: High-Confidence JIT-SDP Benchmark



**ReDef (Revert-based Defect dataset)** is a high-confidence benchmark of function-level modifications curated from **22 large-scale C/C++ projects**. By leveraging revert commits as explicit causal anchors and filtering them through a GPT-assisted triage process, ReDef provides a reliable ground-truth corpus consisting of **3,164 defective** and **10,268 clean** modifications.

---

## 🚀 Getting Started

### Prerequisites
* **Python**: 3.10


### Available Encoding Types
ReDef supports **9 different input encodings** to evaluate how Code Language Models (CLMs) reason about code modifications:
* `After-only`
* `After+Markers`
* `Before+After`
* `Diff_with_tags`
* `Added_to_Deleted`
* `Spurious_change_markers` (Perturbation)
* `Swapped_snapshots` (Perturbation)
* `Reversed_diff_tags` (Perturbation)
* `Swapped_added/deleted_blocks` (Perturbation)

---

## 🛠️ Training & Evaluation

### 1. Encoder-based Models
The benchmark evaluates established encoder models: **CodeBERT** (125M), **CodeT5+** (220M), and **UniXcoder** (220M).

**Example: Training CodeBERT (After-only)**
```bash
python run_base.py \
  --output_dir=./base \
  --model_type roberta \
  --model_name_or_path=microsoft/codebert-base \
  --tokenizer_name=microsoft/codebert-base \
  --train_data_file=1_train.jsonl \
  --eval_data_file=1_valid.jsonl \
  --test_data_file=1_test.jsonl \
  --block_size=512 \
  --seed=12345 \
  --learning_rate=1e-5 \
  --train_batch_size=8 \
  --encoding_type=After-only \
  --do_train --do_test
```

> **Note on UniXcoder-nine**: Manually download the model and provide the dedicated local path in `--model_name_or_path` and `--tokenizer_name` (Set `--model_type roberta`).

---

### 2. Decoder-based Models
The study incorporates **Qwen2.5-7B-Instruct** to evaluate state-of-the-art large-scale decoder capabilities.

**Qwen2.5 Fine-Tuning (512 code + 100 prompt tokens)**
```bash
python run_qwen.py \
  --train_data_file=1_train.jsonl \
  --valid_data_file=1_valid.jsonl \
  --test_data_file=1_test.jsonl \
  --encoding_type=After-only \
  --seed=12345 \
  --max_seq_length=612 \
  --do_train --do_eval --do_test
```

**Qwen2.5 Zero-Shot Evaluation**
```bash
python run_qwen_zero_shot.py \
  --train_data_file=1_train.jsonl \
  --valid_data_file=1_valid.jsonl \
  --test_data_file=1_test.jsonl \
  --encoding_type=After-only \
  --seed=12345 \
  --max_seq_length=612
```

---

## 📂 Custom Dataset Collection

To collect a custom ReDef dataset from any target repository, follow these steps:

1. **Clone the target repository** (e.g., Linux) into the working directory.
2. **Install universal-ctags**:
   ```bash
   sudo apt install universal-ctags
   ```
3. **Extract modifications**:
   ```bash
   python collect_modifications.py
   ```
4. **Generate final dataset** (Train/Val/Test splits):
   ```bash
   python merge_modifications.py
   ```

---
