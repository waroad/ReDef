# ReDef: Do Code Language Models Truly Understand Code Changes for Just-in-Time Software Defect Prediction?

[![FSE 2026](https://img.shields.io/badge/FSE-2026-blue.svg)](https://doi.org/10.1145/3808179)
[![DOI](https://img.shields.io/badge/Figshare-10.6084%2Fm9.figshare.30086968-blue.svg)](https://doi.org/10.6084/m9.figshare.30086968)
[![Docker Hub](https://img.shields.io/badge/Docker-waroad1%2Fredef%3Afse2026-informational.svg?logo=docker)](https://hub.docker.com/r/waroad1/redef)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green.svg?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official replication package and codebase for the FSE 2026 paper: **"ReDef: Do Code Language Models Truly Understand Code Changes for Just-in-Time Software Defect Prediction?"**

---

## 📌 Overview

**ReDef** (Revert-based Defect dataset) is a high-confidence benchmark and evaluation framework designed to assess whether Code Language Models (CLMs) genuinely understand code changes for Just-in-Time (JIT) defect prediction.

* **High-Quality Curated Dataset**: Consists of **3,164 defective** and **10,268 clean** function-level modifications curated across **22 large-scale C/C++ projects** (~107MB).
* **Robust Label Filtering**: Defective modifications are anchored to developer revert commits, with ambiguous cases conservatively filtered using a GPT-4o-assisted voting and triage process.
* **Comprehensive Benchmark Framework**: Supports fine-tuning and evaluating Encoder-based models (**CodeBERT, CodeT5+, UniXcoder**) and Decoder-based models (**Qwen2.5**) across **5 encoding strategies** and **4 counterfactual perturbation tests**.
* **Extensible Data Pipeline**: Provides tools to curate custom ReDef datasets from any target git repository.

---

## 🚀 Quick Start

### Option A: Docker Environment (Recommended)
All dependencies, datasets, and models are pre-installed in the Docker image.

```bash
docker run --gpus all -it --rm waroad1/redef:fse2026
```

### Option B: Manual Installation (Local Environment)

```bash
# 1. Install system dependency
sudo apt update && sudo apt install -y universal-ctags

# 2. Clone repository & install Python packages
git clone https://github.com/waroad/ReDef.git
cd ReDef
pip install -r requirements.txt
```

---

## 🧪 Experiments & Reproduction

> **Docker Tip**: To run any of the commands below inside Docker, prepend `docker run --gpus all -it --rm waroad1/redef:fse2026`.

### 1. Encoder-based Models (CodeBERT, CodeT5+, UniXcoder)

#### Training & Evaluation (CodeBERT Example)
```bash
python run_base.py \
  --output_dir=./codebert_After-only_512 \
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
  --eval_batch_size=8 \
  --epoch=6 \
  --class_weight=3.24496 \
  --encoding_type=After-only \
  --do_train \
  --do_test
```

* **Available Encoding Strategies (`--encoding_type`)**:
  * Standard: `After-only`, `After+Markers`, `Before+After`, `Diff_with_tags`
  * Counterfactual Perturbations: `Added_to_Deleted`, `Spurious_change_markers`, `Swapped_snapshots`, `Reversed_diff_tags`, `Swapped_added_deleted_blocks`
* **Model Switch**:
  * **CodeT5+**: `--model_type=codet5 --model_name_or_path=Salesforce/codet5p-220m --tokenizer_name=Salesforce/codet5p-220m`
  * **UniXcoder**: Provide the local model path to `--model_name_or_path` and `--tokenizer_name`.

#### Instant Verification (Pre-saved Checkpoints)
Pre-saved checkpoints for all 9 encoding strategies are available in the Docker container:
```bash
docker run --gpus all -it --rm waroad1/redef:fse2026 python run_base.py \
  --output_dir=./codebert_Diff_with_tags_512 \
  --model_type roberta \
  --model_name_or_path=/app/models/codebert \
  --tokenizer_name=/app/models/codebert \
  --train_data_file=1_train.jsonl \
  --eval_data_file=1_valid.jsonl \
  --test_data_file=1_test.jsonl \
  --block_size=512 \
  --seed=12345 \
  --class_weight=3.24496 \
  --encoding_type=Diff_with_tags \
  --do_test
```

---

### 2. Decoder-based Models (Qwen2.5)

#### Fine-Tuning & Evaluation
```bash
python run_qwen.py \
  --train_data_file=1_train.jsonl \
  --valid_data_file=1_valid.jsonl \
  --test_data_file=1_test.jsonl \
  --encoding_type=After-only \
  --seed=12345 \
  --max_seq_length=612 \
  --do_train \
  --do_eval \
  --do_test
```

#### Zero-Shot Testing
```bash
python run_qwen_zero_shot.py \
  --train_data_file=1_train.jsonl \
  --valid_data_file=1_valid.jsonl \
  --test_data_file=1_test.jsonl \
  --encoding_type=Diff_with_tags \
  --seed=12345 \
  --max_seq_length=612
```

---

## 🛠️ Custom Data Collection Pipeline

[cite_start]To curate a custom ReDef dataset from any target repository[cite: 13]:

1. **Clone Target Repository**:
   ```bash
   git clone [https://github.com/](https://github.com/)<owner>/<repo>.git
   ```

2. **Extract Defective & Clean Modifications**:
   ```bash
   OPENAI_API_KEY="sk-xxxx" python collect_modifications.py <repo_name>
   ```

3. **Generate Final Dataset Splits**:
   ```bash
   python merge_modifications.py
   ```

> **Quick Verification on Postgres**:
> ```bash
> docker run --rm -it -e OPENAI_API_KEY="sk-xxxx" waroad1/redef:fse2026 python collect_modifications.py postgres
> ```

---

## 📁 Repository Structure

```text
├── dataset/                  # Curated ReDef benchmark data (train/valid/test splits)
├── collect_modifications.py  # Repository extraction & GPT-4o triage pipeline
├── merge_modifications.py    # Multi-project merging & split generator
├── run_base.py               # Training & evaluation script for Encoder models
├── run_qwen.py               # Fine-tuning & evaluation script for Qwen2.5
├── run_qwen_zero_shot.py     # Zero-shot evaluation script for Qwen2.5
├── requirements.txt          # Python library dependencies
└── README.md
```

---

## 📖 Citation

```bibtex
@article{nam2026redef,
  author    = {Nam, Doha and Kim, Taehyoun and Ryu, Duksan and Baik, Jongmoon},
  title     = {ReDef: Do Code Language Models Truly Understand Code Changes for Just-in-Time Software Defect Prediction?},
  journal   = {Proceedings of the ACM on Software Engineering},
  volume    = {3},
  number    = {FSE},
  articleno = {FSE172},
  year      = {2026},
  doi       = {10.1145/3808179},
  publisher = {Association for Computing Machinery}
}
```
