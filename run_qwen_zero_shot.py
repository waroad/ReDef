from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
import os
import json
import difflib
import random
from datasets import Dataset, load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from tqdm import tqdm
from datetime import datetime
import multiprocessing
import gc
from datasets import concatenate_datasets
import glob
import shutil
import sys
import argparse

def get_qwen_text(code_content, label, encoding_type):
    answer = "Yes" if label == 1 else "No"
    desc_map = {
        # Stage 1 & Perturbation
        'After-only': "The provided code is the version AFTER modification.",
        
        'After+Markers': "The code is the version AFTER modification. Lines marked with <CHG> indicate where changes occurred.",
        'Spurious_change_markers': "The code is the version AFTER modification. Lines marked with <CHG> indicate where changes occurred.",
        
        'Before+After': "The code includes both [BEFORE] and [AFTER] snapshots of the modification.",
        'Swapped_snapshots': "The code includes both [BEFORE] and [AFTER] snapshots of the modification.",
        
        'Diff_with_tags': "The code is a diff representation. <DEL> indicates deleted lines and <ADD> indicates added lines.",
        'Reversed_diff_tags': "The code is a diff representation. <DEL> indicates deleted lines and <ADD> indicates added lines.",
        
        'Added_to_Deleted': "The code consists specifically of the actual added and deleted lines, organized under [ADDED LINES] and [DELETED LINES] headers.",
        'Swapped_added/deleted_blocks': "The code consists specifically of the actual added and deleted lines, organized under [ADDED LINES] and [DELETED LINES] headers."
    }
    type_desc = desc_map.get(encoding_type, "") 
    messages = [
        {"role": "system", 
            "content": f"You are a software engineering expert.{type_desc} Determine if the code modification is defective. "
                       "You MUST respond with only 'Yes' or 'No' without any explanation." 
        },
        {"role": "user", 
            "content": f"Is the following code modification defective?\n\nCode:\n{code_content}\n\nAnswer strictly with 'Yes' or 'No'."
        },
        {"role": "assistant", "content": answer}
    ]
    return messages

def process_encoding(args, tokenizer, example, encoding_type, max_seq_length, is_training=True):
    code_before = example['function_before']
    code_after = example['function_after']
    label = example['defective_modification']
    
    before_lines = code_before.splitlines()
    after_lines = code_after.splitlines()
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines)
    
    final_code_text = ""
    code_limit = max_seq_length - 100 # 100 for prompts
    half_limit = code_limit // 2
    # 1. After-only
    if encoding_type == 'After-only':
        final_code_text = code_after

    # 2. After+Markers
    elif encoding_type == 'After+Markers':
        modified_lines = []
        for i, line in enumerate(after_lines):
            is_modified = any(tag in ['replace', 'insert'] and j1 <= i < j2 for tag, i1, i2, j1, j2 in matcher.get_opcodes())
            modified_lines.append(f"<CHG> {line}" if is_modified else line)
        final_code_text = '\n'.join(modified_lines)

    # 3. Before+After
    elif encoding_type == "Before+After":
        before_tokens = tokenizer.encode(code_before, add_special_tokens=False)[:half_limit]
        after_tokens = tokenizer.encode(code_after, add_special_tokens=False)[:half_limit]
        
        final_code_text = f"[BEFORE]\n{tokenizer.decode(before_tokens)}\n[AFTER]\n{tokenizer.decode(after_tokens)}"

    # 4. Diff_with_tags
    elif encoding_type == "Diff_with_tags":
        diff_lines = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ['delete', 'replace']:
                for line in before_lines[i1:i2]: diff_lines.append('<DEL> ' + line)
            if tag in ['insert', 'replace']:
                for line in after_lines[j1:j2]: diff_lines.append('<ADD> ' + line)
        final_code_text = '\n'.join(diff_lines)

    # 5. Added_to_Deleted
    elif encoding_type == "Added_to_Deleted":
        added = [line for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag in ['insert', 'replace'] for line in after_lines[j1:j2]]
        deleted = [line for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag in ['delete', 'replace'] for line in before_lines[i1:i2]]
        
        added_text = ' '.join(added) if added else ''
        deleted_text = ' '.join(deleted) if deleted else ''
        
        final_code_text = ""
        if added_text:
            final_code_text += f"[ADDED LINES]\n{added_text}\n"
        if deleted_text:
            final_code_text += f"[DELETED LINES]\n{deleted_text}"
        
    # 6. Spurious_change_markers (Baselines)
    elif encoding_type == "Spurious_change_markers":
        chg_count = sum(1 for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag in ['replace', 'insert'])
        if len(after_lines) > 0 and chg_count > 0:
            random_indices = random.sample(range(len(after_lines)), min(chg_count, len(after_lines)))
            modified_lines = [f"<CHG> {line}" if i in random_indices else line for i, line in enumerate(after_lines)]
            final_code_text = '\n'.join(modified_lines)
        else:
            final_code_text = code_after

    # 7. Swapped_snapshots
    elif encoding_type == "Swapped_snapshots":
        before_tokens = tokenizer.encode(code_before, add_special_tokens=False)[:half_limit]
        after_tokens = tokenizer.encode(code_after, add_special_tokens=False)[:half_limit]
        
        b_text = tokenizer.decode(before_tokens)
        a_text = tokenizer.decode(after_tokens)

        if not is_training and random.random() > 0.5:
            final_code_text = f"[BEFORE]\n{a_text}\n[AFTER]\n{b_text}"
        else:
            final_code_text = f"[BEFORE]\n{b_text}\n[AFTER]\n{a_text}"

    # 8. Reversed_diff_tags
    elif encoding_type == "Reversed_diff_tags":
        diff_lines = []
        do_reverse = not is_training 
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            del_header = '<ADD>' if do_reverse else '<DEL>'
            add_header = '<DEL>' if do_reverse else '<ADD>'
            
            if tag in ['delete', 'replace']:
                for line in before_lines[i1:i2]: diff_lines.append(f'{del_header} ' + line)
            if tag in ['insert', 'replace']:
                for line in after_lines[j1:j2]: diff_lines.append(f'{add_header} ' + line)
        final_code_text = '\n'.join(diff_lines)

    # 9. Swapped_added/deleted_blocks
    elif encoding_type == "Swapped_added/deleted_blocks":
        added_text = ' '.join([line for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag in ['insert', 'replace'] for line in after_lines[j1:j2]])
        deleted_text = ' '.join([line for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag in ['delete', 'replace'] for line in before_lines[i1:i2]])
        
        if random.random() > 0.5:
            final_code_text = f"[ADDED LINES]\n{deleted_text}\n[DELETED LINES]\n{added_text}"
        else:
            final_code_text = f"[ADDED LINES]\n{added_text}\n[DELETED LINES]\n{deleted_text}"


    code_tokens = tokenizer.encode(final_code_text, add_special_tokens=False)

    if len(code_tokens) > code_limit:
        code_tokens = code_tokens[:code_limit]
        final_code_text = tokenizer.decode(code_tokens)


    return {"messages": get_qwen_text(final_code_text, example['defective_modification'], encoding_type)}

def load_and_prepare_dataset(args, tokenizer, encoding_type, max_seq_length, oversampling=False):
    data_files = {
        "train": args.train_data_file,
        "validation": args.valid_data_file,
        "test": args.test_data_file
    }
    dataset = load_dataset("json", data_files=data_files)

    dataset["train"] = dataset["train"].map(
        lambda x: process_encoding(args, tokenizer, x, encoding_type, max_seq_length, is_training=True), 
        batched=False
    )
    
    dataset["validation"] = dataset["validation"].map(
        lambda x: process_encoding(args, tokenizer, x, encoding_type, max_seq_length, is_training=False), 
        batched=False
    )
    dataset["test"] = dataset["test"].map(
        lambda x: process_encoding(args, tokenizer, x, encoding_type, max_seq_length, is_training=False), 
        batched=False
    )


    if oversampling:
        train_ds = dataset["train"]
        
        pos_ds = train_ds.filter(lambda x: x['defective_modification'] == 1)
        neg_ds = train_ds.filter(lambda x: x['defective_modification'] == 0)
        
        n_pos = len(pos_ds)
        n_neg = len(neg_ds)
        
        print(f"\n⚖️ [Data Balancing] Original - Positive: {n_pos}, Negative: {n_neg}")

        pos_3x = concatenate_datasets([pos_ds] * 3)
        gap = n_neg - len(pos_3x)
        random_extra = pos_ds.shuffle(seed=42).select(range(gap))
        final_pos_ds = concatenate_datasets([pos_3x, random_extra])
        
        dataset["train"] = concatenate_datasets([neg_ds, final_pos_ds]).shuffle(seed=42)
        
        final_n_pos = len(dataset["train"].filter(lambda x: x['defective_modification'] == 1))
        final_n_neg = len(dataset["train"].filter(lambda x: x['defective_modification'] == 0))
        print(f"✅ [Data Balancing] 1:1 Complete - Positive: {final_n_pos}, Negative: {final_n_neg}")



    def apply_template(examples):
        texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in examples["messages"]]
        return {"text": texts}

    dataset = dataset.map(apply_template, batched=True)
    return dataset



def parse_args():
    parser = argparse.ArgumentParser(description="FSE Research: JIT-SDP with CodeLLM")

    parser.add_argument("--train_data_file", required=True, type=str, help="Training data (.jsonl)")
    parser.add_argument("--valid_data_file", required=True, type=str, help="Validation data (.jsonl)")
    parser.add_argument("--test_data_file", required=True, type=str, help="Test data (.jsonl)")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--encoding_type", type=str, required=True, help="""Encoding strategy
        'After-only', 'After+Markers', 'Before+After', 
        'Diff_with_tags', 'Added_to_Deleted', 'Spurious_change_markers', 
        'Swapped_snapshots', 'Reversed_diff_tags', 'Swapped_added/deleted_blocks'""")
    parser.add_argument("--max_seq_length", type=int, default=612, help="Sequence limit")


    return parser.parse_args()


def main():
    args = parse_args()
    import psutil
    import builtins
    builtins.psutil = psutil
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    import psutil
    print(f"\n" + "="*80)
    print(f"🔍 Zero-shot Begin: Encoding={args.encoding_type}, Limit={args.max_seq_length}")
    print("="*80)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length = args.max_seq_length,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model) 

    results_true = []
    results_pred = []
    results_probs = [] 

    dataset = load_and_prepare_dataset(args, tokenizer, args.encoding_type, args.max_seq_length, oversampling=False)

    test_data = dataset["test"] 

    for i in tqdm(range(len(test_data))):
        true_answer = test_data[i]["messages"][-1]["content"].strip().lower()
        true_label = 1 if "yes" in true_answer else 0
        results_true.append(true_label)
        
        prompt_messages = test_data[i]["messages"][:-1]
        inputs = tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize = True, 
            add_generation_prompt = True, 
            return_tensors = "pt",
            truncation = True,           
            max_length = args.max_seq_length   
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(input_ids = inputs, max_new_tokens = 5) 
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            full_text = decoded[0]
            prediction_text = full_text.split("assistant")[-1].strip().lower()
            
            if prediction_text.startswith("yes"):
                is_yes = 1
            elif "yes" in prediction_text and "no" not in prediction_text:
                is_yes = 1
            else:
                is_yes = 0
            
            results_pred.append(is_yes)
            results_probs.append(is_yes)

        if i % 100 == 0:
            print(f"\n[Example {i}]"+"-------------------------------------")
            print(f" - Ground Truth: {'Yes' if true_label == 1 else 'No'}")
            print(f" - Model Output: {prediction_text} ->{'Yes' if is_yes == 1 else 'No'} ")

    accuracy = accuracy_score(results_true, results_pred)
    precision = precision_score(results_true, results_pred)
    recall = recall_score(results_true, results_pred)
    f1 = f1_score(results_true, results_pred)

    print(f"\n[Test Results]")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("fse_zero_shot_results.txt", "a") as f:
        f.write(f"[{current_time}] Type: {args.encoding_type}, Limit: {args.max_seq_length}, Seed: {args.seed} "
                f"Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")


if __name__ == "__main__":
    main()

