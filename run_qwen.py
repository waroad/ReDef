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
import gc
from datasets import concatenate_datasets
import glob
import shutil
import sys
import argparse

def get_qwen_text(code_content, label, encoding_type):
    answer = "Yes" if label == 1 else "No"
    desc_map = {
        # Stage 1 & Pertubation
        'After-only': "The provided code is the version AFTER modification.",
        
        'After+Markers': "The code is the version AFTER modification. Lines marked with <CHG> indicate where changes occurred.",
        'Spurious_change_markers': "The code is the version AFTER modification. Lines marked with <CHG> indicate where changes occurred.",
        
        'Before+After': "The code includes both [BEFORE] and [AFTER] snapshots of the modification.",
        'Swapped_snapshots': "The code includes both [BEFORE] and [AFTER] snapshots of the modification.",
        
        'Diff_with_tags': "The code is a diff representation. <DEL> indicates deleted lines and <ADD> indicates added lines.",
        'Reversed_diff_tags': "The code is a diff representation. <DEL> indicates deleted lines and <ADD> indicates added lines.",
        
        'Added_to_Deleted': "The code consists specifically of the actual added and deleted lines, organized under [ADDED LINES] and [DELETED LINES] headers.",
        'Swapped_added_deleted_blocks': "The code consists specifically of the actual added and deleted lines, organized under [ADDED LINES] and [DELETED LINES] headers."
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
    code_limit = max_seq_length - 100 
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

    # 9. Swapped_added_deleted_blocks
    elif encoding_type == "Swapped_added_deleted_blocks":
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
        
        print(f"\n⚖️ [Data Balancing] Original- Positive: {n_pos}, Negative: {n_neg}")

        pos_3x = concatenate_datasets([pos_ds] * 3)
        gap = n_neg - len(pos_3x)
        random_extra = pos_ds.shuffle(seed=42).select(range(gap))
        final_pos_ds = concatenate_datasets([pos_3x, random_extra])
        
        dataset["train"] = concatenate_datasets([neg_ds, final_pos_ds]).shuffle(seed=42)
        
        final_n_pos = len(dataset["train"].filter(lambda x: x['defective_modification'] == 1))
        final_n_neg = len(dataset["train"].filter(lambda x: x['defective_modification'] == 0))
        print(f"✅ [Data Balancing] 1:1 completed - Positive: {final_n_pos}, Negative: {final_n_neg}")



    def apply_template(examples):
        texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in examples["messages"]]
        return {"text": texts}

    dataset = dataset.map(apply_template, batched=True)
    return dataset



class F1BestModelCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, output_dir, max_seq_length):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.best_f1 = -1.0
        self.max_seq_length = max_seq_length

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(f"\n[Step {state.global_step}] F1 Score validation...")
        model = kwargs['model']
        FastLanguageModel.for_inference(model)
        
        preds, labels = [], []
        
        print("\n" + "="*30 + " Example(top 10) " + "="*30)
        
        for i in range(len(self.eval_dataset)):
            true_label = 1 if "yes" in self.eval_dataset[i]["messages"][-1]["content"].lower() else 0
            labels.append(true_label)
            
            prompt = self.tokenizer.apply_chat_template(
                self.eval_dataset[i]["messages"][:-1], 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length
            ).to("cuda")
            
            with torch.no_grad():
                out = model.generate(input_ids=prompt, max_new_tokens=50)
                decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
                ans = decoded.split("assistant")[-1].strip().lower()
                
                is_yes = 1 if (ans.startswith("yes") or ("yes" in ans and "no" not in ans)) else 0
                preds.append(is_yes)

                if i < 5:
                    gt_str = "DEFECTIVE (Yes)" if true_label == 1 else "CLEAN (No)"
                    pred_str = "DEFECTIVE (Yes)" if is_yes == 1 else "CLEAN (No)"
                    
                    display_ans = ans.replace('\n', ' ') 
                    print(f"[{i+1}] Ground Truth: {gt_str}")
                    print(f"    Model Output: {display_ans} -> {pred_str}")
                    print("-" * 50)
        
        print("="*80 + "\n")

        current_f1 = f1_score(labels, preds)
        print(f"Current F1: {current_f1:.4f} (Previous best: {self.best_f1:.4f})")
        
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            save_path = os.path.join(self.output_dir, "best_f1_checkpoint")
            model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"🏆 New_Best F1! checkpoint saved: {save_path}")
        
        FastLanguageModel.for_training(model)


def parse_args():
    parser = argparse.ArgumentParser(description="FSE Research: JIT-SDP with CodeLLM")


    parser.add_argument("--train_data_file", required=True, type=str, help="Training data (.jsonl)")
    parser.add_argument("--valid_data_file", required=True, type=str, help="Validation data (.jsonl)")
    parser.add_argument("--test_data_file", required=True, type=str, help="Test data (.jsonl)")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run val on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run val on the dev set.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--encoding_type", type=str, required=True, help="""Encoding strategy
        'After-only', 'After+Markers', 'Before+After', 
        'Diff_with_tags', 'Added_to_Deleted', 'Spurious_change_markers', 
        'Swapped_snapshots', 'Reversed_diff_tags', 'Swapped_added_deleted_blocks'""")
    parser.add_argument("--max_seq_length", type=int, default=612, help="Sequence limit")
    

    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--used_percentage", type=float, default=1.0)

    return parser.parse_args()

def main():
    args = parse_args()
    import psutil
    import builtins
    builtins.psutil = psutil
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"🚀 Begin: Encoding={args.encoding_type}, Limit={args.max_seq_length}")

    current_output_dir = f"Qwen_{args.encoding_type}_{args.max_seq_length}_{args.seed}"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length = args.max_seq_length,
        load_in_4bit = True,
    )

    dataset = load_and_prepare_dataset(args, tokenizer, args.encoding_type, args.max_seq_length, oversampling=True)
    num_train_samples = len(dataset["train"])
    subset_size = int(num_train_samples * args.used_percentage) 
    dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(subset_size))
    num_val_samples = len(dataset["validation"])
    val_subset_size = int(num_val_samples * args.used_percentage)
    dataset["validation"] = dataset["validation"].shuffle(seed=args.seed).select(range(val_subset_size))

    if args.do_train:

        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
        )

        total_steps = len(dataset["train"]) // 16
        eval_interval = max(1, total_steps // 10) 
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset["train"],
            eval_dataset = dataset["validation"],
            dataset_text_field = "text",
            max_seq_length = args.max_seq_length,
            args = TrainingArguments(
                per_device_train_batch_size = args.batch_size,
                gradient_accumulation_steps = args.grad_accum,
                num_train_epochs = args.epochs,
                learning_rate = args.learning_rate,
                seed = args.seed,
                eval_strategy = "no",  
                save_strategy = "steps",    
                save_steps = eval_interval, 
                save_total_limit = 20,     
                logging_steps = 10,
                optim = "adamw_8bit",
                gradient_checkpointing = True, 
                output_dir = current_output_dir,
            ),
        )

        if args.encoding_type=="Swapped_snapshots":
            current_output_dir = f"Qwen_Before+After_{args.max_seq_length}_{args.seed}"
        elif args.encoding_type=="Reversed_diff_tags":
            current_output_dir = f"Qwen_Diff_with_tags_{args.max_seq_length}_{args.seed}"
        else: 
            print(f"🚀 From total {num_train_samples}, {len(dataset['train'])}are used for quick training")
            trainer.train()
               
    import gc
    gc.collect()
    torch.cuda.empty_cache()
 
    if args.do_eval and args.encoding_type not in ["Swapped_snapshots", "Reversed_diff_tags"]:
        print("\n" + "="*50)
        print("🧐 Looking for best f1")
        print("="*50)
        checkpoint_dirs = sorted(glob.glob(os.path.join(current_output_dir, "checkpoint-*")), 
                    key=lambda x: int(x.split("-")[-1]))

        best_f1 = -1.0
        best_checkpoint = ""

        for ckpt in checkpoint_dirs:
            print(f"\n[Evaluating] {ckpt}")
            model = FastLanguageModel.for_inference(model) 
            model.load_adapter(ckpt, adapter_name="default")

            
            preds, labels = [], []
            for i in tqdm(range(len(dataset["validation"])), 
                      desc=f"🔍 Eval {os.path.basename(ckpt)}", 
                      unit="sample",
                      leave=True):
                true_label = 1 if "yes" in dataset["validation"][i]["messages"][-1]["content"].lower() else 0
                labels.append(true_label)
                
                prompt = tokenizer.apply_chat_template(
                    dataset["validation"][i]["messages"][:-1], 
                    tokenize=True, add_generation_prompt=True, return_tensors="pt",
                    truncation=True, 
                    max_length=args.max_seq_length 
                ).to("cuda")
                
                with torch.no_grad():
                    out = model.generate(input_ids=prompt, max_new_tokens=5)
                    ans = tokenizer.batch_decode(out, skip_special_tokens=True)[0].split("assistant")[-1].strip().lower()
                    is_yes = 1 if (ans.startswith("yes") or ("yes" in ans and "no" not in ans)) else 0
                    preds.append(is_yes)
            current_f1 = f1_score(labels, preds)
            accuracy = accuracy_score(labels, preds)
            print(f"📌 {os.path.basename(ckpt)} F1: {current_f1:.4f}, Acc: {accuracy:.4f}")
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_checkpoint = ckpt
            
            gc.collect()
            torch.cuda.empty_cache()
            # if hasattr(model, "peft_config") and "default" in model.peft_config:
            #     model.delete_adapter("default")

        print(f"\n🏆 Final Best Checkpoint: {best_checkpoint} (F1: {best_f1:.4f})")

        best_model_final_path = os.path.join(current_output_dir, "best_f1_checkpoint")
        if best_checkpoint:
            if os.path.exists(best_model_final_path):
                shutil.rmtree(best_model_final_path)
            shutil.copytree(best_checkpoint, best_model_final_path)
            print(f"Best model saved:'{best_model_final_path}'")

        for ckpt in checkpoint_dirs:
            try:
                shutil.rmtree(ckpt)
                print(f"Deletion complete: {os.path.basename(ckpt)}")
            except Exception as e:
                print(f"⚠️ {ckpt} Error deleting: {e}")



    if args.do_test:
        if args.encoding_type=="Swapped_snapshots":
            current_output_dir = f"Qwen_Before+After_{args.max_seq_length}_{args.seed}"
        elif args.encoding_type=="Reversed_diff_tags":
            current_output_dir = f"Qwen_Diff_with_tags_{args.max_seq_length}_{args.seed}"
        print("\n" + "="*50)
        print("Begin Dataset Testing")
        print("="*50)
        best_model_path = os.path.join(current_output_dir, "best_f1_checkpoint")

        model.load_adapter(best_model_path, adapter_name="default")
        FastLanguageModel.for_inference(model) 

        results_true = []
        results_pred = []
        results_probs = [] 

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

        with open("result_qwen.txt", "a") as f:
            f.write(f"[{current_time}] Type: {args.encoding_type}, Limit: {args.max_seq_length}, Seed: {args.seed} "
                    f"Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")



if __name__ == "__main__":
    main()
