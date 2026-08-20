# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json

import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
import difflib
cpu_cont = multiprocessing.cpu_count()
from torch.optim import AdamW  
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer, BertForSequenceClassification,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertForSequenceClassification,
                          DistilBertTokenizer, T5Config, T5EncoderModel)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    'codet5': (T5Config, T5EncoderModel, RobertaTokenizer)
}


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)

    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))[0]
        outputs = self.dropout(outputs)

        logits = outputs
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            weight = labels * self.args.class_weight + (1 - labels) * 1.0
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = loss * weight
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


class Model2(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model2, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids=None, labels=None):
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state  # (B, L, H)
        last_hidden_state = self.dropout(last_hidden_state)

        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask  # (B, H)

        logits = self.classifier(pooled)  # (B, 1)
        prob = torch.sigmoid(logits)

        if labels is not None:
            labels = labels.float().unsqueeze(1)  # (B, 1)
            weight = labels * self.args.class_weight + (1 - labels) * 1.0

            loss = -weight * (labels * torch.log(prob + 1e-10) + (1 - labels) * torch.log(1 - prob + 1e-10))
            loss = loss.mean()
            return loss, prob
        else:
            return prob


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label


def convert_examples_to_features(js, tokenizer, args):
    # source
    code_before = js['function_before']
    code_after = js['function_after']
    label = js['defective_modification']
    before_lines = code_before.splitlines()
    after_lines = code_after.splitlines()
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines)

    if args.encoding_type=='After-only':
        code_tokens = tokenizer.tokenize(code_after)[:args.block_size - 2]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        return InputFeatures(source_tokens, source_ids, label)
    elif args.encoding_type=='After+Markers':
        modified_lines = []

        for i, line in enumerate(after_lines):
            is_modified = False
            # Check if this line is part of any modification
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag in ['replace', 'insert'] and j1 <= i < j2:
                    is_modified = True
                    break
                elif tag == 'delete' and i == j1 and j1 < len(after_lines):
                    is_modified = True
                    break
            if is_modified:
                modified_lines.append(f"<CHG> {line}")
            else:
                modified_lines.append(line)

        # Join the modified lines back into code
        modified_code_after = '\n'.join(modified_lines)

        # Tokenize the modified after code
        code_tokens = tokenizer.tokenize(modified_code_after)[:args.block_size - 2]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length

        return InputFeatures(source_tokens, source_ids, label)
    elif args.encoding_type=="Before+After":
        code_tokens = tokenizer.tokenize(code_before)[:args.block_size // 2 - 2]
        code_tokens2 = tokenizer.tokenize(code_after)[:args.block_size // 2 - 2]

        source_tokens = [tokenizer.cls_token] + ['[BEFORE]'] + code_tokens + ['[AFTER]'] + code_tokens2 + [
            tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length

        return InputFeatures(source_tokens, source_ids, label)
    elif args.encoding_type=="Diff_with_tags":
        code_lines = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'delete':
                # Process deleted lines
                for line in before_lines[i1:i2]:
                    code_lines.append('<DEL> ' + line)

            elif tag == 'insert':
                # Process added lines
                for line in after_lines[j1:j2]:
                    code_lines.append('<ADD> ' + line)

            elif tag == 'replace':
                # Process replaced lines (delete old, add new)
                for line in before_lines[i1:i2]:
                    code_lines.append('<DEL> ' + line)
                for line in after_lines[j1:j2]:
                    code_lines.append('<ADD> ' + line)
        full_diff_text = '\n'.join(code_lines)
        # Tokenize with PLM tokenizer
        code_tokens = tokenizer.tokenize(full_diff_text)[:args.block_size - 2]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length

        return InputFeatures(source_tokens, source_ids, label)
    elif args.encoding_type=="Added_to_Deleted":
        added_lines = []
        deleted_lines = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'delete':
                deleted_lines.extend(before_lines[i1:i2])
            elif tag == 'insert':
                added_lines.extend(after_lines[j1:j2])
            elif tag == 'replace':
                deleted_lines.extend(before_lines[i1:i2])
                added_lines.extend(after_lines[j1:j2])

        added_text = ' '.join(added_lines) if added_lines else ''
        deleted_text = ' '.join(deleted_lines) if deleted_lines else ''

        full_text_parts = []

        # Added lines
        if added_text:
            full_text_parts.append('[ADDED LINES]')  # Special token
            added_tokens = tokenizer.tokenize(added_text)
            full_text_parts.extend(added_tokens)
        # Deleted lines
        if deleted_text:
            full_text_parts.append('[DELETED LINES]')  # Special token
            deleted_tokens = tokenizer.tokenize(deleted_text)
            full_text_parts.extend(deleted_tokens)

        if len(full_text_parts) > args.block_size - 2:
            full_text_parts = full_text_parts[:args.block_size - 2]

        source_tokens = [tokenizer.cls_token] + full_text_parts + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length

        return InputFeatures(source_tokens, source_ids, label)
    elif args.encoding_type=="Spurious_change_markers":
        modified_lines = []

        for i, line in enumerate(after_lines):
            is_modified = False

            # Check if this line is part of any modification
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag in ['replace', 'insert'] and j1 <= i < j2:
                    is_modified = True
                    break
                elif tag == 'delete' and i == j1 and j1 < len(after_lines):
                    is_modified = True
                    break
            if is_modified:
                modified_lines.append(f"<CHG> {line}")
            else:
                modified_lines.append(line)

        original_modified_code = '\n'.join(modified_lines)
        chg_count = original_modified_code.count("<CHG>")

        clean_lines = [line for line in after_lines]

        if len(clean_lines) > 0 and chg_count > 0:
            actual_chg_count = min(chg_count, len(clean_lines))
            random_indices = random.sample(range(len(clean_lines)), actual_chg_count)

            counterfactual_lines = []
            for i, line in enumerate(clean_lines):
                if i in random_indices:
                    counterfactual_lines.append(f"<CHG> {line}")
                else:
                    counterfactual_lines.append(line)

            modified_lines = counterfactual_lines
        else:
            modified_lines = clean_lines

        modified_code_after = '\n'.join(modified_lines)

        # Tokenize the modified after code
        code_tokens = tokenizer.tokenize(modified_code_after)[:args.block_size - 2]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length

        return InputFeatures(source_tokens, source_ids, label)
    elif args.encoding_type=="Swapped_snapshots":
        if args.is_training==False and random.random()>0.5:
            code_before = js['function_after']
            code_after = js['function_before']
        code_tokens = tokenizer.tokenize(code_before)[:args.block_size // 2 - 2]
        code_tokens2 = tokenizer.tokenize(code_after)[:args.block_size // 2 - 2]

        source_tokens = [tokenizer.cls_token] + ['[BEFORE]'] + code_tokens + ['[AFTER]'] + code_tokens2 + [
            tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length

        return InputFeatures(source_tokens, source_ids, label)
    elif args.encoding_type=="Reversed_diff_tags":
        code_lines = []
        if args.is_training==True:
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'delete':
                    # Process deleted lines
                    for line in before_lines[i1:i2]:
                        code_lines.append('<DEL> ' + line)

                elif tag == 'insert':
                    # Process added lines
                    for line in after_lines[j1:j2]:
                        code_lines.append('<ADD> ' + line)

                elif tag == 'replace':
                    # Process replaced lines (delete old, add new)
                    for line in before_lines[i1:i2]:
                        code_lines.append('<DEL> ' + line)
                    for line in after_lines[j1:j2]:
                        code_lines.append('<ADD> ' + line)
        else:
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'delete':
                    # Process deleted lines
                    for line in before_lines[i1:i2]:
                        code_lines.append('<ADD> ' + line)

                elif tag == 'insert':
                    # Process added lines
                    for line in after_lines[j1:j2]:
                        code_lines.append('<DEL> ' + line)

                elif tag == 'replace':
                    # Process replaced lines (delete old, add new)
                    for line in before_lines[i1:i2]:
                        code_lines.append('<ADD> ' + line)
                    for line in after_lines[j1:j2]:
                        code_lines.append('<DEL> ' + line)

        full_diff_text = '\n'.join(code_lines)
        # Tokenize with PLM tokenizer
        code_tokens = tokenizer.tokenize(full_diff_text)[:args.block_size - 2]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length

        return InputFeatures(source_tokens, source_ids, label)
    elif args.encoding_type=="Swapped_added_deleted_blocks":
        added_lines = []
        deleted_lines = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'delete':
                deleted_lines.extend(before_lines[i1:i2])
            elif tag == 'insert':
                added_lines.extend(after_lines[j1:j2])
            elif tag == 'replace':
                deleted_lines.extend(before_lines[i1:i2])
                added_lines.extend(after_lines[j1:j2])

        added_text = ' '.join(added_lines) if added_lines else ''
        deleted_text = ' '.join(deleted_lines) if deleted_lines else ''

        full_text_parts = []

        if random.random() > 0.5:
            temp = added_text
            added_text = deleted_text
            deleted_text = temp
        if added_text:
            full_text_parts.append('[ADD]')  # Special token
            added_tokens = tokenizer.tokenize(added_text)
            full_text_parts.extend(added_tokens)

        if deleted_text:
            full_text_parts.append('[DEL]')  # Special token
            deleted_tokens = tokenizer.tokenize(deleted_text)
            full_text_parts.extend(deleted_tokens)

        if len(full_text_parts) > args.block_size - 2:
            full_text_parts = full_text_parts[:args.block_size - 2]

        source_tokens = [tokenizer.cls_token] + full_text_parts + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length

        return InputFeatures(source_tokens, source_ids, label)
    else:
        print("improper encoding type")
        exit()


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        if "train" in file_path:
            args.is_training=True
        else:
            args.is_training=False
        with open(file_path, encoding='utf-8') as f:
            cnt = 0
            for line in f:
                cnt += 1
                # if cnt==100:
                #     break
                js = json.loads(line.strip())
                temp = convert_examples_to_features(js, tokenizer, args)
                if temp:
                    self.examples.append(temp)

            print("Length of data: ", len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_acc = 0.0
    best_f1 = 0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    # Initialize early stopping parameters at the start of training
    early_stopping_counter = 0
    best_loss = None

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            loss, logits = model(inputs, labels)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, eval_when_training=True)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, value)
                            # Save model checkpoint

                    if results['eval_acc'] > best_acc:
                        best_acc = results['eval_acc']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best acc:%s", round(best_acc, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('pytorch_model.bin'))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

                    if results['eval_f1'] > best_f1:
                        best_f1 = results['eval_f1']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('pytorch_model.bin'))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

        # Calculate average loss for the epoch
        avg_loss = train_loss / tr_num

        # Check for early stopping condition
        if args.early_stopping_patience is not None:
            if best_loss is None or avg_loss < best_loss - args.min_loss_delta:
                best_loss = avg_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.early_stopping_patience:
                    logger.info("Early stopping")
                    break  # Exit the loop early


def evaluate(args, model, tokenizer, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4,
                                 pin_memory=True)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 0] > 0.5
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    preds_binary = preds.astype(int)
    labels_binary = labels.astype(int)

    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    accuracy = accuracy_score(labels_binary, preds_binary)
    precision = precision_score(labels_binary, preds_binary)
    recall = recall_score(labels_binary, preds_binary)
    f1 = f1_score(labels_binary, preds_binary)
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    result = {
        "eval_loss": 0.0,
        "eval_acc": round(eval_acc, 4),
        "eval_f1": round(f1, 4),
    }
    return result


def test(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []

    input_lengths = []

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        batch_lengths = (inputs != tokenizer.pad_token_id).sum(dim=1).cpu().numpy()
        input_lengths.append(batch_lengths)

        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    input_lengths = np.concatenate(input_lengths, 0)
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 0] > 0.5
    prob_scores = logits[:, 0]

    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    preds_binary = preds.astype(int)
    labels_binary = labels.astype(int)
    accuracy = accuracy_score(labels_binary, preds_binary)
    precision = precision_score(labels_binary, preds_binary)
    recall = recall_score(labels_binary, preds_binary)
    f1 = f1_score(labels_binary, preds_binary)
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(labels_binary, prob_scores)
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f} PR-AUC: {ap:.4f}")
    with open(f'result_PLM.txt', 'a') as f:  # 'a' mode appends to existing content
        f.write(
            f"{args.model_name_or_path} {args.encoding_type} {args.test_data_file} acc: {accuracy} f1: {f1}, precision: {precision}, recall: {recall}, PR-AUC: {ap:.4f} \n")

def main(seed1=-1):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", required=True, type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", required=True, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", required=True, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run val on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run val on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true', default=True,
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--GPU", default=0, type=int,
                        help="..")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=8,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    # Add early stopping parameters and dropout probability parameters
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                        help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("--min_loss_delta", type=float, default=0.001,
                        help="Minimum change in the loss required to qualify as an improvement.")
    parser.add_argument('--dropout_probability', type=float, default=0, help='dropout probability')
    parser.add_argument('--class_weight', type=float, default=3.24496,
                        help="class weight for imbalance")
    parser.add_argument('--encoding_type', type=str, required=True, help="Choose any of below: "
                                                                         "'After-only', 'After+Markers', 'Before+After', "
                                                                         "'Diff_with_tags', 'Added_to_Deleted', "
                                                                         "'Spurious_change_markers', 'Swapped_snapshots', "
                                                                         "'Reversed_diff_tags', 'Swapped_added_deleted_blocks'")

    args = parser.parse_args()
    if seed1 != -1:
        args.seed+=seed1
        args.output_dir+="_"+str(seed1)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        # Set the GPU(s) explicitly
        if isinstance(args.GPU, list):  # If multiple GPUs are specified
            selected_gpus = ",".join(map(str, args.GPU))
            torch.cuda.set_device(args.GPU[0])  # Set the first GPU as the primary device
            device = torch.device(f"cuda:{args.GPU[0]}")
            args.n_gpu = len(args.GPU)
        else:  # Single GPU case
            torch.cuda.set_device(args.GPU)
            device = torch.device(f"cuda:{args.GPU}")
            args.n_gpu = 1
    else:  # Distributed training
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size // args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    with open(f'result_PLM.txt', 'a') as f:  # 'a' mode appends to existing content
        f.write(f"-------------------------{datetime.now()}------------------------- \n")

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    config.num_labels = 1
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    if args.model_name_or_path:
        base_model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool('.ckpt' in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None
        )
    else:
        base_model = model_class(config)

    if args.encoding_type in ["After+Markers", "Spurious_change_markers"]:
        special_tokens_dict = {
            'additional_special_tokens': [
                '<CHG>'
            ]
        }
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_toks} special tokens")
        base_model.resize_token_embeddings(len(tokenizer))
    elif args.encoding_type in ["Before+After","Swapped_snapshots"]:
        special_tokens_dict = {
            'additional_special_tokens': [
                '[BEFORE]','[AFTER]'
            ]
        }
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_toks} special tokens")
        base_model.resize_token_embeddings(len(tokenizer))
    elif args.encoding_type in ["Diff_with_tags","Reversed_diff_tags"]:
        special_tokens_dict = {
            'additional_special_tokens': [
                '<ADD>','<DEL>'
            ]
        }
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_toks} special tokens")
        base_model.resize_token_embeddings(len(tokenizer))
    elif args.encoding_type in ["Added_to_Deleted", "Swapped_added_deleted_blocks"]:
        special_tokens_dict = {
            'additional_special_tokens': [
                '[ADDED LINES]', '[DELETED LINES]'
            ]
        }
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_toks} special tokens")
        base_model.resize_token_embeddings(len(tokenizer))

    if args.model_type=="codet5":
        model = Model2(base_model, tokenizer=tokenizer, config=config, args=args)
    else:
        model = Model(base_model, tokenizer=tokenizer, config=config, args=args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        if args.local_rank == 0:
            torch.distributed.barrier()

        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-f1/pytorch_model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-f1/pytorch_model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, tokenizer)

    return results


if __name__ == "__main__":
    main()
    # main(1)
    # main(2)
    # main(3)
    # main(4)
    # main(5)
    # main(7)
    # main(8)
    # main(9)
