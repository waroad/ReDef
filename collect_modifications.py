#!/usr/bin/env python3
import subprocess
import re
import os
import sys
import json
import openai
import tempfile
from openai import OpenAI
from typing import List, Dict
import time
import json
import random
import subprocess
from datetime import datetime, timedelta
api_key = 'put appropriate key here'


def run_git_command(cmd, cwd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                            cwd=cwd, encoding='utf-8', errors='ignore')
    return result.stdout.strip().split('\n') if result.stdout else []


def extract_reverted_commit(message):
    """Extract commit hash from revert message"""

    patterns = [
        r'[Rr]evert\s+"?commit\s+([0-9a-f]{6,40})',
        r'[Rr]evert\s+"?([0-9a-f]{6,40})\s+commit',
        r'[Rr]evert\s+commit\s+([0-9a-f]{6,40})',
        r'[Rr]evert\s+([0-9a-f]{6,40})\b',
        r'[Rr]evert\s+"?(?:commit\s+)?([0-9a-f]{6,40})',
        r'[Rr]everts?\s+([0-9a-f]{6,40})',
        r'[Tt]his\s+reverts?\s+commit\s+([0-9a-f]{6,40})',
    ]

    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return match.group(1)
    return None


def save_revert_pairs(repo_name):
    repo_path = os.path.join(os.path.dirname(__file__), repo_name)
    if not os.path.exists(repo_path):
        print(f"Repository {repo_name} not found at {repo_path}")
        return

    # Revert commits with referenced commits
    print(f"Collecting revert commits from {repo_name}...")
    cmd = 'git log --all --grep="revert" -i --format="%H|%s"'
    reverts = run_git_command(cmd, repo_path)
    print(len(reverts))
    cnt = 0
    revert_data = []
    for ind, line in enumerate(reverts):
        if ind%20==0:
            print(ind)
        if '|' in line:
            commit_hash, subject = line.split('|', 1)

            # Get full commit message
            cmd = f'git log -1 --format="%B" {commit_hash}'
            full_msg = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                      cwd=repo_path, encoding='utf-8', errors='ignore').stdout.strip()

            # Check if contains commit reference
            reverted_hash = extract_reverted_commit(full_msg)
            if reverted_hash:
                # Verify and get reverted commit info
                cmd = f'git log -1 --format="%s" {reverted_hash}'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                        cwd=repo_path, encoding='utf-8', errors='ignore')

                if result.returncode == 0:
                    # Check if revert commit only modified one file
                    cmd = f'git show --name-only --pretty="" {commit_hash}'
                    files_modified = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                                    cwd=repo_path, encoding='utf-8',
                                                    errors='ignore').stdout.strip().split('\n')
                    files_modified = [f for f in files_modified if f]  # Remove empty strings
                    if len(files_modified) == 1:
                        file_path = files_modified[0]
                        # Skip header files and non-C/C++ files
                        if file_path.endswith(('.c', '.cpp', '.cc', '.cxx', '.c++')):
                            reverted_msg = result.stdout.strip()
                            # Get full message for reverted commit
                            cmd = f'git log -1 --format="%B" {reverted_hash}'
                            reverted_full_msg = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                                               cwd=repo_path, encoding='utf-8',
                                                               errors='ignore').stdout.strip()

                            # Get date for reverted commit
                            cmd = f'git log -1 --format="%ai" {reverted_hash}'
                            date_result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                                         cwd=repo_path, encoding='utf-8', errors='ignore')
                            date_str = date_result.stdout.strip()
                            if date_result.returncode != 0 or not date_str:
                                print(ind, "wrong", commit_hash)
                                continue
                            else:
                                date = date_str.split()[0]  # YYYY-MM-DD
                            revert_data.append({
                                "date": date,
                                "defective_modification": 1,
                                "project": repo_name,
                                "file_path": file_path,
                                "commit": reverted_hash,
                                "commit_message": reverted_full_msg,
                                "revert_commit": commit_hash,
                                "revert_message": full_msg
                            })
                            cnt+=1
                            if cnt%20==0:
                                print("Collected: ", cnt)


    revert_data.sort(key=lambda x: x['date'])
    cutoff_date = "2025-02-28"
    revert_data = [item for item in revert_data if item['date'] < cutoff_date]

    # Save as JSONL
    with open(f'{repo_name}_defective_1.jsonl', 'w', encoding='utf-8') as f:
        for item in revert_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Found {len(revert_data)} revert pairs")
    print(f"Saved to {repo_name}_defective_.jsonl")


def extract_single_function_commits(jsonl_file, repo_name):
    """Extract functions from revert commits"""
    cnt=0

    repo_path = os.path.join(os.path.dirname(__file__), repo_name)
    output_data = []

    reverted_commits_set = set()
    function_set = set()

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            commit_hash = data['commit']
            file_path = data['file_path']
            if commit_hash in reverted_commits_set:
                continue

            # Get the diff of the revert commit
            cmd = ['git', 'diff', f'{commit_hash}^', commit_hash, '--', file_path]
            diff_result = subprocess.run(cmd, capture_output=True, text=True,
                                         cwd=repo_path, encoding='utf-8', errors='ignore')

            if diff_result.returncode != 0:
                print("Error getting diff")
                continue
            # print(diff_result)
            if not diff_result.stdout:
                print("Empty diff")
                continue

            # Parse diff to find modified line numbers
            modified_lines = {'added': [], 'deleted': []}
            current_line_old = 0
            current_line_new = 0

            for line in diff_result.stdout.split('\n'):
                if line.startswith('@@'):
                    match = re.search(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
                    if match:
                        current_line_old = int(match.group(1))
                        current_line_new = int(match.group(2))
                elif line.startswith('+') and not line.startswith('+++'):
                    modified_lines['added'].append(current_line_new)
                    current_line_new += 1
                elif line.startswith('-') and not line.startswith('---'):
                    modified_lines['deleted'].append(current_line_old)
                    current_line_old += 1
                else:
                    current_line_old += 1
                    current_line_new += 1

            if not modified_lines['added'] and not modified_lines['deleted']:
                print("No modified lines found")
                continue

            print(f"Modified lines: {modified_lines['added'][:10]}{'...' if len(modified_lines['added']) > 10 else ''}")
            print(f"Modified lines: {modified_lines['deleted'][:10]}{'...' if len(modified_lines['deleted']) > 10 else ''}")
            parent_functions=[]
            current_functions=[]
            if modified_lines['deleted']:
                parent_functions = get_function_names_at_lines(
                    f"{commit_hash}^", file_path, modified_lines['deleted'], repo_path
                )

            # For added lines - use current commit
            if modified_lines['added']:
                current_functions = get_function_names_at_lines(
                    commit_hash, file_path, modified_lines['added'], repo_path
                )
            # Combine both
            function_names = set(parent_functions + current_functions)

            if len(function_names)==1:
                print(function_names)

                func_name = list(function_names)[0]
                # Get before version (at reverted_commit^)
                cmd = ['git', 'show', f'{commit_hash}^:{file_path}']
                before_content = subprocess.run(cmd, capture_output=True, text=True,
                                                cwd=repo_path, encoding='utf-8', errors='ignore').stdout
                # Get after version (at reverted_commit)
                cmd = ['git', 'show', f'{commit_hash}:{file_path}']
                after_content = subprocess.run(cmd, capture_output=True, text=True,
                                               cwd=repo_path, encoding='utf-8', errors='ignore').stdout

                # Extract function from both versions
                before_func = extract_function_by_name(before_content, func_name)
                after_func = extract_function_by_name(after_content, func_name)
                if before_func==after_func:
                    print("Identical Functions")
                    continue
                if cnt%100==0:
                    print("before: \n", before_func)
                    print("after: \n",after_func)
                if before_func and after_func and before_func not in function_set:
                    output_data.append({
                        "date": data["date"],
                        "defective_modification": data["defective_modification"],
                        "project": data["project"],
                        "file_path": data["file_path"],
                        "function_name":func_name,
                        "function_before": before_func,
                        "function_after": after_func,
                        "commit": data['commit'],
                        "commit_message": data['commit_message'],
                        "revert_commit": data['revert_commit'],
                        "revert_message": data['revert_message']
                    })
                    reverted_commits_set.add(data['commit'])
                    function_set.add(before_func)
                    cnt+=1
                    print(cnt)

                # Save results
            with open(f'{repo_name}_defective_2.jsonl', 'w', encoding='utf-8') as f:
                for item in output_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            print(f"Extracted {len(output_data)} function pairs")


def extract_function_by_name(file_content, func_name):
    """
    ctags to extract certain function
    """
    if not file_content or not func_name:
        return None

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False, encoding='utf-8') as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    try:
        # Run ctags with same options as original
        ctags_cmd = ['ctags', '-x', '--c-kinds=f', tmp_path]
        ctags_result = subprocess.run(ctags_cmd, capture_output=True, text=True)

        if ctags_result.returncode != 0:
            print(f"ctags run failed: {ctags_result.stderr}")
            return None

        # Parse ctags output
        functions_info = []
        for line in ctags_result.stdout.split('\n'):
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0]
                    line_no = int(parts[2])
                    functions_info.append((name, line_no))

        # Find target function
        target_func_info = None
        for name, start_line in functions_info:
            if name == func_name:
                target_func_info = (name, start_line)
                break

        if target_func_info is None:
            return None

        # Get function using brace matching for accurate boundaries
        lines = file_content.split('\n')
        start_line = target_func_info[1] - 1  # Convert to 0-based index

        # Find function boundaries using brace counting
        brace_count = 0
        func_started = False
        end_line = start_line
        in_string = False
        in_char = False
        in_comment = False
        in_multiline_comment = False

        for i in range(start_line, len(lines)):
            line = lines[i]
            j = 0
            while j < len(line):
                # Handle multi-line comments
                if j < len(line) - 1 and line[j:j + 2] == '/*' and not in_string and not in_char:
                    in_multiline_comment = True
                    j += 2
                    continue
                elif j < len(line) - 1 and line[j:j + 2] == '*/' and in_multiline_comment:
                    in_multiline_comment = False
                    j += 2
                    continue

                # Skip if in multi-line comment
                if in_multiline_comment:
                    j += 1
                    continue

                # Handle single-line comments
                if j < len(line) - 1 and line[j:j + 2] == '//' and not in_string and not in_char:
                    break  # Skip rest of line

                char = line[j]

                # Handle string literals
                if char == '"' and not in_char and (j == 0 or line[j - 1] != '\\'):
                    in_string = not in_string

                # Handle character literals
                elif char == "'" and not in_string and (j == 0 or line[j - 1] != '\\'):
                    in_char = not in_char

                # Count braces only if not in string/char/comment
                elif not in_string and not in_char:
                    if char == '{':
                        brace_count += 1
                        func_started = True
                    elif char == '}':
                        brace_count -= 1

                j += 1

            # Check if function is complete
            if func_started and brace_count == 0:
                end_line = i
                break

        # If brace matching failed, fall back to next function or end of file
        if brace_count != 0:
            # Sort functions by line number
            functions_info.sort(key=lambda x: x[1])

            # Find next function
            target_index = None
            for i, (name, line_no) in enumerate(functions_info):
                if name == func_name:
                    target_index = i
                    break

            if target_index is not None and target_index + 1 < len(functions_info):
                end_line = functions_info[target_index + 1][1] - 2  # Line before next function
            else:
                end_line = len(lines) - 1

        # Extract function content
        function_lines = lines[start_line:end_line + 1]

        # Remove trailing empty lines
        while function_lines and not function_lines[-1].strip():
            function_lines.pop()

        return '\n'.join(function_lines)

    except subprocess.CalledProcessError as e:
        print(f"ctags run failed: {e}")
        return None
    except Exception as e:
        print(f"Error during function extraction: {e}")
        return None
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def get_function_names_at_lines(commit_hash, file_path, line_numbers, repo_path):
    """Extract functions using ctags - returns empty list if all lines are outside functions"""

    # Get file content
    cmd = ['git', 'show', f'{commit_hash}:{file_path}']
    file_content = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=repo_path, encoding='utf-8', errors='ignore').stdout

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False, encoding='utf-8') as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    # Run ctags
    ctags_cmd = ['ctags', '-x', '--c-kinds=f', tmp_path]
    ctags_result = subprocess.run(ctags_cmd, capture_output=True, text=True)

    functions_info = []
    for line in ctags_result.stdout.split('\n'):
        if line:
            parts = line.split()
            if len(parts) >= 3:
                func_name = parts[0]
                line_no = int(parts[2])
                functions_info.append((func_name, line_no))
    functions_info.sort(key=lambda x: x[1])

    # Clean up
    os.unlink(tmp_path)

    # Find functions containing modified lines
    lines = file_content.split('\n')
    found_functions = set()
    lines_in_functions = 0

    for mod_line in line_numbers:
        for i, (name, start) in enumerate(functions_info):
            end = functions_info[i + 1][1] if i + 1 < len(functions_info) else len(lines)

            if start <= mod_line < end:
                found_functions.add(name)
                lines_in_functions += 1
                break

    if lines_in_functions == 0:
        return []

    return list(found_functions)


def filter_only_bug_related(jsonl_file, repo_name, api_key=None, gpt_model="gpt-4o"):
    """Filter only bug-related commits using GPT-4o"""

    # API key setup
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not client.api_key:
        raise ValueError("OpenAI API key not found")

    def is_likely_bug_related2(messages: List[str], num_runs=3, threshold=3) -> List[bool]:
        """Batch processing with GPT-4o"""

        prompt = """You are a code review expert analyzing reverted commits.

For each case below, you'll see:
1. ORIGINAL: The commit that was later reverted (the commit we're evaluating)
2. REVERT: The commit message explaining why it was reverted

Your task: Classify this revert case into one of the following categories.

Return "NEW" if this introduces or exposes bugs:
- The ORIGINAL commit was NOT primarily a bug fix (e.g., new features, refactoring, optimization, code cleanup)
- BUT it was reverted because it introduced new bugs, exposed existing bugs, system crashed or caused functional regressions

Return "YES" if this is a FAILED bug fix attempt:
- The ORIGINAL commit was attempting to fix a bug (crashes, errors, memory issues, logic errors, security vulnerabilities)
- AND it was reverted because the bug fix failed or was incomplete (e.g., test flaky, breaks compile, better solution found)

Return "NO" for ALL other cases, including:
- Insufficient information in revert message
- "Not needed anymore" or "better solution found"
- Simple feature changes without clear technical problems
- Any case where you need to make assumptions about the reason

IMPORTANT: When in doubt, choose "NO". Only use "NEW" and "YES" when there's explicit evidence of technical problems.

For each case:
- Start your answer with either NEW,YES, or NO, followed by a brief explanation of your reasoning.
- After each case, add this exact separator on a new line: ------------------------------------------------

Cases to analyze:
"""
        # Build batch prompt
        for i, msg in enumerate(messages):
            prompt += f"\n{i + 1}. {msg}"

            # Store votes for each message across runs
        votes1 = [0] * len(messages)  # Count of YES votes for each message
        votes2 = [0] * len(messages)  # Count of YES votes for each message
        expl = [{"NEW": [], "YES": [], "NO": []} for _ in range(len(messages))]  # Store explanations by vote type


        for run in range(num_runs):
            try:
                cnt=0
                while True:
                    response = client.chat.completions.create(
                        model=gpt_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=1000
                    )

                    # Parse responses
                    raw_text = response.choices[0].message.content.strip()


                    blocks = [b.strip() for b in re.split(r'-{5,}\s*\n', raw_text) if b.strip()]
                    if len(blocks) < len(messages):
                        number_pattern = r'(?:^|\n)(?=\d+[\.\)]\s*(?:\*\*)?(?:YES|NO|NEW))'
                        alt_blocks = re.split(number_pattern, gpt_response)
                        alt_blocks = [b.strip() for b in alt_blocks if b.strip()]

                        if len(alt_blocks) > len(blocks):
                            blocks = alt_blocks
                            print(f"[INFO] Used number pattern to split response (found {len(blocks)} blocks)")

                    parsed_answers = []
                    for block in blocks:
                        match = re.search(r'\b(YES|NO|NEW)\b', block, re.IGNORECASE)
                        if match:
                            label = match.group(1).upper()
                            clean_block = re.sub(r'```', '', block)
                            clean_block = re.sub(r'^\d+\.\s*', '', clean_block)
                            parsed_answers.append((label, clean_block.strip()))
                        else:
                            print(f"[WARN] No label found in case:\n{block}\n")

                    filtered_answers = []
                    filtered_explanations = []
                    for i, (label, explanation) in enumerate(parsed_answers, 1):
                        print(f"Case {i}: {label}")
                        print(explanation)
                        print("------")
                        filtered_answers.append(label)
                        filtered_explanations.append(explanation)
                    print(filtered_answers)
                    # Validation check
                    cnt += 1
                    if len(filtered_answers) == len(messages):
                        # Count votes and store explanations
                        for i, (ans, explanation) in enumerate(zip(filtered_answers, filtered_explanations)):
                            if i < len(messages):  # Safety check
                                ans_upper = ans.strip().upper()
                                if ans_upper == "NEW":
                                    votes1[i] += 1
                                    expl[i]["NEW"].append(explanation)
                                elif ans_upper == "YES":
                                    votes2[i] += 1
                                    expl[i]["YES"].append(explanation)
                                else:  # NO
                                    expl[i]["NO"].append(explanation)
                        break
                    else:
                        print("retry")
                    if cnt == 5:
                        print(f"Retry failed. Logging problematic prompt.")
                        # Save problematic prompt to file
                        with open('problematic_prompts.txt', 'a', encoding='utf-8') as f:
                            f.write(f"\n{'=' * 80}\n")
                            f.write(f"Timestamp: {datetime.now()}\n")
                            f.write(f"Run: {run + 1}, Expected: {len(messages)} responses\n")
                            f.write(f"Got: {len(filtered_answers)} responses\n")
                            f.write(f"Raw response:\n{response.choices[0].message.content}\n")
                            f.write(f"Prompt:\n{prompt}\n")
                            f.write(f"{'=' * 80}\n")
                        break



            except Exception as e:
                print(f"GPT API error on run {run + 1}: {e}")
                continue

        results = []
        for i, (vote1, vote2) in enumerate(zip(votes1, votes2)):
            total_votes = num_runs
            bug_votes = vote1 + vote2

            if vote1 >= threshold and vote2 == 0:
                # NEW majority
                label = "NEW"
                ratio = f"{vote1}/{total_votes}"
                # Save one of NEW explanations
                explanation = expl[i]["NEW"][0] if expl[i]["NEW"] else "No explanation available"
            elif vote2 >= threshold and vote1 == 0:
                # YES majority
                label = "YES"
                ratio = f"{vote2}/{total_votes}"
                # Save one of YES explanations
                explanation = expl[i]["YES"][0] if expl[i]["YES"] else "No explanation available"
            elif bug_votes >= threshold:
                # MIX - Pick the explanation from major votes
                label = "MIX"
                ratio = f"{vote1}/{total_votes}+{vote2}/{total_votes}"
                if vote1 > vote2:
                    explanation = expl[i]["NEW"][0] if expl[i]["NEW"] else "No explanation available"
                elif vote2 > vote1:
                    explanation = expl[i]["YES"][0] if expl[i]["YES"] else "No explanation available"
                else:
                    explanation = expl[i]["NEW"][0] if expl[i]["NEW"] else (
                        expl[i]["YES"][0] if expl[i]["YES"] else "No explanation available")
            else:
                label = "NO"
                ratio = f"{bug_votes}/{total_votes}"
                explanation = expl[i]["NO"][0] if expl[i]["NO"] else "No explanation available"

            results.append((label, ratio, explanation))

        return results


    # Process data
    filtered_data = []
    batch_size = 5  # Process 5 at a time

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]

    print(f"Processing {len(all_data)} commits...")

    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i + batch_size]

        # Prepare messages for GPT
        messages = []
        for ind,item in enumerate(batch):
            # Combine both commit messages for context
            combined_msg = f"{ind}.\nORIGINAL:\n {item['commit_message'][:1000]}\n\nREVERT:\n {item['revert_message'][:1000]} \n"
            messages.append(combined_msg)

        decisions = is_likely_bug_related2(messages)
        # Filter based on decisions
        for item, (decision, confidence, expl) in zip(batch, decisions):
            if decision != "NO":
                if decision=="NEW":
                    item['commit_type'] = "CLEAN_TO_DEFECTIVE"
                elif decision=="YES":
                    item['commit_type'] = "DEFECTIVE_TO_DEFECTIVE"
                else:
                    item['commit_type'] = "MIX"
                item['confidence'] = confidence
                item['explanation'] = expl
                print(decision, confidence, expl)
                filtered_data.append(item)
        print("--------------")
        # Rate limiting
        time.sleep(0.5)

        if (i + batch_size) % 5 == 0:
            print(f"Processed {min(i + batch_size, len(all_data))}/{len(all_data)}")

    # Save filtered results
    output_file = f'{repo_name}_defective_3.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nFiltering complete:")
    print(f"Total commits: {len(all_data)}")
    print(f"Bug-related commits: {len(filtered_data)} ({len(filtered_data) / len(all_data) * 100:.1f}%)")
    print(f"Saved to: {output_file}")

    return filtered_data














def collect_clean_commits(repo_name, defective_file):
    """
    Collect potentially clean commits from a repository.

    Args:
        repo_path: Path of the git repository
    """

    repo_path = os.path.join(os.path.dirname(__file__), repo_name)
    defective_dates = []
    seen_commits = set()
    with open(defective_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            defective_dates.append(data['date'])
            seen_commits.add(data['commit'])
    defective_dates.sort()
    def_idx = 0  # Start from the earliest defective date

    # Set cutoff date
    cutoff_date = datetime(2025, 2, 28)
    start_date = datetime(2020, 4, 3)
    # Get all commits before cutoff date
    cmd = ['git', 'log', f'--after={start_date.strftime("%Y-%m-%d")}', f'--before={cutoff_date.strftime("%Y-%m-%d")}', '--format=%H|%ad', '--date=short']
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_path, encoding='utf-8', errors='ignore')

    if result.returncode != 0:
        print(f"Git error: {result.stderr}")
        return

    all_commits = [
        parts for line in result.stdout.strip().split('\n') if line
        for parts in [line.split('|')]
        if len(parts) == 2
    ]
    all_commits.reverse()
    print(f"Number of commits: {len(all_commits)}")

    # Define problematic keywords
    problematic_keywords = [
        'revert', 'rollback', 'undo', 'back out',
        'incomplete', 'partial', 'temporary',
        'fix bug', 'bug fix', 'hotfix',
        'broke', 'broken', 'regression'
    ]

    clean_candidates = []
    collected_per_date = {}
    seen_functions = set()
    processed_combinations = set()

    for commit_hash, commit_date in all_commits:
        # Find which defective date range this belongs to
        if def_idx >= len(defective_dates):
            break
        target_date = defective_dates[def_idx]
        # Skip if this commit is older than current target_date
        if commit_date < target_date:
            continue
        # If 4 already collected for this date, move to the next
        if collected_per_date.get(target_date, 0) >= 4:
            def_idx += 1
            # Re-check this commit against the next target_date in next iteration
            continue

        # Single function check
        result = is_single_function_modification(repo_path, commit_hash)
        if not result[0]:
            continue

        _, file_path, func_name = result


        combination_key = f"{file_path}::{func_name}"
        if combination_key in processed_combinations:
            print(f"Skipping already processed: {combination_key}")
            continue

        processed_combinations.add(combination_key)

        # Get the function content at this commit
        cmd = ['git', 'show', f'{commit_hash}^:{file_path}']
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=repo_path, encoding='utf-8', errors='ignore')
        function_before = extract_function_by_name(result.stdout, func_name)

        cmd = ['git', 'show', f'{commit_hash}:{file_path}']
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=repo_path, encoding='utf-8', errors='ignore')
        function_target = extract_function_by_name(result.stdout, func_name)
        if not function_target or not function_before:
            continue
        if function_before==function_target:
            print("Identical result")
            continue
        # Get all subsequent commits that modify this file
        cmd = ['git', 'log', f'{commit_hash}..HEAD', '--pretty=format:%H%n%B%n---END---', '--', file_path]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=repo_path, encoding='utf-8', errors='ignore')

        if result.returncode != 0:
            continue


        commits_data = result.stdout.strip().split('---END---')
        subsequent_commits = []

        for commit_data in commits_data:
            if not commit_data.strip():
                continue

            lines = commit_data.strip().split('\n')
            if len(lines) >= 1:
                commit_hash_temp = lines[0]
                commit_message = '\n'.join(lines[1:]).strip()
                subsequent_commits.append((commit_hash_temp, commit_message))

        # Check up to 5 subsequent commits for function modifications
        is_problematic = False
        checked_count = 0
        for sub_commit_hash, commit_message in subsequent_commits:
            if checked_count >= 5:
                break

            # Get the function content at this subsequent commit
            cmd = ['git', 'show', f'{sub_commit_hash}:{file_path}']
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    cwd=repo_path, encoding='utf-8', errors='ignore')

            function_after = extract_function_by_name(result.stdout, func_name)

            # If function disappeared (renamed or deleted), stop checking
            if not function_after:
                is_problematic = True
                break

            # If function was modified
            if function_target != function_after:
                checked_count += 1
                # Check for problematic keywords in commit message
                commit_message_lower = commit_message.lower()
                func_name_lower = func_name.lower()
                print("Check", checked_count, sub_commit_hash, commit_message_lower[:20])

                for keyword in problematic_keywords:
                    if keyword in commit_message_lower:
                        is_problematic = True
                        print("Buggy keyword found")
                        break

                if is_problematic:
                    break

                # Update function_target for next comparison
                function_target = function_after
        # If no problems found (either checked 10 modifications or function disappeared without issues)
        if is_problematic:
            print("Skipping", len(subsequent_commits), commit_date)
            continue
        # Dead code identification
        if checked_count == 0:
            print("No modifications found - skipping")
            continue

        # Get commit details
        commit_details = get_commit_details(repo_path, commit_hash, file_path, func_name)
        if commit_details is None:
            continue  # Skip this commit if details couldn't be retrieved
        commit_details['project']=repo_name

        commit_hash = commit_details["commit"]
        function_code = commit_details["function_before"]
        if commit_hash in seen_commits or function_code in seen_functions:
            continue  # Skip if either is already seen
        seen_commits.add(commit_hash)
        seen_functions.add(function_code)

        print(commit_details["date"], commit_details["commit"])
        clean_candidates.append(commit_details)

        collected_per_date[target_date] = collected_per_date.get(target_date, 0) + 1
        print(f"[{commit_date}] Collected {len(clean_candidates)} candidates")

    # Save to file
    repo_name = os.path.basename(repo_path)
    with open(f'{repo_name}_clean_1.jsonl', 'w', encoding='utf_8') as f:
        for candidate in clean_candidates:
            f.write(json.dumps(candidate) + '\n')

    print(f"Collected {len(clean_candidates)} clean commit candidates")





def get_commit_details(repo_path, commit_hash, file_path, func_name):
    """Get detailed information about a commit."""

    msg_cmd = ['git', 'show', '-s', '--format=%B', commit_hash]
    msg_result = subprocess.run(msg_cmd, capture_output=True, text=True,
                                cwd=repo_path, encoding='utf-8', errors='ignore')
    commit_message = msg_result.stdout.strip()
    # Get commit date separately
    date_cmd = ['git', 'show', '-s', '--format=%ai', commit_hash]
    date_result = subprocess.run(date_cmd, capture_output=True, text=True,
                                 cwd=repo_path, encoding='utf-8', errors='ignore')
    date = date_result.stdout.strip().split()[0]  # 'YYYY-MM-DD'


    cmd = ['git', 'show', f'{commit_hash}^:{file_path}']
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=repo_path, encoding='utf-8', errors='ignore')
    function_before = extract_function_by_name(result.stdout, func_name)

    cmd = ['git', 'show', f'{commit_hash}:{file_path}']
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=repo_path, encoding='utf-8', errors='ignore')
    function_after = extract_function_by_name(result.stdout, func_name)

    if function_before == function_after:
        print(function_before)
        print("=-----------------",commit_hash)
        print(function_after)
        print("Identical Functions")
        return None
    return {
        "date": date,
        "defective_modification": 0,
        "project": repo_path,
        "file_path": file_path,
        "function_name": func_name,
        "function_before": function_before,
        "function_after": function_after,
        "commit": commit_hash,
        "commit_message": commit_message
    }


def is_single_function_modification(repo_path, commit_hash):
    """Check if commit modifies only a single function."""
    # Check single file modification
    cmd = f'git show --name-only --pretty="" {commit_hash}'
    files = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           cwd=repo_path, encoding='utf-8', errors='ignore').stdout.strip().split('\n')
    files = [f for f in files if f]

    if len(files) != 1 or not files[0].endswith(('.c', '.cpp', '.cc', '.cxx', '.c++')):
        return False, False

    file_path = files[0]

    # Check single function modification
    cmd = ['git', 'diff', f'{commit_hash}^', commit_hash, '--', file_path]
    diff_result = subprocess.run(cmd, capture_output=True, text=True,
                                 cwd=repo_path, encoding='utf-8', errors='ignore')

    if not diff_result.stdout:
        return False, False

    # Parse diff to find modified line numbers
    modified_lines = {'added': [], 'deleted': []}
    current_line_old = 0
    current_line_new = 0

    for line in diff_result.stdout.split('\n'):
        if line.startswith('@@'):
            match = re.search(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
            if match:
                current_line_old = int(match.group(1))
                current_line_new = int(match.group(2))
        elif line.startswith('+') and not line.startswith('+++'):
            modified_lines['added'].append(current_line_new)
            current_line_new += 1
        elif line.startswith('-') and not line.startswith('---'):
            modified_lines['deleted'].append(current_line_old)
            current_line_old += 1
        else:
            current_line_old += 1
            current_line_new += 1

    if not modified_lines['added'] and not modified_lines['deleted']:
        return False, False

    parent_functions = []
    current_functions = []

    if modified_lines['deleted']:
        parent_functions = get_function_names_at_lines(
            f"{commit_hash}^", file_path, modified_lines['deleted'], repo_path
        )

    if modified_lines['added']:
        current_functions = get_function_names_at_lines(
            commit_hash, file_path, modified_lines['added'], repo_path
        )

    function_names = set(parent_functions + current_functions)

    return len(function_names) == 1, file_path, list(function_names)[0] if len(function_names) == 1 else None


def classify_clean_commits(repo_name, clean_file, gpt_model="gpt-4o"):
    """Classify clean commits into bug-fix or improvement and filter problematic ones"""

    repo_path = os.path.join(os.path.dirname(__file__), repo_name)
    # API key setup
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not client.api_key:
        raise ValueError("OpenAI API key not found")

    def classify_commit_batch(batch_data, num_runs=3, threshold=3):
        """Classify commits as bug-fix or improvement"""
        prompt = """Classify each commit based on its message and code changes.

Categories:
- "DEFECTIVE_TO_CLEAN": Bug fix (fixing crashes, errors, memory issues, logic bugs)
- "CLEAN_TO_CLEAN": Improvement (refactoring, optimization, feature addition, cleanup)
- "OTHER": Changes that are trivial, ambiguous, or irrelevant (comment-only changes, formatting, whitespace edits, merge commits, renames without logic change)


For each case:
- Start your answer with either "DEFECTIVE_TO_CLEAN", "CLEAN_TO_CLEAN", or "OTHER", followed by a brief explanation of your reasoning.
- After each case, add this exact separator on a new line: ------------------------------------------------

Cases:
        """
        for i, item in enumerate(batch_data):
            prompt += f"\n\n{i + 1}. Commit message: {item['commit_message']}\n Function before: {item['function_before']}\n Function after: {item['function_after']}"

        votes_d2c = [0] * len(batch_data)
        votes_c2c = [0] * len(batch_data)
        explanations = [{"D2C": [], "C2C": []} for _ in range(len(batch_data))]

        for run in range(num_runs):
            try:
                cnt = 0
                while True:
                    response = client.chat.completions.create(
                        model=gpt_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=1000
                    )

                    raw_text = response.choices[0].message.content.strip()
                    blocks = [b.strip() for b in re.split(r'-{5,}\s*\n', raw_text) if b.strip()]
                    if len(blocks) < len(batch_data):
                        number_pattern = r'(?:^|\n)(?=\d+[\.\)]\s*(?:\*\*)?(?:DEFECTIVE_TO_CLEAN|CLEAN_TO_CLEAN))'
                        alt_blocks = re.split(number_pattern, gpt_response)
                        alt_blocks = [b.strip() for b in alt_blocks if b.strip()]

                        if len(alt_blocks) > len(blocks):
                            blocks = alt_blocks
                            print(f"[INFO] Used number pattern to split response (found {len(blocks)} blocks)")
                    # Validate response
                    if len(blocks) == len(batch_data):
                        valid = True
                        temp_d2c = [False] * len(batch_data)
                        temp_c2c = [False] * len(batch_data)

                        for i, block in enumerate(blocks):
                            if "DEFECTIVE_TO_CLEAN" in block:
                                temp_d2c[i] = True
                            elif "CLEAN_TO_CLEAN" in block:
                                temp_c2c[i] = True
                            elif "OTHER" in block:
                                continue
                            else:
                                valid = False
                                break

                        if valid:
                            # Update votes and explanations
                            for i, block in enumerate(blocks):
                                if temp_d2c[i]:
                                    votes_d2c[i] += 1
                                    explanations[i]["D2C"].append(block.strip())
                                elif temp_c2c[i]:
                                    votes_c2c[i] += 1
                                    explanations[i]["C2C"].append(block.strip())
                            break
                    else:
                        print("GPT response out of shape, retry", cnt)
                    cnt += 1
                    if cnt == 5:
                        print(f"Failed after 5 retries in run {run + 1}")
                        return [("MIX", "0/0", "Failed to process")] * len(batch_data)

            except Exception as e:
                print(f"GPT error: {e}")
                continue

        results = []
        for i in range(len(batch_data)):
            if votes_d2c[i] >= threshold:
                category = "DEFECTIVE_TO_CLEAN"
                confidence = f"{votes_d2c[i]}/{num_runs}"
                explanation = explanations[i]["D2C"][0] if explanations[i]["D2C"] else ""
            elif votes_c2c[i] >= threshold:
                category = "CLEAN_TO_CLEAN"
                confidence = f"{votes_c2c[i]}/{num_runs}"
                explanation = explanations[i]["C2C"][0] if explanations[i]["C2C"] else ""
            else:
                category = "MIX"
                confidence = f"{votes_d2c[i]}+{votes_c2c[i]}/{num_runs}"
                explanation = "Mixed classification"

            results.append((category, confidence, explanation))

        return results

    # Load and process
    with open(clean_file, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]

    batch_size = 5

    print("Classifying with GPT", len(all_data))
    filtered_data = []
    for i in range(0, len(all_data), batch_size):
        print(i)
        batch = all_data[i:i + batch_size]
        classify_results = classify_commit_batch(batch)

        for item, (category, conf, expl) in zip(batch, classify_results):
            if category != "MIX":
                item['commit_type'] = category
                item['confidence'] = conf
                item['explanation'] = expl
                print(category,conf,expl)
                filtered_data.append(item)

        if (i + batch_size) % 50 == 0:
            print(f"Processed {min(i + batch_size, len(all_data))}/{len(all_data)}")

        time.sleep(0.5)

    # Save results
    output_file = f'{repo_path}_clean_2.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in filtered_data:
            f.write(json.dumps(item) + '\n')

    print(f"\nFiltering complete:")
    print(f"Total: {len(all_data)}")
    print(f"After filtering: {len(filtered_data)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    repo_name = sys.argv[1] if len(sys.argv) > 1 else "ladybird"
    save_revert_pairs(repo_name)
    extract_single_function_commits(f"{repo_name}_defective_1.jsonl", repo_name)
    filter_only_bug_related(f"{repo_name}_defective_2.jsonl", repo_name, api_key=api_key)
    collect_clean_commits(repo_name, f"{repo_name}_defective_3.jsonl")
    classify_clean_commits(repo_name, f"{repo_name}_clean_1.jsonl")
