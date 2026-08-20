import json
import glob
import os
from collections import defaultdict
import random

total_defective_cnt,total_clean_cnt=0,0
def split_dataset_by_time():
    """
    Split Policy: Chronologically splits data into Train/Valid/Test (8:1:1) per project.
    """
    global total_defective_cnt, total_clean_cnt
    # Group by project
    project_data = defaultdict(list)

    file_paths = glob.glob('./*_defective_3.jsonl') + glob.glob('./*_clean_2.jsonl')
    print("Total Repos:", len(file_paths)//2)
    for file_path in file_paths:
        project_name = os.path.basename(file_path).split('_')[0]
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)

                # Check for unanimous decision
                confidence = data.get('confidence', '')

                if '+' in confidence:
                    parts = confidence.split('+')
                    x = int(parts[0].split('/')[0])
                    y = int(parts[1].split('/')[0])
                    if x + y != 3:
                        continue
                elif '3/3' not in confidence:
                        continue
                project_data[project_name].append(data)

    train_data_all = []
    valid_data_all = []
    test_data_all = []

    # Process each project
    for project, entries in project_data.items():
        # Sort by commit date
        entries.sort(key=lambda x: x.get('date', x.get('timestamp', '')))

        # Separate by label
        current_clean_modifications = [e for e in entries if e['defective_modification'] == 0]
        current_defective_modifications = [e for e in entries if e['defective_modification'] == 1]

        # Balance 1:1 ratio per project
        min_count = min(len(current_clean_modifications), len(current_defective_modifications))
        total_defective_cnt+=len(current_defective_modifications)
        total_clean_cnt+=len(current_clean_modifications)
        print(project, "defective modification:", len(current_defective_modifications), "clean modification:", len(current_clean_modifications))

        # Time match split
        def split_by_time(data, ratios=[0.8, 0.1, 0.1]):
            n = len(data)
            if n == 0:
                return [], [], []

            train_size = round(n * ratios[0])
            valid_size = round(n * ratios[1])
            test_size = n - train_size - valid_size

            train = data[:train_size]
            valid = data[train_size:train_size + valid_size]
            test = data[train_size + valid_size:]

            return train, valid, test

        train_split_clean, valid_split_clean, test_split_clean = split_by_time(current_clean_modifications)
        train_split_defective, valid_split_defective, test_split_defective = split_by_time(current_defective_modifications)

        train_data_all.extend(train_split_clean + train_split_defective)
        valid_data_all.extend(valid_split_clean + valid_split_defective)
        test_data_all.extend(test_split_clean + test_split_defective)

    random.shuffle(train_data_all)
    random.shuffle(valid_data_all)
    random.shuffle(test_data_all)

    # Write split files
    for data, filename in [(train_data_all, '1_train.jsonl'),
                           (valid_data_all, '1_valid.jsonl'),
                           (test_data_all, '1_test.jsonl')]:
        with open(filename, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"{filename}: {len(data)} entries")
        cnt = 0
        for entry in data:
            if entry['defective_modification'] == 0:
                cnt += 1
        print(f"  defective_modification=1: {len(data) - cnt}, defective_modification=0: {cnt}")


random.seed(42)
split_dataset_by_time()
print(f"Total defective modification: {total_defective_cnt}, Total clean modification:  {total_clean_cnt}")