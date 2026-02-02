import json
import glob
import os
from collections import defaultdict
import random

tt1,tt2=0,0
def split_dataset_by_time():
    global tt1, tt2
    # Group by project
    project_data = defaultdict(list)
    all_defective_data = []  # For sampling

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
                all_defective_data.append(data)

    train_data_all = []
    valid_data_all = []
    test_data_all = []

    # Process each project
    for project, entries in project_data.items():
        # Sort by commit date
        entries.sort(key=lambda x: x.get('date', x.get('timestamp', '')))

        # Separate by label
        fixed_0 = [e for e in entries if e['defective_modification'] == 0]
        fixed_1 = [e for e in entries if e['defective_modification'] == 1]

        # Balance 1:1 ratio per project
        min_count = min(len(fixed_0), len(fixed_1))
        tt1+=len(fixed_1)
        tt2+=len(fixed_0)
        print(project, "defective modification:", len(fixed_1), "clean modification:", len(fixed_0))

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

        train_0, valid_0, test_0 = split_by_time(fixed_0)
        train_1, valid_1, test_1 = split_by_time(fixed_1)

        train_data_all.extend(train_0 + train_1)
        valid_data_all.extend(valid_0 + valid_1)
        test_data_all.extend(test_0 + test_1)

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
        print(f"  defective_modification=0: {cnt}, defective_modification=1: {len(data) - cnt}")


random.seed(42)
split_dataset_by_time()
print(f"Total defective modification: {tt1}, Total clean modification:  {tt2}")