from datasets import load_dataset
import json
def load_difficulty_mapping(jsonl_path):
    """
    Read difficulty jsonl file to build a mapping dictionary from instance_id to pre_difficulty
    """
    difficulty_map = {}

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            data = json.loads(line)
            instance_id = data["instance_id_or_commit_hash"]
            pre_difficulty = data["reward_mean"]
            if instance_id is not None and pre_difficulty is not None:
                difficulty_map[instance_id] = pre_difficulty

    return difficulty_map


dataset_path = "xx/swe_dataset_path" #folder, including parquets
output_path = "xx/swe_dataset_path_add_difficulty.json" #json

difficulty_json_path = "./data_preparation/metadata/bon_stats_oss_others.jsonl"

difficulty_map = load_difficulty_mapping(difficulty_json_path)
ds = load_dataset(dataset_path, split="test") #split maybe train or test, based on the datasets.


print("Processing dataset...")
final_data_list = []
total_count = 0
matched_count = 0

if "smith" in difficulty_json_path: #swe-smith (not-use)
    for item in ds:
        total_count += 1
        instance_id = item["instance_id"]
        
        if instance_id in difficulty_map:
            item["pre_difficulty"] = difficulty_map[instance_id]
            final_data_list.append(item)
            matched_count += 1
else: #other swe datasets
    for item in ds:
        total_count += 1
        if 'commit_hash' in item:
            instance_id = item['commit_hash']
        else:
            instance_id = item["base_commit"]
        
        if instance_id in difficulty_map:
            item["pre_difficulty"] = difficulty_map[instance_id]
            final_data_list.append(item)
            matched_count += 1
            
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(final_data_list, f, ensure_ascii=False, indent=2)
