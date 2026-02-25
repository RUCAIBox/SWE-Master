import glob
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

from rllm.data.dataset import DatasetRegistry


# Change this to: Local json/jsonl file path or directory path
# - Can be a single file: "/path/to/train.jsonl"
# - Can also be a directory: "/path/to/my_data_dir" (will search for json/jsonl recursively)
INPUT_SOURCES = [
    "./data_examples/inference_data/ood_data/equal_0.3999999_0.9999999_swe-gym_and_rebench_demo_add_make_test_spec.json"
]

# A "Dataset Name" used to create a directory under DatasetRegistry
# Final output will be written to: rllm/data/datasets/<DATASET_NAME>/
# Example: /code/sunshuang/policy_rl/ss_debug/rllm/rllm/data/datasets/SWE_REBENCH_AND_R2E_AND_GYM_ALL_FILTER_OVERLONG_OOD_RL/
DATASET_NAME = "SWE_REBENCH_AND_R2E_AND_GYM_ALL_FILTER_OVERLONG_OOD_RL"

# Shard settings
TRAIN_NUM_SHARDS = 1
TEST_NUM_SHARDS = 1

# If the filename can distinguish between train/test, use these rules
# - e.g., filenames containing "train" are treated as train
# - filenames containing "test" / "val" / "valid" / "dev" are treated as test
# Files that do not match will default to train (you can modify this)
SPLIT_RULES = {
    "train": ["train"],
    "test": ["test", "val", "valid", "dev"],
}


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in jsonl file {path} at line {line_no}: {e}") from e


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # Supports: JSON root as either list or dict
        if isinstance(obj, list):
            return [dict(x) for x in obj]
        if isinstance(obj, dict):
            # Single sample dict -> convert to list of length 1
            return [dict(obj)]
        raise ValueError(f"{path}: JSON root must be list or dict, got {type(obj)}")
    elif ext == ".jsonl":
        return [dict(x) for x in iter_jsonl(path)]
    else:
        raise ValueError(f"Unsupported file type: {path} (only .json/.jsonl)")


def discover_input_files(sources: List[str]) -> List[str]:
    """Expand files/directories in INPUT_SOURCES into a list of all json/jsonl files."""
    files: List[str] = []
    for src in sources:
        if os.path.isdir(src):
            # Search recursively
            files.extend(glob.glob(os.path.join(src, "**", "*.json"), recursive=True))
            files.extend(glob.glob(os.path.join(src, "**", "*.jsonl"), recursive=True))
        elif os.path.isfile(src):
            files.append(src)
        else:
            raise FileNotFoundError(f"Input source not found: {src}")
    # Deduplicate + sort (ensures reproducibility)
    return sorted(list(dict.fromkeys(files)))


def infer_split_from_filename(path: str) -> str:
    """Infer split from filename; defaults to train if unable to infer."""
    # name = os.path.basename(path).lower()
    # for split, keywords in SPLIT_RULES.items():
    #     for kw in keywords:
    #         if kw in name:
    #             return split
    return "train"


def make_process_fn():
    def process_fn(row: Dict[str, Any]) -> Dict[str, Any]:
        # Implement your preprocessing logic here
        if 'make_test_spec' in row:
            temp = row['make_test_spec']
            row['make_test_spec'] = json.dumps(temp)
            # print(row) 
            # exit()
        row_dict = dict(row)
        # problem_statement = row_dict.get("problem_statement", "")
        return row_dict

    return process_fn


def shard_list(data: List[Dict[str, Any]], num_shards: int) -> List[List[Dict[str, Any]]]:
    """Split data evenly into num_shards parts."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if len(data) == 0:
        return [[] for _ in range(num_shards)]

    shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_shards)]
    for i, item in enumerate(data):
        shards[i % num_shards].append(item)
    return shards


def register_shards(dataset_name: str, split: str, data: List[Dict[str, Any]], num_shards: int):
    """
    Shard data for a split and register them one by one:
    - split=train -> train_00000, train_00001, ...
    - split=test  -> test_00000, ...
    """
    shards = shard_list(data, num_shards=num_shards)
    registered = []
    for shard_id, shard_data in enumerate(shards):
        shard_split_name = f"{split}_{shard_id:05d}"
        ds = DatasetRegistry.register_dataset(dataset_name, shard_data, shard_split_name)
        registered.append(ds)
        print(f"[OK] Registered {dataset_name}:{shard_split_name} with {len(shard_data)} examples")
    return registered


def prepare_swe_data_from_local_json(
    input_sources: List[str],
    dataset_name: str,
    train_num_shards: int = 10,
    test_num_shards: int = 1,
) -> Tuple[List[Any], List[Any]]:
    """
    Read from local json/jsonl -> Preprocess -> Shard -> Save as parquet and register.
    """
    if not input_sources:
        raise ValueError("INPUT_SOURCES is empty. Please provide json/jsonl files or directories.")

    process_fn = make_process_fn()

    files = discover_input_files(input_sources)
    print(f"Discovered {len(files)} json/jsonl files.")
    for p in files:
        print(f" - {p}")

    train_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for path in files:
        split = infer_split_from_filename(path)
        print(f"\nLoading {path}  -> split={split}")
        rows = load_json_or_jsonl(path)
        print(f"Loaded {len(rows)} raw rows")

        processed = [process_fn(r) for r in rows]
        print(f"Processed {len(processed)} rows")

        if split == "test":
            test_rows.extend(processed)
        else:
            train_rows.extend(processed)

    print("\n=== Summary before sharding ===")
    print(f"Train examples: {len(train_rows)}")
    print(f"Test  examples: {len(test_rows)}")

    train_datasets = register_shards(dataset_name, "train", train_rows, num_shards=train_num_shards)
    test_datasets = []
    if len(test_rows) > 0:
        test_datasets = register_shards(dataset_name, "test", test_rows, num_shards=test_num_shards)

    return train_datasets, test_datasets


if __name__ == "__main__":
    train_datasets, test_datasets = prepare_swe_data_from_local_json(
        input_sources=INPUT_SOURCES,
        dataset_name=DATASET_NAME,
        train_num_shards=TRAIN_NUM_SHARDS,
        test_num_shards=TEST_NUM_SHARDS,
    )

    print("\nSummary:")
    print(f"Total train shard datasets: {len(train_datasets)}")
    print(f"Total test shard datasets: {len(test_datasets)}")

    # if train_datasets and len(train_datasets[0]) > 0:
    #     print("\nSample train example from first shard:")
    #     print(train_datasets[0].get_data()[0])

    # if test_datasets and len(test_datasets[0]) > 0:
    #     print("\nSample test example from first shard:")
    #     print(test_datasets[0].get_data()[0])