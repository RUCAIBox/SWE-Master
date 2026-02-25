import json

from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry
from rllm.data.utils import fetch_live_code_bench_system_prompt


def prepare_deepcoder_data(train_size: int = None, test_size: int = None):
    train_dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="train")
    test_dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="test")

    def preprocess_fn(example, idx):
        starter_code = example.get("starter_code", "")
        question = fetch_live_code_bench_system_prompt(example["problem"], starter_code if starter_code else None)

        tests = json.loads(example["tests"])
        metadata = example.get("metadata", {})

        for test in tests:
            if test.get("testtype") == "functional" and metadata.get("func_name") is not None:
                test["metadata"] = {"func_name": metadata["func_name"]}
            else:
                test["metadata"] = {"func_name": None}

        return {"question": question, "ground_truth": tests, "data_source": "livecodebench", "uid": f"deepcoder_{idx}", "index": idx, "starter_code": starter_code, "metadata": metadata}

    if train_size:
        train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))
    if test_size:
        test_dataset = test_dataset.select(range(min(test_size, len(test_dataset))))

    train_dataset = train_dataset.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16)
    train_dataset = DatasetRegistry.register_dataset("deepcoder", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("deepcoder", test_dataset, "test")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_deepcoder_data()
    print(f"  - Train dataset: {len(train_dataset.get_data())} examples")
    print(f"  - Test dataset: {len(test_dataset.get_data())} examples")
    print(train_dataset.get_data()[0])
