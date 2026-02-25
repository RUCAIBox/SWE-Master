import json
import argparse
from dataclasses import asdict
from pathlib import Path
from tqdm import tqdm

# Original imports remain unchanged
from r2egym.agenthub.trajectory.swebench_utils import make_test_spec, swebench_parse
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER, get_eval_type
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
from swebench_fork_swerebench.harness.test_spec.test_spec import make_test_spec as make_test_spec_swerebench
from swebench_fork_swerebench.harness.grading import get_logs_eval_new as get_logs_eval_swerebench
from swebench_fork_swerebench.harness.constants import EvalType as EvalType_SWEREBENCH
from swebench_fork_swerebench.harness.constants import FAIL_ONLY_REPOS as FAIL_ONLY_REPOS_SWEREBENCH
from swebench_fork_swerebench.harness.grading import get_eval_tests_report as get_eval_tests_report_swerebench
from swebench_fork_swerebench.harness.grading import get_resolution_status as get_resolution_status_swerebench
from swebench_fork_swegym.harness.test_spec import make_test_spec as make_test_spec_swegym
from swebench_fork_swegym.harness.grading import get_logs_eval_new as get_logs_eval_swegym
from swebench_fork_swegym.harness.grading import get_eval_tests_report as get_eval_tests_report_swegym
from swebench_fork_swegym.harness.grading import get_resolution_status as get_resolution_status_swegym

# Load stored cache
def load_from_json(filename):
    if "swe_rebench" in str(filename):
        from swebench_fork_swerebench.harness.test_spec.test_spec import TestSpec
    elif "swe_gym" in str(filename):
        from swebench_fork_swegym.harness.test_spec import TestSpec
    elif "swe_bench" in str(filename):
        from swebench.harness.test_spec.test_spec import TestSpec
    else:
        from r2egym.agenthub.trajectory.swebench_utils import TestSpec 
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def process_make_test_spec_for_all(ds):
    assert ds, f"Dataset not provided"
    docker_image = ds.get("docker_image") or ds.get("image_name")
    if not docker_image:
        raise ValueError(f"No docker image found in ds: {ds}")

    swebench_verified = "swebench" in docker_image or "epoch-research" in docker_image
    swesmith = "swesmith" in docker_image
    swerebench = "swerebench" in docker_image
    swegym = "xingyaoww" in docker_image

    test_spec = None
    if swegym:
        test_spec = make_test_spec_swegym(ds)
    elif swebench_verified:
        test_spec = make_test_spec(ds)
    elif swerebench:
        test_spec = make_test_spec_swerebench(ds)
    
    return test_spec

def save_to_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def check_cache_exists(file_path):
    return Path(file_path).exists()

def judge_type_dataset(docker_image, instance_id, base_dir):
    """
    Returns the corresponding cache path based on the docker_image type
    """
    base_dir = Path(base_dir)
    if "xingyaoww" in docker_image:
        sub_dir = "swe_gym"
    elif "swebench" in docker_image or "epoch-research" in docker_image:
        sub_dir = "swe_bench_verified"
    elif "swerebench" in docker_image:
        sub_dir = "swe_rebench"
    else:
        sub_dir = "others"
    
    target_dir = base_dir / sub_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{instance_id}.json"

def main():
    # 1. Build argument parser
    parser = argparse.ArgumentParser(description="Process SWE-bench instances and generate test specs.")
    
    parser.add_argument(
        "--base_dir", 
        type=str, 
        default="./R2E-Gym/test_spec_cache",
        help="Directory to store and read test spec cache files"
    )
    parser.add_argument(
        "--data_file_path", 
        type=str, 
        required=True,
        help="Path to the source JSON data file"
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_add_make_test_spec",
        help="Suffix to add to the processed JSON file"
    )

    args = parser.parse_args()

    # 2. Convert paths
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    data_file_path = Path(args.data_file_path)

    # Load dataset
    if not data_file_path.exists():
        print(f"Error: Data file {data_file_path} not found.")
        return

    with open(data_file_path, "r") as f:
        ds_selected = json.load(f)

    # Statistics
    total_instances = len(ds_selected)
    cached_count = 0
    processed_count = 0
    error_count = 0
    
    for ds in tqdm(ds_selected, desc="Processing instances"):
        instance_id = ds.get('instance_id', 'unknown')
        try:
            docker_image = ds.get("docker_image", "")
            
            # Get the corresponding cache file path
            file_path = judge_type_dataset(docker_image, instance_id, base_dir)

            if check_cache_exists(file_path):
                cached_count += 1
                cached_make_test_spec = load_from_json(file_path)
                ds['make_test_spec'] = cached_make_test_spec
                continue 
            
            # Process and save
            test_spec_obj = process_make_test_spec_for_all(ds)
            if test_spec_obj:
                print(f"Storage path: {file_path}")
                test_spec_dict = asdict(test_spec_obj)
                save_to_json(file_path, test_spec_dict)
                ds['make_test_spec'] = test_spec_dict
                processed_count += 1
            
        except Exception as e:
            error_count += 1
            print(f"Error processing instance {instance_id}: {e}")
            with open(base_dir / "error_log.txt", "a") as log_file:
                log_file.write(f"{instance_id}: {e}\n")

    # 3. Save new results
    new_file_path = data_file_path.parent / (data_file_path.stem + args.output_suffix + ".json")
    with open(new_file_path, "w") as f:
        json.dump(ds_selected, f, indent=2)
    
    print(f"\nFinished! New file written to: {new_file_path}")
    print(f"Summary: Total: {total_instances}, Cached: {cached_count}, Processed: {processed_count}, Errors: {error_count}")

if __name__ == "__main__":
    main()