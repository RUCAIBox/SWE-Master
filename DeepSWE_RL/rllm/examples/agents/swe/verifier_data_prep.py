#!/usr/bin/env python3
"""
Clean Verifier Data Preparation Script

Simple 4-step process:
1) Load all trajectories
2) Filter to mixed reward pairs (exactly 2 per docker image: 1 positive + 1 negative)
3) Convert to verifier format
4) Push to HuggingFace
"""

import logging
import multiprocessing as mp
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import datasets
import fire
import numpy as np
import orjson
from tqdm import tqdm
from transformers import AutoTokenizer

# R2E-Gym imports
try:
    from r2egym.agenthub.trajectory.trajectory import Trajectory

    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Warning: Missing dependencies: {e}")
    HAS_DEPENDENCIES = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def compute_total_tokens(training_data_entry, tokenizer_name="Qwen/Qwen2.5-Coder-32B-Instruct"):
    """
    Compute the total number of tokens in the training data entry
       Args:
           training_data_entry: e.g.,
               [{'role': 'system', 'content': 'System prompt'},
                {'role': 'user', 'content': 'User prompt'},
                {'role': 'assistant', 'content': 'Assistant prompt'}]
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    combined_text = " ".join([x["content"] for x in training_data_entry])
    # Encode the text to get the token count
    # add_special_tokens=False so that e.g. GPT-2's <|endoftext|> is not counted
    tokens = tokenizer.encode(combined_text, add_special_tokens=False)
    return len(tokens)


def condense(
    input_str: str,
    max_tokens: int = 31000,
    tokenizer_name="Qwen/Qwen2.5-Coder-32B-Instruct",
) -> str:
    """
    If the token count of input_str exceeds max_tokens, then starting with the second
    [USER]...[/USER] block (the oldest after the first), replace its inner content with
    a placeholder until the total token count is under the limit.

    The first [USER] block is left intact.
    """
    placeholder = "<Observation condensed for saving context>"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Check initial token count
    if len(tokenizer.encode(input_str)) <= max_tokens:
        return input_str

    # Regex to match [USER] blocks
    pattern = re.compile(r"(\[USER\])(.*?)(\[/USER\])", re.DOTALL)

    new_str = input_str
    # Continue condensing until token count is below the threshold or nothing changes.
    while len(tokenizer.encode(new_str)) > max_tokens:
        # Re-find all [USER] blocks in the updated string
        matches = list(pattern.finditer(new_str))
        if len(matches) <= 1:
            # Nothing more to condense (either no [USER] blocks or only one exists)
            break

        replaced = False
        # Iterate over all [USER] blocks starting from the second one
        for i, m in enumerate(matches):
            if i == 0:
                continue  # leave the first [USER] block unchanged
            # If already condensed, skip it
            if m.group(2).strip() == placeholder:
                continue
            # Build the new block with condensed content
            new_block = m.group(1) + placeholder + m.group(3)
            # Replace this block in the string using its current indices
            start, end = m.start(), m.end()
            new_str = new_str[:start] + new_block + new_str[end:]
            replaced = True
            # Break out after replacing one block so we can re-check token count
            break
        if not replaced:
            # All subsequent [USER] blocks are already condensed
            break

    # print("TRUNCATION HAPPENED")
    # print(new_str)
    # print("=" * 100)
    return new_str


def condense_thoughts(
    input_str: str,
    max_tokens: int = 31000,
    tokenizer_name="Qwen/Qwen2.5-Coder-32B-Instruct",
) -> str:
    """
    If the token count of input_str exceeds max_tokens, then starting with the second
    [ASSISTANT]...[/ASSISTANT] block (the oldest after the first), replace its inner content with
    a placeholder until the total token count is under the limit.

    The first [ASSISTANT] block is left intact.
    """
    placeholder = "<Thought condensed for saving context>"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Check initial token count
    if len(tokenizer.encode(input_str)) <= max_tokens:
        return input_str

    # Regex to match thoughts between [ASSISTANT] and <function
    pattern = re.compile(r"(\[ASSISTANT\])(.*?)(<function)", re.DOTALL)

    new_str = input_str
    # Continue condensing until token count is below the threshold or nothing changes.
    while len(tokenizer.encode(new_str)) > max_tokens:
        # Re-find all [ASSISTANT] blocks in the updated string
        matches = list(pattern.finditer(new_str))
        if len(matches) <= 1:
            # Nothing more to condense (either no [ASSISTANT] blocks or only one exists)
            break

        # Sort matches by content length (descending) - biggest first
        # Filter out already condensed blocks
        uncondensed_matches = [m for m in matches if m.group(2).strip() != placeholder]

        if not uncondensed_matches:
            # All blocks are already condensed
            break

        # Sort by content length (group(2) is the content)
        uncondensed_matches.sort(key=lambda m: len(m.group(2)), reverse=True)

        # Replace the longest uncondensed block
        m = uncondensed_matches[0]
        new_block = m.group(1) + placeholder + m.group(3)
        # Replace this block in the string using its current indices
        start, end = m.start(), m.end()
        new_str = new_str[:start] + new_block + new_str[end:]

        # print warning for removing
        print(f"Warning: Removing {len(tokenizer.encode(m.group(2)))} tokens from [ASSISTANT] block")

    return new_str


def load_file_trajectories(file_info: tuple[str, str]) -> list[dict]:
    """Simply load ALL trajectories from a single file."""
    filename, filepath = file_info
    trajectories = []

    with open(filepath, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = orjson.loads(line)
                    docker_img = data.get("docker_image", "unknown")
                    reward = data.get("reward", data.get("success", 0))

                    trajectories.append({"data": data, "docker_image": docker_img, "reward": reward, "file": filename})
                except Exception:
                    continue

    return trajectories


def load_all_trajectories(verifier_traj_dir: str) -> list[dict]:
    """Step 1: Load ALL trajectories from all JSONL files."""
    jsonl_files = [f for f in os.listdir(verifier_traj_dir) if f.endswith(".jsonl")]

    if not jsonl_files:
        logger.error(f"No JSONL files found in {verifier_traj_dir}")
        return []

    logger.info(f"Step 1: Loading trajectories from {len(jsonl_files)} files...")

    file_infos = [(f, os.path.join(verifier_traj_dir, f)) for f in jsonl_files]

    # Use multiprocessing for fast file processing
    max_workers = min(len(file_infos), mp.cpu_count())
    all_trajectories = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(load_file_trajectories, file_infos), total=len(file_infos), desc="Loading files"))

    for file_trajectories in results:
        all_trajectories.extend(file_trajectories)

    logger.info(f"Loaded {len(all_trajectories)} total trajectories")
    return all_trajectories


def filter_to_mixed_reward_pairs(all_trajectories: list[dict]) -> list[dict]:
    """
    Step 2: Filter to exactly 2 trajectories per docker image (1 positive + 1 negative).
    """
    logger.info("Step 2: Filtering to docker images with both positive and negative rewards...")

    # Group trajectories by docker image and reward
    docker_trajectories = {}  # {docker_image: {0: [trajs], 1: [trajs]}}

    for traj in all_trajectories:
        docker_img = traj["docker_image"]
        reward = traj["reward"]

        if docker_img not in docker_trajectories:
            docker_trajectories[docker_img] = {0: [], 1: []}

        if reward in [0, 1]:
            docker_trajectories[docker_img][reward].append(traj)

    logger.info(f"Total unique docker images: {len(docker_trajectories)}")

    # Filter to docker images with BOTH positive and negative rewards
    mixed_reward_dockers = {}
    for docker_img, reward_trajs in docker_trajectories.items():
        has_positive = len(reward_trajs[1]) > 0
        has_negative = len(reward_trajs[0]) > 0

        if has_positive and has_negative:
            mixed_reward_dockers[docker_img] = reward_trajs

    logger.info(f"Docker images with BOTH + and - rewards: {len(mixed_reward_dockers)}")

    # Select exactly 1 positive and 1 negative trajectory per docker image
    filtered_trajectories = []

    for docker_img, reward_trajs in mixed_reward_dockers.items():
        # Take first positive and first negative trajectory
        positive_traj = reward_trajs[1][0]  # First positive (reward=1)
        negative_traj = reward_trajs[0][0]  # First negative (reward=0)

        filtered_trajectories.append(positive_traj)
        filtered_trajectories.append(negative_traj)

    expected_count = len(mixed_reward_dockers) * 2
    positive_count = sum(1 for t in filtered_trajectories if t["reward"] == 1)
    negative_count = sum(1 for t in filtered_trajectories if t["reward"] == 0)

    logger.info(f"Filtered trajectories: {len(filtered_trajectories)} (expected: {expected_count})")
    logger.info(f"Distribution: {len(mixed_reward_dockers)} docker images × 2 trajectories each")
    logger.info(f"Positive trajectories: {positive_count}")
    logger.info(f"Negative trajectories: {negative_count}")

    return filtered_trajectories


def filter_with_agent_priority(all_trajectories: list[dict]) -> list[dict]:
    """
    Step 2 Alternative: Filter with priority for exit_reason='agent'.

    For mixed reward docker images:
    - Select 1 positive and 1 negative trajectory, prioritizing exit_reason='agent'

    For remaining docker images:
    - Select 500 positive and 500 negative trajectories, prioritizing exit_reason='agent'
    """
    logger.info("Step 2: Filtering with priority for exit_reason='agent'...")

    # Group trajectories by docker image and reward
    docker_trajectories = {}  # {docker_image: {0: [trajs], 1: [trajs]}}

    for traj in all_trajectories:
        docker_img = traj["docker_image"]
        reward = traj["reward"]

        if docker_img not in docker_trajectories:
            docker_trajectories[docker_img] = {0: [], 1: []}

        if reward in [0, 1]:
            docker_trajectories[docker_img][reward].append(traj)

    logger.info(f"Total unique docker images: {len(docker_trajectories)}")

    # Separate mixed reward and single reward docker images
    mixed_reward_dockers = {}
    single_reward_dockers = {0: [], 1: []}  # Lists of (docker_img, trajectories)

    for docker_img, reward_trajs in docker_trajectories.items():
        has_positive = len(reward_trajs[1]) > 0
        has_negative = len(reward_trajs[0]) > 0

        if has_positive and has_negative:
            mixed_reward_dockers[docker_img] = reward_trajs
        elif has_positive:
            single_reward_dockers[1].append((docker_img, reward_trajs[1]))
        elif has_negative:
            single_reward_dockers[0].append((docker_img, reward_trajs[0]))

    logger.info(f"Docker images with BOTH + and - rewards: {len(mixed_reward_dockers)}")
    logger.info(f"Docker images with only positive rewards: {len(single_reward_dockers[1])}")
    logger.info(f"Docker images with only negative rewards: {len(single_reward_dockers[0])}")

    filtered_trajectories = []

    # Helper function to sort trajectories with agent priority
    def get_exit_reason(traj):
        try:
            exit_reason = traj["data"].get("exit_reason", "")
            return (exit_reason != "agent", exit_reason)  # False comes first, so 'agent' is prioritized
        except Exception:
            return (True, "")

    # Process mixed reward docker images first
    for docker_img, reward_trajs in mixed_reward_dockers.items():
        # Sort by exit_reason priority (agent first)
        positive_sorted = sorted(reward_trajs[1], key=get_exit_reason)
        negative_sorted = sorted(reward_trajs[0], key=get_exit_reason)

        # Take first (highest priority) from each
        positive_traj = positive_sorted[0]
        negative_traj = negative_sorted[0]

        filtered_trajectories.append(positive_traj)
        filtered_trajectories.append(negative_traj)

    mixed_count = len(filtered_trajectories)
    logger.info(f"Selected {mixed_count} trajectories from mixed reward docker images")

    # Process remaining single reward docker images
    # Collect all single reward trajectories and sort by exit_reason priority
    all_positive_single = []
    all_negative_single = []

    for docker_img, trajs in single_reward_dockers[1]:
        all_positive_single.extend(trajs)

    for docker_img, trajs in single_reward_dockers[0]:
        all_negative_single.extend(trajs)

    # Sort by exit_reason priority
    all_positive_single.sort(key=get_exit_reason)
    all_negative_single.sort(key=get_exit_reason)

    # Take up to 500 of each
    N = 500
    additional_positive = all_positive_single[:N]
    additional_negative = all_negative_single[:N]

    filtered_trajectories.extend(additional_positive)
    filtered_trajectories.extend(additional_negative)

    # Log statistics
    total_count = len(filtered_trajectories)
    positive_count = sum(1 for t in filtered_trajectories if t["reward"] == 1)
    negative_count = sum(1 for t in filtered_trajectories if t["reward"] == 0)
    agent_count = sum(1 for t in filtered_trajectories if t["data"].get("exit_reason", "") == "agent")

    logger.info(f"Total filtered trajectories: {total_count}")
    logger.info(f"  From mixed reward docker images: {mixed_count}")
    logger.info(f"  Additional from single reward docker images: {total_count - mixed_count}")
    logger.info("Distribution:")
    logger.info(f"  Positive trajectories: {positive_count}")
    logger.info(f"  Negative trajectories: {negative_count}")
    logger.info(f"  Trajectories with exit_reason='agent': {agent_count}")

    return filtered_trajectories


def traj2verifier_data(
    json_entry: dict,
    system_prompt: str | None = None,
    instance_prompt: str | None = None,
    max_tokens: int = 65536,
) -> tuple[list[dict], bool]:
    """Convert a trajectory entry to verifier training data format."""
    try:
        # Extract trajectory data
        if "trajectory_steps" in json_entry:
            problem_statement = json_entry.get("problem_statement", "")
            traj = json_entry["trajectory_steps"]
            reward = json_entry.get("reward", 0)
        else:
            problem_statement = json_entry.get("problem_statement", "")
            traj = json_entry.get("trajectory", [])
            reward = json_entry.get("success", json_entry.get("reward", 0))

        # Create trajectory object for additional data with proper parsing
        try:
            import json

            trajclass = Trajectory.load_from_model_dump_json(json.dumps(json_entry))
            output_patch = trajclass.true_output_patch_only_existing_files
        except Exception as e:
            logger.warning(f"Could not get true_output_patch: {e}")
            try:
                trajclass = Trajectory.model_construct(**json_entry)
                output_patch = getattr(trajclass, "output_patch", "")
            except Exception:
                output_patch = json_entry.get("output_patch", "")

        # Default system prompt for verifier
        if system_prompt is None:
            system_prompt = """You are an expert judge evaluating AI assistant interactions. Your task is to determine if the assistant successfully resolved the user's request.

Key evaluation criteria:
1. Did the assistant complete the main task requested by the user?
2. Did the assistant handle all edge cases and requirements specified?
3. Were there any errors or issues in the final solution?
4. Did the assistant verify the solution works as intended?

Respond only with "<judgement>YES</judgement>" or "<judgement>NO</judgement>"."""

        # Default instance prompt
        if instance_prompt is None:
            instance_prompt = """You are a software engineer working on a repository. A user has submitted an issue, and you need to resolve it.

Repository Issue:
{problem_statement}

Please analyze the issue and implement a solution."""

        # Build the training data entry
        data_entry = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]

        # Create the user message with interaction log
        user_message = "Please evaluate the following interaction between an AI assistant and a user:\n\n"
        user_message += "=== INTERACTION LOG ===\n\n"
        user_message += f"[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n\n"
        user_message += f"[USER]\n{instance_prompt.format(problem_statement=problem_statement)}\n[/USER]"

        # Add trajectory steps
        for stepidx, entry in enumerate(traj):
            thought = entry.get("thought", "")
            action = entry.get("action", "")
            observation = entry.get("observation", "")

            assistant_response = f"{thought}\n\n{action}" if thought else action

            user_message += f"\n\n[STEP]\n{stepidx}\n[/STEP]"
            user_message += f"\n\n[ASSISTANT]\n{assistant_response}\n[/ASSISTANT]"
            user_message += f"\n\n[USER]\n{observation}\n[/USER]"

        # Add final patch and evaluation request
        user_message += "\n\n=== END INTERACTION ==="
        user_message += "\n\n=== FINAL PATCH ==="
        user_message += f"\n\n[PATCH]\n{output_patch}\n[/PATCH]"
        user_message += "\n\n=== END FINAL PATCH ==="
        user_message += "\n\nBased on the above interaction, did the assistant successfully resolve the user's initial request? Respond with YES or NO."

        data_entry.append({"role": "user", "content": user_message})
        data_entry.append(
            {
                "role": "assistant",
                "content": "<judgement>" + ("YES" if reward == 1 else "NO") + "</judgement>",
            }
        )

        # total_tokens = compute_total_tokens(data_entry)
        total_nonuser_tokens = compute_total_tokens([data_entry[0], data_entry[2]])
        data_entry[1]["content"] = condense_thoughts(
            data_entry[1]["content"],
            max_tokens=max_tokens - total_nonuser_tokens - 500,  ## 500 is just a buffer
        )

        # if total_tokens > max_tokens:
        #     return [], False
        return data_entry, True

    except Exception as e:
        logger.error(f"Error processing trajectory entry: {e}")
        return [], False


def process_trajectories_to_verifier_format(
    filtered_trajectories: list[dict],
    max_workers: int = 8,
    max_tokens: int = 72 * 1024,  # 72k tokens
) -> tuple[list[list[dict]], list[str], list[int], list[str], list[int], list[int]]:
    """Step 3: Process filtered trajectories to verifier training format."""
    logger.info("Step 3: Converting trajectories to verifier training format...")

    data = []
    docker_images = []
    rewards = []
    exp_names = []
    num_steps = []
    patch_sizes = []
    exit_reasons = []
    p2p_rates = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                traj2verifier_data,
                json_entry=traj["data"],
                max_tokens=max_tokens,
            ): traj
            for traj in filtered_trajectories
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing trajectories"):
            traj = futures[future]
            try:
                data_entry, success = future.result()
                if success:
                    data.append(data_entry)
                    docker_images.append(traj["docker_image"])
                    rewards.append(traj["reward"])
                    # print(traj.keys())
                    # exit_reasons.append(traj['data'].get('exit_reason', ''))
                    # trajclass = Trajectory.load_from_model_dump_json(json.dumps(traj['data']))
                    # p2p_rates.append(trajclass.p2p_count)
                    # print(trajclass.p2p_count)
                    # Try to get additional metadata with proper parsing
                    try:
                        if HAS_DEPENDENCIES:
                            import json

                            trajclass = Trajectory.load_from_model_dump_json(json.dumps(traj["data"]))
                            exp_names.append(getattr(trajclass, "exp_name", "unknown"))
                            num_steps.append(getattr(trajclass, "num_steps", len(traj["data"].get("trajectory_steps", []))))
                            patch_size = len(trajclass.true_output_patch)
                            patch_sizes.append(patch_size)
                            exit_reasons.append(traj["data"].get("exit_reason", ""))
                            p2p_rates.append(trajclass.p2p_count)
                        else:
                            exp_names.append("unknown")
                            num_steps.append(len(traj["data"].get("trajectory_steps", traj["data"].get("trajectory", []))))
                            patch_sizes.append(len(traj["data"].get("output_patch", "")))
                    except Exception:
                        exp_names.append("unknown")
                        num_steps.append(len(traj["data"].get("trajectory_steps", traj["data"].get("trajectory", []))))
                        patch_sizes.append(len(traj["data"].get("output_patch", "")))
            except Exception as e:
                logger.error(f"Error processing trajectory: {e}")

    logger.info(f"Successfully converted {len(data)} trajectories to verifier format")
    return data, docker_images, rewards, exp_names, num_steps, patch_sizes, exit_reasons, p2p_rates


def create_verifier_dataset(
    verifier_traj_dir: str = "verifier_traj",
    output_dataset_path: str = "./verifier_dataset_mixed_rewards",
    hub_repo_name: str | None = None,
    push_to_hub: bool = False,
    max_tokens: int = 72 * 1024,  # 72k tokens
    max_workers: int = 8,
    filter_method: str = "mixed_pairs",  # "mixed_pairs" or "agent_priority" or "none"
):
    """
    Create verifier dataset with configurable filtering.

    Args:
        filter_method: "mixed_pairs" for exactly 2 per docker image (default)
                      "agent_priority" for prioritizing exit_reason='agent'

    Clean 4-step process:
    1) Load all trajectories
    2) Filter based on selected method
    3) Convert to verifier format
    4) Push to HuggingFace
    """
    if not HAS_DEPENDENCIES:
        logger.error("Missing dependencies. Please install required packages.")
        return

    logger.info("Starting clean verifier dataset creation process...")

    if not os.path.exists(verifier_traj_dir):
        logger.error(f"Directory {verifier_traj_dir} does not exist")
        return

    # Step 1: Load all trajectories
    all_trajectories = load_all_trajectories(verifier_traj_dir)

    if not all_trajectories:
        logger.error("No trajectories loaded. Exiting.")
        return

    # Step 2: Filter based on selected method
    if filter_method == "mixed_pairs":
        filtered_trajectories = filter_to_mixed_reward_pairs(all_trajectories)
    elif filter_method == "agent_priority":
        filtered_trajectories = filter_with_agent_priority(all_trajectories)
    elif filter_method == "none":
        filtered_trajectories = all_trajectories
    else:
        logger.error(f"Unknown filter method: {filter_method}. Use 'mixed_pairs' or 'agent_priority'")
        return

    if not filtered_trajectories:
        logger.error("No trajectories after filtering. Exiting.")
        return

    # Step 3: Convert to verifier format
    data, docker_images, rewards, exp_names, num_steps, patch_sizes, exit_reasons, p2p_rates = process_trajectories_to_verifier_format(
        filtered_trajectories,
        max_workers=max_workers,
        max_tokens=max_tokens,
    )

    if not data:
        logger.error("No data entries after processing. Exiting.")
        return

    # Step 4: Create and push HuggingFace dataset
    logger.info("Step 4: Creating HuggingFace dataset...")

    dataset_dict = {
        "messages": data,
        "docker_images": docker_images,
        "rewards": rewards,
        "exp_names": exp_names,
        "num_steps": num_steps,
        "patch_sizes": patch_sizes,
        "exit_reasons": exit_reasons,
        "p2p_rates": p2p_rates,
    }

    dataset = datasets.Dataset.from_dict(dataset_dict)
    logger.info(f"Created dataset with {len(dataset)} entries")

    # Log dataset summary
    reward_counts = np.bincount(rewards)
    logger.info(f"Reward distribution: Negative={reward_counts[0]}, Positive={reward_counts[1] if len(reward_counts) > 1 else 0}")

    # Save dataset to disk
    dataset.save_to_disk(output_dataset_path)
    logger.info(f"Dataset saved to {output_dataset_path}")

    # Push to Hub if requested
    if push_to_hub:
        if not hub_repo_name:
            repo_name = "verifier_dataset_mixed_rewards"
        else:
            repo_name = hub_repo_name

        full_repo_name = f"r2e-edits/{repo_name}"
        logger.info(f"Pushing dataset to HuggingFace Hub: {full_repo_name}")

        try:
            dataset.push_to_hub(repo_id=full_repo_name, private=False)
            logger.info(f"✅ Successfully pushed dataset to {full_repo_name}")
        except Exception as e:
            logger.error(f"❌ Failed to push dataset to Hub: {e}")

    logger.info("🏁 Verifier dataset creation completed!")


if __name__ == "__main__":
    fire.Fire(
        {
            "create": create_verifier_dataset,
        }
    )
